
#include "fp32_to_fp16.h"

#include <tvm/ir/attrs.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/object.h>

#include "pattern_utils.h"

namespace std {
template <>
struct hash<tvm::DataType> {
  std::size_t operator()(tvm::DataType const& dtype) const {
    return dtype.code() * 3 + dtype.bits() * 5 + dtype.lanes() * 7;
  }
};
}  // namespace std

namespace tvm {
namespace relay {

// Only for pairs of std::hash-able types for simplicity.
// You can of course template this struct to allow other hash functions
struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& pair) const {
    auto h1 = std::hash<T1>()(pair.first);
    auto h2 = std::hash<T2>()(pair.second);

    return h1 ^ h2;
  }
};

// A map of call nodes to their fp16 conversion type
using CallColorMap = std::unordered_map<const CallNode*, FP16ConversionCategory>;
using CachedCastNodes = std::unordered_map<std::pair<const ExprNode*, DataType>, Expr, pair_hash>;
using ColorFunc = std::function<FP16ConversionCategory(const CallNode*)>;
using OutputDtypeFunc = std::function<FP16OpDType(const CallNode*)>;

class GraphColorer : public ExprVisitor {
 private:
  CallColorMap color_map;
  ColorFunc func;

  void VisitExpr_(const CallNode* l) final {
    // FP16ConversionCategory c = func(l);
    color_map[l] = func(l);
    ExprVisitor::VisitExpr_(l);
  }

 public:
  GraphColorer(ColorFunc func = DefaultFP16Colorer()) : func(func) {}

  CallColorMap result() { return color_map; }
};

class PropagateColors : public ExprVisitor {
 private:
  CallColorMap color_map;
  OutputDtypeFunc func;

  void VisitExpr_(const CallNode* l) final {
    ExprVisitor::VisitExpr_(l);
    auto result = color_map.find(l);
    if (result == color_map.end()) {
      LOG(FATAL) << "Unknown node not in initial color map!";
    }
    FP16ConversionCategory color = result->second;
    if (color != GRAY) return;

    for (Expr arg : l->args) {
      if (!is_fp16_compatible_arg(arg)) {
        color_map[l] = RED;
        return;
      }
    }

    color_map[l] = GREEN;
  }

  bool is_fp16_compatible_arg(Expr arg) {
    if (arg->IsInstance<VarNode>() || arg->IsInstance<ConstantNode>()) {
      return true;
    } else if (const CallNode* call = arg.as<CallNode>()) {
      auto result = color_map.find(call);
      if (result == color_map.end()) {
        LOG(FATAL) << "Unknown node not in initial color map!";
      }
      FP16ConversionCategory color = result->second;
      return color == GREEN && func(call).output_dtype == DataType::Float(16);
    } else if (const TupleGetItemNode* tuple_get_item = arg.as<TupleGetItemNode>()) {
      return is_fp16_compatible_arg(tuple_get_item->tuple);
    } else if (const TupleNode* tuple = arg.as<TupleNode>()) {
      for (Expr exp : tuple->fields) {
        if (!is_fp16_compatible_arg(exp)) {
          return false;
        }
      }
      return true;
    } else {
      LOG(FATAL) << "Unknown node not in initial color map!";
    }

    return true;
  }

 public:
  PropagateColors(CallColorMap initial_color_map, OutputDtypeFunc func = DefaultFP16OpDefinition())
      : color_map(initial_color_map), func(func) {}
  CallColorMap result() { return color_map; }
};

class RewriteBasedOnColors : public ExprMutator {
 private:
  CallColorMap color_map;
  OutputDtypeFunc output_func;
  CachedCastNodes cached_cast_nodes;

  Expr GetTypedExpr(const Expr& expr) {
    auto mod = IRModule::FromExpr(expr);
    mod = transform::InferType()(mod);
    if (expr.as<FunctionNode>()) {
      return mod->Lookup("main");
    } else {
      return mod->Lookup("main").as<FunctionNode>()->body;
    }
  }

  Expr cached_cast(Expr expr, DataType dtype, DataType wanted_dtype) {
    // If this is not a floating point type, do not cast. E.g. it might be an integer
    if (!dtype.is_float()) {
      return expr;
    }

    const ExprNode* expr_node = expr.as<ExprNode>();
    if (!expr_node) {
      LOG(FATAL) << "None expression node found in cast: " << expr;
    }

    auto search = cached_cast_nodes.find({expr_node, wanted_dtype});
    if (search != cached_cast_nodes.end()) {
      // Use cached result
      return search->second;
    }

    Expr result = dtype == wanted_dtype ? expr : Cast(expr, wanted_dtype);
    cached_cast_nodes[{expr_node, wanted_dtype}] = result;
    return result;
  }

  Expr cast_helper(Expr expr, Type t, DataType wanted_dtype) {
    if (const TensorTypeNode* tensor_type = t.as<TensorTypeNode>()) {
      return cached_cast(expr, tensor_type->dtype, wanted_dtype);
    } else if (const TupleTypeNode* tuple_type = t.as<TupleTypeNode>()) {
      Array<Expr> new_expr;
      for (int i = 0; i < (tuple_type->fields).size(); i++) {
        Expr tuple_expr_element = GetField(expr, i);
        Type tuple_expr_element_dtype = (tuple_type->fields)[i];
        new_expr.push_back(cast_helper(tuple_expr_element, tuple_expr_element_dtype, wanted_dtype));
      }
      return Tuple(new_expr);
    } else {
      LOG(FATAL) << "Unknown type " << t;
      return expr;
    }
  }

  Array<Expr> get_new_args(const CallNode* call, DataType arg_cast_datatype) {
    Array<Expr> ret;
    for (Expr arg : call->args) {
      arg = VisitExpr(arg);
      Type arg_type = GetTypedExpr(arg)->checked_type();
      Expr new_arg;
      if (arg->IsInstance<VarNode>() || arg->IsInstance<ConstantNode>()) {
        // Assume every var and const node is by default fp32, so cast if we are not casting to that
        new_arg = cast_helper(arg, arg_type, arg_cast_datatype);
      } else if (const CallNode* arg_call = arg.as<CallNode>()) {
        auto entry = color_map.find(arg_call);
        if (entry == color_map.end()) {
          LOG(FATAL) << "Unknown node not in initial color map!";
        }
        // FP16ConversionCategory color = entry->second;

        new_arg = cast_helper(arg, arg_type, arg_cast_datatype);
        /*
        // Cast result of a call, if we are going to rewrite it
        if (color == GREEN) {
          new_arg = output_func(arg_call).output_dtype != arg_cast_datatype
                        ? Cast(arg, arg_cast_datatype)
                        : arg;
        } else {
          // Was RED, assume fp32 output so cast to type
          new_arg = cast_helper(arg, arg_type, arg_cast_datatype);
        }
        */
      } else {
        // Else assume it's a composite type composed of cast elements
        new_arg = arg;
      }

      ret.push_back(new_arg);
    }

    return ret;
  }

  Attrs get_new_attrs(const CallNode* call, DataType accumulation_dtype) {
    Attrs new_attrs = Attrs(call->attrs);
    if (new_attrs.get() != nullptr) {
      // TODO: Figure out a better way to do this
      if (auto attrs = new_attrs.as<Conv1DAttrs>()) {
        modify_output_dtype(attrs, accumulation_dtype);
      } else if (auto attrs = new_attrs.as<Conv1DTransposeAttrs>()) {
        modify_output_dtype(attrs, accumulation_dtype);
      } else if (auto attrs = new_attrs.as<Conv2DAttrs>()) {
        modify_output_dtype(attrs, accumulation_dtype);
      } else if (auto attrs = new_attrs.as<Conv2DTransposeAttrs>()) {
        modify_output_dtype(attrs, accumulation_dtype);
      } else if (auto attrs = new_attrs.as<Conv2DWinogradAttrs>()) {
        modify_output_dtype(attrs, accumulation_dtype);
      } else if (auto attrs = new_attrs.as<Conv2DWinogradNNPACKWeightTransformAttrs>()) {
        modify_output_dtype(attrs, accumulation_dtype);
      } else if (auto attrs = new_attrs.as<DeformableConv2DAttrs>()) {
        modify_output_dtype(attrs, accumulation_dtype);
      } else if (auto attrs = new_attrs.as<Conv3DAttrs>()) {
        modify_output_dtype(attrs, accumulation_dtype);
      } else if (auto attrs = new_attrs.as<Conv3DTransposeAttrs>()) {
        modify_output_dtype(attrs, accumulation_dtype);
      } else if (auto attrs = new_attrs.as<Conv3DWinogradAttrs>()) {
        modify_output_dtype(attrs, accumulation_dtype);
      } else if (auto attrs = new_attrs.as<DenseAttrs>()) {
        modify_output_dtype(attrs, accumulation_dtype);
      } else if (auto attrs = new_attrs.as<BatchMatmulAttrs>()) {
        modify_output_dtype(attrs, accumulation_dtype);
      }

      if (auto attrs = new_attrs.as<InitOpAttrs>()) {
        modify_dtype(attrs, accumulation_dtype);
      }
    }

    return new_attrs;
  }

  template <typename T>
  void modify_output_dtype(const T* attrs, DataType accumulation_dtype) {
    // Helper template to modify relevant attributes with out_dtype type.
    // TODO: think about a better way to do this
    T* mutable_attrs = const_cast<T*>(attrs);
    mutable_attrs->out_dtype = accumulation_dtype;
  }

  template <typename T>
  void modify_dtype(const T* attrs, DataType accumulation_dtype) {
    // Helper template to modify relevant attributes with dtype type.
    // TODO: think about a better way to do this
    T* mutable_attrs = const_cast<T*>(attrs);
    mutable_attrs->dtype = accumulation_dtype;
  }

 public:
  RewriteBasedOnColors(CallColorMap color_map,
                       OutputDtypeFunc output_func = DefaultFP16OpDefinition())
      : color_map(color_map), output_func(output_func) {}

  Expr VisitExpr_(const LetNode* op) final {
    // First convert the bound value to FP16
    Expr value = this->Mutate(op->value);

    // Then rewrite the var type
    Var var = Downcast<Var>(this->Mutate(op->var));
    VarNode* mutable_var = const_cast<VarNode*>((op->var).as<VarNode>());
    mutable_var->type_annotation = GetTypedExpr(value)->checked_type();
    mutable_var->checked_type_ = mutable_var->type_annotation;

    // Mutate body last as it may depend on previous results
    Expr body = this->Mutate(op->body);

    return Let(var, value, body, op->span);
  }

  Expr VisitExpr_(const CallNode* call) final {
    auto result = color_map.find(call);
    if (result == color_map.end()) LOG(FATAL) << "Unknown node not in initial color map!";
    FP16ConversionCategory color = result->second;

    if (color == GRAY) {
      LOG(FATAL) << "Had gray colored node during rewrite! Make sure other passes color all nodes!";
    }

    Expr new_op = Mutate(call->op);
    FP16OpDType output_dtypes = output_func(call);

    // Create new node, ensure inputs are all fp32 if red, fp16 if green.
    // For sttrs we may overwrite the accumulation dtype field "output_dtype"
    // TODO: extend to bfloat types
    DataType arg_cast_dtype = color == GREEN ? DataType::Float(16) : DataType::Float(32);

    Array<Expr> new_args = get_new_args(call, arg_cast_dtype);
    Attrs new_attrs = get_new_attrs(call, output_dtypes.accumulation_dtype);
    Expr output = Call(new_op, new_args, new_attrs, call->type_args, call->span);
    color_map[output.as<CallNode>()] = color_map[call];

    if (output_dtypes.accumulation_dtype != output_dtypes.output_dtype) {
      output = Cast(output, output_dtypes.output_dtype);
      color_map[output.as<CallNode>()] = color_map[call];
    }

    return output;
  };

  Expr VisitExpr_(const FunctionNode* func) final {
    const_cast<FunctionNode*>(func)->ret_type = Type(nullptr);
    return ExprMutator::VisitExpr_(func);
  }
};

class ColorPrinter : public ExprVisitor {
 private:
  CallColorMap color_map;

 public:
  explicit ColorPrinter(CallColorMap& color_map) : color_map(color_map) {}
  explicit ColorPrinter() {}

  void VisitExpr_(const CallNode* l) final {
    ExprVisitor::VisitExpr_(l);
    std::cout << l->op << " is " << conversion_category_strings[color_map[l]] << std::endl;
  }
};

Expr RewriteFp16Graph(const Expr& expr, bool debug) {
  // Do an initial coloring based on each operation
  GraphColorer initial_colorer = GraphColorer();
  initial_colorer.VisitExpr(expr);
  CallColorMap color_map_initial = initial_colorer.result();

  if (debug) {
    std::cout << "Initial color map:" << std::endl;
    ColorPrinter(color_map_initial).VisitExpr(expr);
    std::cout << std::endl;
  }

  // Propagate colors so gray nodes in adjacent green regions are green
  // and those in red regions are red.
  PropagateColors propagate_colorer = PropagateColors(color_map_initial);
  propagate_colorer.VisitExpr(expr);
  CallColorMap color_map_final = propagate_colorer.result();

  if (debug) {
    std::cout << "Propagate color map:" << std::endl;
    ColorPrinter(color_map_final).VisitExpr(expr);
  }

  // Replace all green nodes with fp16 versions of the ops, inserting casts along way.
  RewriteBasedOnColors rewriter = RewriteBasedOnColors(color_map_final);

  // TODO: think about removing extraneous casts which can sometimes be added
  // (Usually interactions with non-Call nodes like Tuples)

  // Insert an extraneous cast to FP32 to match old module output
  Expr result = rewriter.Mutate(expr);

  // Old type annotations may no longer be accurate so rewrite
  if (const FunctionNode* func = result.as<FunctionNode>()) {
    const_cast<FunctionNode*>(func)->ret_type = Type(nullptr);
  }

  return result;
}

namespace transform {

Pass RewriteFP16(bool debug) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(RewriteFp16Graph(f, debug));
      };
  return CreateFunctionPass(pass_func, 10, "RewriteFp16", {});
}

TVM_REGISTER_GLOBAL("relay._transform.RewriteFP16").set_body_typed(RewriteFP16);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
