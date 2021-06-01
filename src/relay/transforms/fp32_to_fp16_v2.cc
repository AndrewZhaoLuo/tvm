
#include <tvm/ir/attrs.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/object.h>

#include "fp32_to_fp16.h"
#include "pattern_utils.h"

using ExprColorMap = std::unordered_map<const ExprNode*, FP16ConversionCategory>;
using ColorFunc = std::function<FP16ConversionCategory(const CallNode*)>;
using OutputDtypeFunc = std::function<FP16OpDType(const CallNode*)>;

class FP32ToFP16 : public ExprMutator {
 private:
  ExprColorMap color_map;
  OutputDtypeFunc output_func;
  ColorFunc color_func;

 public:
  RewriteBasedOnColors(ExprColorMap color_map,
                       OutputDtypeFunc output_func = DefaultFP16OpDefinition(),
                       ColorFunc color_func = DefaultFP16Colorer())
      : color_map(color_map), output_func(output_func), color_func(color_func) {}
  Expr VisitExpr_(const LetNode* op) final {
    // throw std::invalid_argument("Let nodes not supported for FP16 for now.");
    return ExprMutator::VisitExpr_(op);
  }

  Expr VisitExpr_(const CallNode* call) final {
    FP16ConversionCategory color = color_func(call);

    if (color == GREEN) {
      // set argument cast type to fp16
    } else if (color == GRAY) {
      // examine arguments
    } else if (color == RED) {

    } else if (color == FUNCTION) {
        
    } else {
      // error case
    }
  };

  Expr VisitExpr_(const VarNode* op) final {
    color_map[op] = GREEN;
    // TODO: mutate VarNode dtype to FP16
    return Var(op->vid, type, op->span)
  }

  Expr VisitExpr_(const ConstantNode* op) final {}
  Expr VisitExpr_(const GlobalVarNode* op) final;
  Expr VisitExpr_(const OpNode* op) final;
  Expr VisitExpr_(const TupleNode* op) final;
  Expr VisitExpr_(const FunctionNode* op) final;
  Expr VisitExpr_(const LetNode* op) final;
  Expr VisitExpr_(const IfNode* op) final;
  Expr VisitExpr_(const TupleGetItemNode* op) final;
  Expr VisitExpr_(const RefCreateNode* op) final;
  Expr VisitExpr_(const RefReadNode* op) final;
  Expr VisitExpr_(const RefWriteNode* op) final;
  Expr VisitExpr_(const ConstructorNode* op) final;
  Expr VisitExpr_(const MatchNode* op) final;

  Type VisitType(const Type& t) {
    if
      return t;
  }
};
