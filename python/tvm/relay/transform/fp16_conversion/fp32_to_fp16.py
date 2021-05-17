# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Relay type recasting pass"""
from typing import *

import numpy as np
import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprVisitor
from tvm.relay.testing import resnet
from tvm.relay.transform import InferType
from tvm.relay.transform.fp16_conversion import fp16_op_description, graph_colors


class InitialGraphColorer(ExprVisitor):
    """Color ops"""

    def __init__(self, color_function: Callable[[relay.Call], graph_colors.ConversionCategory]):
        super().__init__()
        self.color_function = color_function
        self.result_map = {}

    def visit_call(self, call: relay.Call):
        self.result_map[call] = self.color_function(call)
        super().visit_call(call)


class PropagateColors(ExprVisitor):
    """Propagate colors outward through gray colored nodes.

    A gray node becomes green if all it's inputs are fp16 or compile time constants (which can be cast at compile time).
    Otherwise the node will become red.
    """

    def __init__(
        self,
        result_map: Dict[relay.Call, graph_colors.ConversionCategory],
        output_dtype_function: Callable[[relay.Call], fp16_op_description.FP16OutDtype],
    ):
        super().__init__()
        self.result_map = result_map.copy()
        self.output_dtype_function = output_dtype_function

    def visit_call(self, call: relay.Call):
        super().visit_call(call)

        if self.result_map[call] != graph_colors.ConversionCategory.GRAY:
            return

        is_green = True
        for arg in call.args:
            is_green = is_green and self.is_fp16_compatible_arg(arg)

        self.result_map[call] = (
            graph_colors.ConversionCategory.GREEN
            if is_green
            else graph_colors.ConversionCategory.RED
        )

    def is_fp16_compatible_arg(self, arg: relay.Expr) -> bool:
        """
        For vars and constants, assume can cast to fp16 always and have constant folding
        """
        if isinstance(arg, relay.Var) or isinstance(arg, relay.Constant):
            return True
        elif isinstance(arg, relay.Call):
            return (
                self.output_dtype_function(arg).output_dtype == "float16"
                and self.result_map[arg] == graph_colors.ConversionCategory.GREEN
            )
        elif isinstance(arg, relay.TupleGetItem):
            return self.is_fp16_compatible_arg(arg.tuple_value)
        # TODO: propogate through other control flow
        else:
            raise ValueError(f"Unknown node type {type(arg)} for args")


class RewriteBasedOnColors(relay.ExprMutator):
    def __init__(
        self,
        result_map: Dict[relay.Call, graph_colors.ConversionCategory],
        fp16_dtype_func: Callable[[relay.Call], fp16_op_description.FP16OutDtype],
    ):
        super().__init__()
        self.result_map = result_map.copy()
        self.fp16_dtype_func = fp16_dtype_func

    def visit_call(self, call):
        if self.result_map[call] == graph_colors.ConversionCategory.GRAY:
            raise ValueError("Rewriting encountered gray! Remember to run PropagateColors pass!")
        elif self.result_map[call] == graph_colors.ConversionCategory.RED:
            return super().visit_call(call)

        call_op = self.visit(call.op)
        args = [self.visit(arg) for arg in call.args]
        new_args = []
        for arg in args:
            if isinstance(arg, relay.Var) or isinstance(arg, relay.Constant):
                new_args.append(relay.cast(arg, "float16"))
            elif isinstance(arg, relay.Call):
                if self.result_map[arg] == graph_colors.ConversionCategory.GREEN:
                    arg = (
                        arg
                        if self.fp16_dtype_func(arg).output_dtype == "float16"
                        else relay.cast(arg, "float16")
                    )
                else:
                    arg = relay.cast(arg, "float16")
                new_args.append(arg)
            else:
                new_args.append(arg)

        # TODO: what do we do about operations without control over the accumulation dtype?
        fp16_op_output = self.fp16_dtype_func(call)

        if call.attrs is not None and "out_dtype" in call.attrs.keys():
            new_attr_dict = {}
            for attr in call.attrs.keys():
                attr_value = call.attrs[attr]
                if isinstance(attr_value, tvm.ir.container.Array):
                    attr_value = tuple(attr_value)
                new_attr_dict[str(attr)] = attr_value
            new_attr_dict["out_dtype"] = fp16_op_output.accumulation_dtype
            attr_type = str(call.attrs).split("(")[0]
            new_attrs = tvm.ir.make_node(attr_type, **new_attr_dict)
        else:
            new_attrs = call.attrs

        # Inject proper arg types here based on fp16 op description func
        output = relay.Call(call_op, new_args, new_attrs)

        if fp16_op_output.accumulation_dtype != fp16_op_output.output_dtype:
            output = relay.cast(output, fp16_op_output.output_dtype)

        self.result_map[output] = self.result_map[call]
        return output


class PrintVisitor(ExprVisitor):
    def __init__(self, result_map: Dict[relay.Call, graph_colors.ConversionCategory]):
        super().__init__()
        self.result_map = result_map.copy()

    def visit_call(self, call):
        super().visit_call(call)

        if call.checked_type == None:
            raise ValueError(
                "Warning! Could not infer type for f{call.op} operation. Did you run InferType pass?"
            )

        if isinstance(call.checked_type, tvm.ir.tensor_type.TensorType):
            # Assume this refers to the output tensor
            output_dtype = call.checked_type.dtype
        elif isinstance(call.checked_type, tvm.ir.type.TupleType):
            output_dtype = call.checked_type.fields[0].dtype
        else:
            raise ValueError(f"Unknown type {type(call.checked_type)}")

        print(f"Operation {call.op} output dtype {output_dtype}, color {self.result_map[call]}")


def quantize_to_fp16(body: relay.Expr, debug: bool = False) -> relay.Expr:
    mod = tvm.ir.IRModule.from_expr(body)

    infer_type_pass = InferType()
    out = infer_type_pass(mod)
    body_typed = out["main"].body

    color_func = graph_colors.DefaultColorer()
    colorer = InitialGraphColorer(color_func)
    colorer.visit(body_typed)

    if debug:
        print("Initial color")
        visitor = PrintVisitor(colorer.result_map)
        visitor.visit(body_typed)

    fp16_op_descriptor = fp16_op_description.DefaultFP16TypeDefinition()
    propagater = PropagateColors(colorer.result_map, fp16_op_descriptor)
    propagater.visit_call(body_typed)

    if debug:
        print()
        print("After propogate")
        visitor = PrintVisitor(propagater.result_map)
        visitor.visit(body_typed)

    rewriter = RewriteBasedOnColors(propagater.result_map, fp16_op_descriptor)
    out = rewriter.visit_call(body_typed)

    return out


def run_module(mod, mod_params):
    dev = tvm.device("llvm", 0)
    intrp = relay.create_executor("debug", mod, device=dev, target="llvm")
    # in_data = [tvm.nd.array(value) for value in in_data.values()]
    return intrp.evaluate()(**mod_params).asnumpy()


def test_resnet18():
    mod, mod_params = resnet.get_workload(1, 5, num_layers=18, image_shape=(1, 32, 32))
    mod_params["data"] = np.random.uniform(-10, 10, (1, 1, 32, 32)).astype("float32")

    results = run_module(mod, mod_params)
    print(results)
    print(results.shape)
    print()

    output = quantize_to_fp16(mod["main"].body)

    fp16_mod = tvm.ir.IRModule.from_expr(output)
    results = run_module(fp16_mod, mod_params)
    print(results)
    print(results.shape)
    print()


def test_conv2d():
    data = relay.var("data", dtype="float32", shape=(1, 3, 32, 32))
    weight = relay.var("weight", dtype="float32", shape=(5, 3, 1, 1))
    out = relay.nn.conv2d(
        data, weight, data_layout="NCHW", kernel_layout="OIHW", out_dtype="float32"
    )

    mod = tvm.ir.IRModule.from_expr(out)
    mod_params = {
        "data": np.random.uniform(-10, 10, (1, 3, 32, 32)).astype("float32"),
        "weight": np.random.uniform(-1, 1, (5, 3, 1, 1)).astype("float32"),
    }
    print(run_module(mod, mod_params).shape)

    output_mod = tvm.ir.IRModule.from_expr(quantize_to_fp16(out))

    print(run_module(output_mod, mod_params).shape)


if __name__ == "__main__":
    test_resnet18()
