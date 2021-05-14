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
import enum
from typing import *

import tvm
from tvm import relay
from tvm.ir import IRModule
from tvm.relay.analysis import count_layers
from tvm.relay.expr_functor import Call, ExprVisitor
from tvm.relay.testing import resnet
from tvm.relay.transform import InferType
from tvm.relay.transform.fp16_conversion import graph_colors


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
        output_dtype_function: Callable[[relay.Call], str],
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
        For vars and constants, assume can cast to fp16 
        """
        if isinstance(arg, relay.Var) or isinstance(arg, relay.Constant):
            return True
        elif isinstance(arg, relay.Call):
            return (
                self.output_dtype_function(arg) == "fp16"
                and self.result_map[arg] == graph_colors.ConversionCategory.GREEN
            )
        elif isinstance(arg, relay.TupleGetItem):
            return self.is_fp16_compatible_arg(arg.tuple_value)
        else:
            raise ValueError(f"Unknown node type {type(arg)} for args")


class PrintVisitor(ExprVisitor):
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

        print(f"Operation {call.op} output dtype {output_dtype}")

        if call.op == relay.op.get("nn.batch_norm"):
            pass
        elif call.op == relay.op.get("nn.conv2d"):
            pass
        elif call.op == relay.op.get("nn.relu"):
            pass
        elif call.op == relay.op.get("add"):
            pass
        elif call.op == relay.op.get("nn.global_avg_pool2d"):
            pass
        elif call.op == relay.op.get("nn.batch_flatten"):
            pass
        elif call.op == relay.op.get("nn.dense"):
            pass
        elif call.op == relay.op.get("nn.bias_add"):
            pass
        elif call.op == relay.op.get("nn.softmax"):
            pass
        else:
            raise ValueError(f"Unknown call {call.op}")

        # print()
        # import pdb
        # pdb.set_trace()
        # print(call)


if __name__ == "__main__":
    c = resnet.get_net(1, 5, num_layers=18, image_shape=(1, 32, 32))

    infer_type_pass = InferType()

    mod = tvm.IRModule.from_expr(c)

    out = infer_type_pass(mod)
    relay_node_out = out["main"].body

    visitor = PrintVisitor()
    visitor.visit(relay_node_out)

    color_func = graph_colors.DefaultColorer()
    colorer = InitialGraphColorer(color_func)
    colorer.visit(relay_node_out)

    propagater = PropagateColors(colorer.result_map, lambda x: "float16")
    propagater.visit_call(relay_node_out)

    import pdb

    pdb.set_trace()
