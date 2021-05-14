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


class ConversionCategory(enum.Enum):
    """
    Green: always worth casting
    """

    GREEN = "Green"
    GRAY = "Gray"
    RED = "Red"


class DefaultColorer:
    # Default lists inspired from TF's classifications:
    # https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/core/grappler/optimizers/auto_mixed_precision_lists.h
    # They might have a bias toward NVidia's Tensor Cores so be aware and modify lists per your hardware choice.

    # These should always be done in fp16 if possible
    DEFAULT_GREEN_LIST = {
        # 
        "nn.conv1d",
        "nn.conv2d",
        "nn.conv3d",
        "nn.conv1d_transpose",
        "nn.conv2d_transpose",
        "nn.conv3d_transpose",
        "nn.dense",
    }

    # These can be done in fp16 or fp32 with no point in casting between
    DEFAULT_GRAY_LIST = {
        # These ops add new data or change shape
        "nn.pad",
        "nn.batch_flatten",
        # Simple arithmetic
        "add",
        "nn.bias_add",
        "nn.batch_norm",
        # Simple activations
        "nn.relu",
        "nn.leaky_relu",
        "nn.prelu",
        "nn.dropout",
        # Pooling operations
        "nn.max_pool1d",
        "nn.max_pool2d",
        "nn.max_pool3d",
        "nn.avg_pool1d",
        "nn.avg_pool2d",
        "nn.avg_pool3d",
        ## "nn.global_max_pool1d", # does not exist
        "nn.global_max_pool2d",
        ## "nn.global_max_pool3d", # does not exist
        ## "nn.global_avg_pool1d", # does not exist
        "nn.global_avg_pool2d",
        ## "nn.global_avg_pool3d", # does not exist
        "nn.adaptive_max_pool1d",
        "nn.adaptive_max_pool2d",
        "nn.adaptive_max_pool3d",
        "nn.adaptive_avg_pool1d",
        "nn.adaptive_avg_pool2d",
        "nn.adaptive_avg_pool3d",
    }

    # These should always be done in fp32
    DEFAULT_RED_LIST = {
        # Activations with exponents or division
        "nn.cross_entropy",
        "nn.cross_entropy_with_logits",
        "nn.softmax",
        # Other
        "nn.l2_normalize",
    }

    def __init__(
        self,
        green_list: List[str] = DEFAULT_GREEN_LIST,
        gray_list: List[str] = DEFAULT_GRAY_LIST,
        red_list: List[str] = DEFAULT_RED_LIST,
    ):
        # Convert each list to entry
        green_list = self.create_op_list(green_list)
        gray_list = self.create_op_list(gray_list)
        red_list = self.create_op_list(red_list)

        # Create lookup table mapping relay op -> color in grpah
        self.lookup_table = {}
        for op_list, val in [
            (green_list, ConversionCategory.GREEN),
            (gray_list, ConversionCategory.GRAY),
            (red_list, ConversionCategory.RED),
        ]:
            for op in op_list:
                self.lookup_table[op] = val

    def __call__(self, call_node: relay.Call, ignore_missing: bool = False) -> ConversionCategory:
        if call_node.op not in self.lookup_table:
            if ignore_missing:
                return ConversionCategory.RED
            else:
                raise ValueError(f"Unknown op {call_node.op}")

        return self.lookup_table[call_node.op]

    @staticmethod
    def create_op_list(op_list: List[str]) -> List[tvm.ir.Op]:
        return [relay.op.get(op_name) for op_name in op_list]


class InitialGraphColorer(ExprVisitor):
    """Cast operations to the target type."""

    def __init__(self, color_function: Callable[[relay.Call], ConversionCategory]):
        super().__init__()
        self.color_function = color_function
        self.color_of = {}

    def visit_call(self, call: relay.Call):
        self.color_of[call] = self.color_function(call)
        super().visit_call(call)


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

    color_func = DefaultColorer()
    colorer = InitialGraphColorer(color_func)
    colorer.visit(relay_node_out) 

    import pdb 
    pdb.set_trace()
