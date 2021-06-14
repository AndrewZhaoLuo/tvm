# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""TODO"""
import tvm
from tvm import relay

from ..op import register_mixed_precision_conversion

# Conversion types
MIXED_PRECISION_ALWAYS = 0
MIXED_PRECISION_FOLLOW = 1
MIXED_PRECISION_NEVER = 2


# Functions for FTVMMixedPrecisionConversionType which
# Take in CallNodes and a DType and returns a conversion type,
# an accumulation dtype, and an output_dtype.
def get_generic_dtypes(call_node, mixed_precision_type):
    # TODO: examine attributes
    if hasattr(call_node.attrs, "out_dtype"):
        return ["float32", mixed_precision_type]

    return [mixed_precision_type, mixed_precision_type]


def generic_always_op(call_node, mixed_precision_type):
    return [MIXED_PRECISION_ALWAYS] + get_generic_dtypes(call_node, mixed_precision_type)


def generic_follow_op(call_node, mixed_precision_type):
    return [MIXED_PRECISION_FOLLOW] + get_generic_dtypes(call_node, mixed_precision_type)


def generic_never_op(call_node, mixed_precision_type):
    return [MIXED_PRECISION_NEVER] + get_generic_dtypes(call_node, mixed_precision_type)


# Default lists inspired from TF's classifications:
# github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/core/grappler/optimizers/auto_mixed_precision_lists.h
# They have a bias toward Nvidia Tensor Cores so modify lists per your hardware choice.
DEFAULT_ALWAYS_LIST = [
    "nn.conv1d",
    "nn.conv2d",
    "nn.conv3d",
    "nn.conv1d_transpose",
    "nn.conv2d_transpose",
    "nn.conv3d_transpose",
    "nn.dense",
    # "nn.batch_matmul", # Handled by a special case
]
DEFAULT_FOLLOW_LIST = [
    # These ops add new data or change shape
    "nn.pad",
    "nn.batch_flatten",
    "concatenate",
    "zeros",
    "split",
    "squeeze",
    "transpose",
    "expand_dims",
    "reshape",
    "dyn.reshape",
    "broadcast_to_like",
    "dyn.broadcast_to",
    "strided_slice",
    "dyn.strided_slice",
    "take",
    "argwhere",
    "where",
    "tile",
    "dyn.tile",
    "scatter",
    "full",
    "dyn.full",
    # Comparison
    "less",
    "greater",
    "less_equal",
    "greater_equal",
    # By definition copy and cast will depend on inputs for output.
    "copy",
    "cast",
    "cast_like",
    # Simple arithmetic
    "add",
    "subtract",
    "multiply",
    "divide",
    "nn.bias_add",
    "nn.batch_norm",
    "sum",
    "mean",
    "sqrt",
    "shape_of",
    # Simple activations
    "max",
    "min",
    "maximum",
    "minimum",
    "nn.relu",
    "nn.leaky_relu",
    "nn.prelu",
    "nn.dropout",
    # Complicated activations which saturate in a narrow range
    "sigmoid",
    "tanh",
    # Pooling operations
    "nn.max_pool1d",
    "nn.max_pool2d",
    "nn.max_pool3d",
    "nn.avg_pool1d",
    "nn.avg_pool2d",
    "nn.avg_pool3d",
    # "nn.global_max_pool1d", # does not exist yet
    "nn.global_max_pool2d",
    # "nn.global_max_pool3d", # does not exist yet
    # "nn.global_avg_pool1d", # does not exist yet
    "nn.global_avg_pool2d",
    # "nn.global_avg_pool3d", # does not exist yet
    "nn.adaptive_max_pool1d",
    "nn.adaptive_max_pool2d",
    "nn.adaptive_max_pool3d",
    "nn.adaptive_avg_pool1d",
    "nn.adaptive_avg_pool2d",
    "nn.adaptive_avg_pool3d",
]
DEFAULT_NEVER_LIST = [
    # In general if |f(x)| >> |x| for expected inputs then put the op here.
    "exp",
    "power",
    "nn.cross_entropy",
    "nn.cross_entropy_with_logits",
    "nn.softmax",
    "nn.l2_normalize",
    # Error function doesn't seem to be able to be lowered into fp16 version in llvm.
    # Move to follow list when it does.
    "erf",
]


def register_default_mixed_precision_attributes():
    for list_of_ops, func in zip(
        [DEFAULT_ALWAYS_LIST, DEFAULT_FOLLOW_LIST, DEFAULT_NEVER_LIST],
        [generic_always_op, generic_follow_op, generic_never_op],
    ):
        for op_name in list_of_ops:
            register_mixed_precision_conversion(op_name, func=func)

    @register_mixed_precision_conversion("nn.batch_matmul")
    def nn_batch_matmul(call_node, mixed_precision_type):
        # TODO(AndrewZhaoLuo): remove when batch_matmul handles accumulation dtypes well.
        # Batched matmul has inconsistent support for mixed precision operations.
        # Many schedules ignore the out_dtype attribute which leads to errors when
        # input types do not match the out_dtype. Therefore, accumulate to output_dtype.
        return [MIXED_PRECISION_ALWAYS, "float16", "float16"]
