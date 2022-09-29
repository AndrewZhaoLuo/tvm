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
# pylint: disable=invalid-name,arguments-differ,no-else-return,unused-argument,missing-docstring
"""
QNN pass transformation infrastructure.
"""
from operator import is_

import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import (
    DFPatternCallback,
    is_constant,
    is_expr,
    is_op,
    rewrite,
    wildcard,
)


def CanonicalizeOps():
    """Converts/Lowers an expression containing QNN ops to an expression containing only core
    (non-Dialect) Relay ops. Each QNN op is lowered to a sequence of existing Relay ops. This is a
    target-independent pass. One can register the lowering/transformation function for this op using
    FTVMQnnCanonicalize attr_name for FTVMLegalize op attribute.  An example of this transformation
    is below

    Examples
    ________

    .. code-block:: python

        # Original expression
        qnn_expr = relay.qnn.op.requantize(y,
                                           input_scale=1,
                                           input_zero_point=0,
                                           output_scale=1,
                                           output_zero_point=0,
                                           out_dtype='int8')

        # We want to utilize all the existing Relay infrastructure. So, instead of supporting this
        # QNN requantize op, we convert it into a sequence of existing Relay operators.
        mod = tvm.IRModule.from_expr(qnn_expr)
        mod = relay.qnn.transform.CanonicalizeOps()(mod)
        relay_expr = mod['main']
        print(relay_expr)

        def @main(%quantized_data: Tensor[(200), int32]) -> Tensor[(200), int8] {
          %0 = cast(%quantized_data, dtype="int64") /* ty=Tensor[(200), int64] */;
          %1 = multiply(%0, 2 /* ty=int64 */) /* ty=Tensor[(200), int64] */;
          %2 = multiply(%1, 1073741824 /* ty=int64 */) /* ty=Tensor[(200), int64] */;
          %3 = add(%2, 1073741824 /* ty=int64 */) /* ty=Tensor[(200), int64] */;
          %4 = right_shift(%3, 31 /* ty=int64 */) /* ty=Tensor[(200), int64] */;
          %5 = add(0 /* ty=int64 */, %4) /* ty=Tensor[(200), int64] */;
          %6 = clip(%5, a_min=-128f, a_max=127f) /* ty=Tensor[(200), int64] */;
          cast(%6, dtype="int8") /* ty=Tensor[(200), int8] */
        }

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass that canonicalizes QNN ops to Relay ops.
    """

    return relay.transform.Legalize("FTVMQnnCanonicalize")


def Legalize():
    """Legalizes QNN ops. As opposed to Relay Legalize, this one legalizes only QNN ops. One can
    register a transformation/legalization function for an op by using the FTVMQnnLegalize attr_name
    for FTVMLegalize op attribute. The isolation of QNN and Relay Legalize gives us separation of
    concerns, leading to a better software practice. The legalization can be configured to happen
    per target. An example of this type of legalization is shown below.

    Examples
    ________

    Suppose the original graph is as follows

            data(u8)  weight(u8)
                |       |
                |       |
               qnn.conv2d (int32)
                   |
                   |
                nn.relu (int32)

    Now, we know that Intel Cascade Lake has VNNI instructions to speedup convolution. However, it
    only works on u8 x i8 inputs. So, here, we can use QNN Legalize to transform the above graph as
    follows

            data(u8)  weight(u8)
               |          |
               |          |
               |     requantize(i8)
               |        |
               |        |
               qnn.conv2d (int32)
                   |
                   |
                 nn.relu (int32)

    In this legalization, since we have isolated legalization for QNN ops, it will only trigger the
    transformation for qnn.conv2d (and not nn.relu). This pass can be followed by CanonicalizeOps to
    further lower the qnn.requantize and qnn.conv2d into an expr containing only Relay ops.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass that legalizes QNN ops.
    """

    return relay.transform.Legalize("FTVMQnnLegalize")


class LayerNormQuantizedRewrite(DFPatternCallback):
    """
      A callback to rewrite the following operators into a single layer normalization operator.
      Note this is meant to be brittle, and more of a proof of concept.

      In the future, a better plan for doing this will be put into place

    %24 = qnn.dequantize(%23, 0.0235469f /* ty=float32 */, 0 /* ty=int32 */, axis=1) /* ty=Tensor[(1, 128, 768), float32] */;
    %25 = mean(%24, axis=[-1], keepdims=True) /* ty=Tensor[(1, 128, 1), float32] */;
    %26 = subtract(%24, %25) /* ty=Tensor[(1, 128, 768), float32] */;
    %27 = power(%26, 2f /* ty=float32 */) /* ty=Tensor[(1, 128, 768), float32] */;
    %28 = mean(%27, axis=[-1], keepdims=True) /* ty=Tensor[(1, 128, 1), float32] */;
    %29 = qnn.quantize(%28, 0.00230949f /* ty=float32 */, 0 /* ty=int32 */, out_dtype="int8", axis=1) /* ty=Tensor[(1, 128, 1), int8] */;
    %30 = qnn.quantize(1e-05f /* ty=float32 */, 7.87402e-08f /* ty=float32 */, 0 /* ty=int32 */, out_dtype="int8", axis=0) /* ty=int8 */;
    %31 = qnn.add(%29, %30, 0.00230949f /* ty=float32 */, 0 /* ty=int32 */, 7.87402e-08f /* ty=float32 */, 0 /* ty=int32 */, 0.00230957f /* ty=float32 */, 0 /* ty=int32 */, lhs_axis=1, rhs_axis=0) /* ty=Tensor[(1, 128, 1), int8] */;
    %32 = qnn.dequantize(%31, 0.00230957f /* ty=float32 */, 0 /* ty=int32 */, axis=1) /* ty=Tensor[(1, 128, 1), float32] */;
    %33 = sqrt(%32) /* ty=Tensor[(1, 128, 1), float32] */;
    %34 = divide(%26, %33) /* ty=Tensor[(1, 128, 768), float32] */;
    %35 = qnn.quantize(%34, 0.0240893f /* ty=float32 */, 0 /* ty=int32 */, out_dtype="int8", axis=1) /* ty=Tensor[(1, 128, 768), int8] */;
    %36 = qnn.mul(%35, meta[relay.Constant][28] /* ty=Tensor[(768), int8] */, 0.0240893f /* ty=float32 */, 0 /* ty=int32 */, 0.00785967f /* ty=float32 */, 0 /* ty=int32 */, 0.0221752f /* ty=float32 */, 0 /* ty=int32 */, lhs_axis=1, rhs_axis=0) /* ty=Tensor[(1, 128, 768), int8] */;
    %37 = qnn.add(%36, meta[relay.Constant][29] /* ty=Tensor[(768), int8] */, 0.0221752f /* ty=float32 */, 0 /* ty=int32 */, 0.00785718f /* ty=float32 */, 0 /* ty=int32 */, 0.0283614f /* ty=float32 */, 0 /* ty=int32 */, lhs_axis=1, rhs_axis=0) /* ty=Tensor[(1, 128, 768), int8] */;
    """

    def __init__(self):
        super(LayerNormQuantizedRewrite, self).__init__()
        self.data = wildcard()

        # input data qparams
        self.data_sf = wildcard()
        self.data_zp = wildcard()

        # TODO: pin important attribute of nodes
        e24 = is_op("qnn.dequantize")(self.data, self.data_sf, self.data_zp)
        e25 = is_op("mean")(e24)
        e26 = is_op("subtract")(e24, e25)
        e27 = is_op("power")(e26, wildcard())
        e28 = is_op("mean")(e27)
        e29 = is_op("qnn.quantize")(e28, wildcard(), wildcard())
        e31 = is_op("qnn.add")(
            e29,
            wildcard(),  # epsilon
            wildcard(),
            wildcard(),
            wildcard(),
            wildcard(),
            wildcard(),
            wildcard(),
        )
        e32 = is_op("qnn.dequantize")(e31, wildcard(), wildcard())
        e33 = is_op("sqrt")(e32)
        e34 = is_op("divide")(e26, e33)
        e35 = is_op("qnn.quantize")(e34, wildcard(), wildcard())

        self.gamma_quantized = wildcard()
        self.gamma_sf = wildcard()
        self.gamma_zp = wildcard()

        self.beta_quantized = wildcard()
        self.beta_sf = wildcard()
        self.beta_zp = wildcard()

        self.output_sf = wildcard()
        self.output_zp = wildcard()

        e36 = is_op("qnn.mul")(
            e35,
            self.gamma_quantized,
            wildcard(),
            wildcard(),
            self.gamma_sf,
            self.gamma_zp,
            wildcard(),
            wildcard(),
        )
        e37 = is_op("qnn.add")(
            e36,
            self.beta_quantized,
            wildcard(),
            wildcard(),
            self.beta_sf,
            self.beta_zp,
            self.output_sf,
            self.output_zp,
        )

        self.pattern = e37

        self.quantize_op = e35

        # self.pattern = e26

    def callback(self, pre, post, node_map):
        data = node_map[self.data][0]
        data_sf = node_map[self.data_sf][0]
        data_zp = node_map[self.data_zp][0]

        quantize_op = node_map[self.quantize_op][0]

        # return data

        gamma_quantized = node_map[self.gamma_quantized][0]
        gamma_sf = node_map[self.gamma_sf][0]
        gamma_zp = node_map[self.gamma_zp][0]

        beta_quantized = node_map[self.beta_quantized][0]
        beta_sf = node_map[self.beta_sf][0]
        beta_zp = node_map[self.beta_zp][0]

        output_sf = node_map[self.output_sf][0]
        output_zp = node_map[self.output_zp][0]

        gamma = relay.qnn.op.dequantize(gamma_quantized, gamma_sf, gamma_zp)
        gamma_mod = tvm.IRModule.from_expr(gamma)
        gamma = relay.transform.FoldConstantExpr(gamma, gamma_mod, True)

        beta = relay.qnn.op.dequantize(beta_quantized, beta_sf, beta_zp)
        beta_mod = tvm.IRModule.from_expr(beta)
        beta = relay.transform.FoldConstantExpr(beta, beta_mod, True)

        data_dq = relay.qnn.op.dequantize(data, data_sf, data_zp, axis=1)
        layer_norm_result = relay.op.nn.layer_norm(data_dq, gamma, beta)
        layer_norm_result_quantized = relay.qnn.op.quantize(
            layer_norm_result, output_sf, output_zp, axis=1, out_dtype=pre.checked_type.dtype
        )
        return layer_norm_result_quantized
