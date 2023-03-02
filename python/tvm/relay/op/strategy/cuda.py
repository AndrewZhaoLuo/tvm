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
"""Definition of CUDA/GPU operator strategy."""
# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
from tvm import topi
from tvm.auto_scheduler import is_auto_scheduler_enabled
from tvm.contrib import nvcc
from tvm.contrib.thrust import can_use_thrust
from tvm.meta_schedule import is_meta_schedule_enabled
from tvm.te import SpecializedCondition

from ....target import Target
from ....tir import IntImm
from .. import op as _op
from .generic import *


@schedule_injective.register(["cuda", "gpu"])
def schedule_injective_cuda(attrs, outs, target):
    """schedule injective ops for cuda"""
    with target:
        return topi.cuda.schedule_injective(outs)


@schedule_reduce.register(["cuda", "gpu"])
def schedule_reduce_cuda(attrs, outs, target):
    """schedule reduction ops for cuda"""
    with target:
        return topi.cuda.schedule_reduce(outs)


@concatenate_strategy.register(["cuda", "gpu"])
def concatenate_strategy_cuda(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_concat(topi.transform.concatenate),
        wrap_topi_schedule(topi.cuda.schedule_injective),
        name="concatenate.cuda",
    )
    return strategy


@schedule_pool.register(["cuda", "gpu"])
def schedule_pool_cuda(attrs, outs, target):
    """schedule pooling ops for cuda"""
    with target:
        return topi.cuda.schedule_pool(outs, attrs.layout)


@schedule_pool_grad.register(["cuda", "gpu"])
def schedule_pool_grad_cuda(attrs, outs, target):
    """schedule pooling gradient ops for cuda"""
    with target:
        return topi.cuda.schedule_pool_grad(outs)


@schedule_adaptive_pool.register(["cuda", "gpu"])
def schedule_adaptive_pool_cuda(attrs, outs, target):
    """schedule adaptive pooling ops for cuda"""
    with target:
        return topi.cuda.schedule_adaptive_pool(outs, attrs.layout)


@softmax_strategy.register(["cuda", "gpu"])
def softmax_strategy_cuda(attrs, inputs, out_type, target):
    """softmax cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_softmax(topi.nn.softmax),
        wrap_topi_schedule(topi.cuda.schedule_softmax),
        name="softmax.cuda",
    )
    if target.kind.name == "cuda" and "cudnn" in target.libs:
        strategy.add_implementation(
            wrap_compute_softmax(topi.cuda.softmax_cudnn),
            wrap_topi_schedule(topi.cuda.schedule_softmax_cudnn),
            name="softmax.cudnn",
            plevel=15,
        )
    return strategy


@fast_softmax_strategy.register(["cuda", "gpu"])
def fast_softmax_strategy_cuda(attrs, inputs, out_type, target):
    """fast_softmax cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_softmax(topi.nn.fast_softmax),
        wrap_topi_schedule(topi.cuda.schedule_softmax),
        name="fast_softmax.cuda",
    )
    return strategy


@log_softmax_strategy.register(["cuda", "gpu"])
def log_softmax_strategy_cuda(attrs, inputs, out_type, target):
    """log_softmax cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_softmax(topi.nn.log_softmax),
        wrap_topi_schedule(topi.cuda.schedule_softmax),
        name="log_softmax.cuda",
    )
    if target.kind.name == "cuda" and "cudnn" in target.libs:
        strategy.add_implementation(
            wrap_compute_softmax(topi.cuda.log_softmax_cudnn),
            wrap_topi_schedule(topi.cuda.schedule_log_softmax_cudnn),
            name="log_softmax.cudnn",
            plevel=15,
        )
    return strategy


@schedule_lrn.register(["cuda", "gpu"])
def schedule_lrn_cuda(attrs, outs, target):
    """schedule LRN for cuda"""
    with target:
        return topi.cuda.schedule_lrn(outs)


@conv2d_strategy.register(["cuda", "gpu"])
def conv2d_strategy_cuda(attrs, inputs, out_type, target):
    """conv2d cuda strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    stride_h, stride_w = attrs.get_int_tuple("strides")
    dilation_h, dilation_w = attrs.get_int_tuple("dilation")
    padding = attrs.get_int_tuple("padding")
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")
    if groups == 1:
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            if (
                (target.kind.name in ["cuda", "vulkan", "rocm"])
                and data.dtype in ("int8", "uint8")
                and kernel.dtype in ("int8", "uint8")
            ):
                assert data.dtype == kernel.dtype
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.conv2d_nchw_int8),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_nchw_int8),
                    name="conv2d_nchw_int8.cuda",
                )
            else:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.conv2d_nchw),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_nchw),
                    name="conv2d_nchw.cuda",
                )
            N, _, H, W = get_const_tuple(data.shape)
            CO, CI, KH, KW = get_const_tuple(kernel.shape)
            (_, _, judge_winograd_auto_scheduler) = judge_winograd(
                N,
                H,
                W,
                KH,
                KW,
                CI,
                CO,
                padding,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                data.dtype,
                kernel.dtype,
                pre_flag=False,
            )
            if is_meta_schedule_enabled() and judge_winograd_auto_scheduler:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.nn.conv2d_winograd_nchw),
                    naive_schedule,  # this implementation should never be picked by autotvm
                    name="conv2d_nchw_winograd.cuda",
                    plevel=15,
                )
            elif (
                (2 < KH < 8 and 2 < KW < 8 and KH == KW)
                and (stride_h == 1 and stride_w == 1)
                and (dilation_h == 1 and dilation_w == 1)
            ):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.conv2d_nchw_winograd),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_nchw_winograd),
                    name="conv2d_nchw_winograd.cuda",
                    plevel=5,
                )
        elif layout == "HWCN":
            assert kernel_layout == "HWIO"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.conv2d_hwcn),
                wrap_topi_schedule(topi.cuda.schedule_conv2d_hwcn),
                name="conv2d_hwcn.cuda",
            )
        elif layout == "NHWC" and kernel_layout == "HWIO":
            strategy.add_implementation(
                wrap_compute_conv2d(topi.gpu.conv2d_nhwc),
                wrap_topi_schedule(topi.gpu.schedule_conv2d_nhwc),
                name="conv2d_nhwc.gpu",
            )

            N, H, W, _ = get_const_tuple(data.shape)
            KH, KW, CI, CO = get_const_tuple(kernel.shape)
            # Winograd shape related judgment
            (
                judge_winograd_tensorcore,
                judge_winograd_autotvm,
                judge_winograd_auto_scheduler,
            ) = judge_winograd(
                N,
                H,
                W,
                KH,
                KW,
                CI,
                CO,
                padding,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                data.dtype,
                kernel.dtype,
                pre_flag=False,
            )
            if judge_winograd_autotvm:
                if (
                    target.kind.name == "cuda"
                    and nvcc.have_tensorcore(target=target)
                    and judge_winograd_tensorcore
                ):
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.cuda.conv2d_nhwc_winograd_tensorcore),
                        wrap_topi_schedule(topi.cuda.schedule_conv2d_nhwc_winograd_tensorcore),
                        name="conv2d_nhwc_winograd_tensorcore.cuda",
                        plevel=5,
                    )
                else:
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.cuda.conv2d_nhwc_winograd_direct),
                        wrap_topi_schedule(topi.cuda.schedule_conv2d_nhwc_winograd_direct),
                        name="conv2d_nhwc_winograd_direct.cuda",
                        plevel=5,
                    )
            if (
                target.kind.name == "cuda"
                and not is_auto_scheduler_enabled()
                and not is_meta_schedule_enabled()
                and nvcc.have_tensorcore(target=target)
                and (
                    (N % 16 == 0 and CI % 16 == 0 and CO % 16 == 0)
                    or (N % 8 == 0 and CI % 16 == 0 and CO % 32 == 0)
                    or (N % 32 == 0 and CI % 16 == 0 and CO % 8 == 0)
                )
            ):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.conv2d_nhwc_tensorcore),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_nhwc_tensorcore),
                    name="conv2d_nhwc_tensorcore.cuda",
                    plevel=20,
                )

            # register auto-scheduler implementations
            if is_auto_scheduler_enabled() and judge_winograd_auto_scheduler:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.nn.conv2d_winograd_nhwc),
                    naive_schedule,  # this implementation should never be picked by autotvm
                    name="conv2d_nhwc.winograd",
                    plevel=15,
                )
            # register meta-schedule implementations
            if is_meta_schedule_enabled() and judge_winograd_auto_scheduler:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.nn.conv2d_winograd_nhwc),
                    naive_schedule,  # this implementation should never be picked by autotvm
                    name="conv2d_nhwc.winograd",
                    plevel=15,
                )

        elif layout == "HWNC":
            assert kernel_layout in ["HWOI", "HWOI16o16i", "HWOI8o32i", "HWOI32o16i"]
            _, _, N, in_channels = get_const_tuple(data.shape)
            pre_computed = len(kernel.shape) == 6
            if pre_computed:
                _, _, oc_chunk, _, oc_block_factor, _ = get_const_tuple(kernel.shape)
                out_channels = oc_chunk * oc_block_factor
            else:
                _, _, out_channels, _ = get_const_tuple(kernel.shape)

            tensorcore_dtypes = ["int4", "uint4", "int8", "uint8"]
            if (
                target.kind.name == "cuda"
                and nvcc.have_tensorcore(target=target)
                and kernel.dtype in tensorcore_dtypes
                and (
                    (
                        data.dtype in ["int4", "uint4"]
                        and N % 8 == 0
                        and in_channels % 32 == 0
                        and out_channels % 8 == 0
                    )
                    or (
                        data.dtype in ["int8", "uint8"]
                        and N % 8 == 0
                        and in_channels % 16 == 0
                        and out_channels % 32 == 0
                    )
                )
            ):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.conv2d_hwnc_tensorcore),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_hwnc_tensorcore),
                    name="conv2d_hwnc_tensorcore_direct.cuda",
                    plevel=20,
                )
            else:
                raise RuntimeError(
                    "Unsupported shape for conv2d HWNC.\
                                    Need to satisfy tensor core schedule."
                )
        elif (
            (target.kind.name in ["cuda", "vulkan", "rocm"])
            and layout == "NCHW4c"
            and data.dtype in ["int8", "uint8"]
        ):
            assert kernel_layout == "OIHW4o4i"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.conv2d_NCHWc_int8, need_data_layout=True),
                wrap_topi_schedule(topi.cuda.schedule_conv2d_NCHWc_int8),
                name="conv2d_NCHWc_int8.cuda",
            )
        elif is_auto_scheduler_enabled() or is_meta_schedule_enabled():
            strategy.add_implementation(
                wrap_compute_conv2d(
                    topi.nn.conv, need_data_layout=True, need_kernel_layout=True, has_groups=True
                ),
                naive_schedule,
                name="conv2d.cuda",
                plevel=15,
            )
        elif target.kind.name == "cuda" and "cudnn" not in target.libs:
            # No TVM native kernel applicable
            raise RuntimeError("Unsupported conv2d layout {} for CUDA".format(layout))

        if (
            target.kind.name == "cuda"
            and "cudnn" in target.libs
            and layout in ["NCHW", "NHWC"]
            and padding[0] == padding[2]
            and padding[1] == padding[3]
            and not (data.dtype in ["uint8", "int8"] or kernel.dtype in ["uint8", "int8"])
        ):
            # add cudnn implementation
            if layout == "NHWC":
                assert kernel_layout == "OHWI"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.conv2d_cudnn, need_data_layout=True, has_groups=True),
                wrap_topi_schedule(topi.cuda.schedule_conv2d_cudnn),
                name="conv2d_cudnn.cuda",
                plevel=25,
            )

    elif is_depthwise_conv2d(data.shape, layout, kernel.shape, kernel_layout, groups) and (
        layout == "NCHW" or "cudnn" not in target.libs
    ):  # cuDNN requires a different kernel layout for NHWC inputs.
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.depthwise_conv2d_nchw),
                wrap_topi_schedule(topi.cuda.schedule_depthwise_conv2d_nchw),
                name="depthwise_conv2d_nchw.cuda",
            )
        elif layout == "NHWC":
            assert kernel_layout == "HWOI"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.depthwise_conv2d_nhwc),
                wrap_topi_schedule(topi.cuda.schedule_depthwise_conv2d_nhwc),
                name="depthwise_conv2d_nhwc.cuda",
            )
        else:
            raise RuntimeError("Unsupported depthwise_conv2d layout {}".format(layout))
    else:  # group_conv2d
        # add cudnn implementation, if any
        cudnn_impl = False
        if target.kind.name == "cuda" and "cudnn" in target.libs:
            if (
                layout in ["NCHW", "NHWC"]
                and padding[0] == padding[2]
                and padding[1] == padding[3]
                and not (data.dtype in ["uint8", "int8"] or kernel.dtype in ["uint8", "int8"])
            ):
                strategy.add_implementation(
                    wrap_compute_conv2d(
                        topi.cuda.conv2d_cudnn, need_data_layout=True, has_groups=True
                    ),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_cudnn),
                    name="conv2d_cudnn.cuda",
                    plevel=25,
                )
                cudnn_impl = True

        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            _, channels, _, _ = get_const_tuple(data.shape)
            out_channels, in_channels, _, _ = get_const_tuple(kernel.shape)
            oc_chunk = out_channels // 4
            ic_chunk = in_channels // 4

            if (
                (target.kind.name in ["cuda", "vulkan", "rocm"])
                and data.dtype in ["int8", "uint8"]
                and kernel.dtype in ["int8", "uint8"]
                and channels % groups == 0
                and out_channels % groups == 0
                and channels % 4 == 0
                and out_channels % 4 == 0
                and groups <= oc_chunk
                and groups <= ic_chunk
            ):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.group_conv2d_nchw_int8, has_groups=True),
                    wrap_topi_schedule(topi.cuda.schedule_group_conv2d_nchw_int8),
                    name="group_conv2d_nchw_int8.cuda",
                )
            else:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.group_conv2d_nchw, has_groups=True),
                    wrap_topi_schedule(topi.cuda.schedule_group_conv2d_nchw),
                    name="group_conv2d_nchw.cuda",
                )
        elif layout == "NCHW4c" and data.dtype in ["int8", "uint8"]:
            assert kernel_layout == "OIHW4o4i"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.group_conv2d_NCHWc_int8, has_groups=True),
                wrap_topi_schedule(topi.cuda.schedule_group_conv2d_NCHWc_int8),
                name="group_conv2d_NCHWc_int8.cuda",
            )
        elif not cudnn_impl:
            raise RuntimeError("Unsupported group_conv2d layout {}".format(layout))
    return strategy


def judge_winograd(
    N,
    H,
    W,
    KH,
    KW,
    CI,
    CO,
    padding,
    stride_h,
    stride_w,
    dilation_h,
    dilation_w,
    data_dtype,
    kernel_dtype,
    pre_flag,
):
    """Winograd judgement about tensorcore and shape"""
    if H % 8 == 0:
        tile_size = 4
    else:
        tile_size = 2
    if pre_flag:
        alpha = KH
        KH = KW = alpha + 1 - tile_size
    pt, pl, pb, pr = topi.nn.get_pad_tuple(padding, (KH, KW))
    OH = (H + pt + pb - KH) // stride_h + 1
    OW = (W + pl + pr - KW) // stride_w + 1
    nH, nW = (OH + tile_size - 1) // tile_size, (OW + tile_size - 1) // tile_size
    if not isinstance(N, int):
        return False, False, False
    P = N * nH * nW

    judge_winograd_tensorcore = (
        (P % 16 == 0 and CI % 16 == 0 and CO % 16 == 0)
        or (P % 8 == 0 and CI % 16 == 0 and CO % 32 == 0)
        or (P % 32 == 0 and CI % 16 == 0 and CO % 8 == 0)
    )

    judge_winograd_autotvm = (
        2 < KH < 8
        and 2 < KW < 8
        and KH == KW
        and stride_h == 1
        and stride_w == 1
        and dilation_h == 1
        and dilation_w == 1
    )

    judge_winograd_auto_scheduler = (
        ("float" in data_dtype and "float" in kernel_dtype)
        and (KH == 3 and KW == 3)
        and (stride_h == 1 and stride_w == 1)
        and (dilation_h == 1 and dilation_w == 1)
    )

    return judge_winograd_tensorcore, judge_winograd_autotvm, judge_winograd_auto_scheduler


@conv2d_winograd_without_weight_transform_strategy.register(["cuda", "gpu"])
def conv2d_winograd_without_weight_transform_strategy_cuda(attrs, inputs, out_type, target):
    """conv2d_winograd_without_weight_transform cuda strategy"""
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    layout = attrs.data_layout
    data, kernel = inputs
    stride_h, stride_w = attrs.get_int_tuple("strides")
    padding = attrs.get_int_tuple("padding")
    assert dilation == (1, 1), "Do not support dilate now"
    assert groups == 1, "Do not support arbitrary group number"
    strategy = _op.OpStrategy()
    if layout == "NCHW":
        if is_meta_schedule_enabled():
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.conv2d_winograd_nchw_without_weight_transform),
                naive_schedule,  # this implementation should never be picked by autotvm
                name="conv2d_nchw_winograd_without_weight_transform",
                plevel=15,
            )
        else:
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.conv2d_nchw_winograd_without_weight_transform),
                wrap_topi_schedule(
                    topi.cuda.schedule_conv2d_nchw_winograd_without_weight_transform
                ),
                name="conv2d_nchw_winograd_without_weight_transform.cuda",
            )
    elif layout == "NHWC":
        N, H, W, _ = get_const_tuple(data.shape)
        alpha, _, CI, CO = get_const_tuple(kernel.shape)
        dilation_h, dilation_w = dilation
        judge_winograd_tensorcore, _, _ = judge_winograd(
            N,
            H,
            W,
            alpha,
            alpha,
            CI,
            CO,
            padding,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            data.dtype,
            kernel.dtype,
            pre_flag=True,
        )
        if (
            target.kind.name == "cuda"
            and nvcc.have_tensorcore(target=target)
            and judge_winograd_tensorcore
        ):
            strategy.add_implementation(
                wrap_compute_conv2d(
                    topi.cuda.conv2d_nhwc_winograd_tensorcore_without_weight_transform
                ),
                wrap_topi_schedule(
                    topi.cuda.schedule_conv2d_nhwc_winograd_tensorcore_without_weight_transform
                ),
                name="conv2d_nhwc_winograd_tensorcore_without_weight_transform.cuda",
            )
        else:
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.conv2d_nhwc_winograd_direct_without_weight_transform),
                wrap_topi_schedule(
                    topi.cuda.schedule_conv2d_nhwc_winograd_direct_without_weight_transform
                ),
                name="conv2d_nhwc_winograd_direct_without_weight_transform.cuda",
            )

        if is_auto_scheduler_enabled():
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.conv2d_winograd_nhwc_without_weight_transform),
                naive_schedule,  # this implementation should never be picked by autotvm
                name="conv2d_nhwc_winograd_without_weight_transform",
                plevel=15,
            )
        if is_meta_schedule_enabled():
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.conv2d_winograd_nhwc_without_weight_transform),
                naive_schedule,  # this implementation should never be picked by autotvm
                name="conv2d_nhwc_winograd_without_weight_transform",
                plevel=15,
            )
    else:
        raise RuntimeError(
            "Unsupported conv2d_winograd_without_weight_transform layout {}".format(layout)
        )
    return strategy


@deformable_conv2d_strategy.register(["cuda", "gpu"])
def deformable_conv2d_strategy_cuda(attrs, inputs, out_type, target):
    """deformable_conv2d cuda strategy"""
    layout = attrs.data_layout
    strategy = _op.OpStrategy()

    if layout == "NCHW":
        strategy.add_implementation(
            wrap_compute_deformable_conv2d(topi.cuda.deformable_conv2d_nchw),
            wrap_topi_schedule(topi.cuda.schedule_deformable_conv2d_nchw),
            name="deformable_conv2d_nchw.cuda",
        )
    elif layout == "NHWC":
        # This implementation should never be picked by autotvm
        strategy.add_implementation(
            wrap_compute_deformable_conv2d(topi.nn.deformable_conv2d_nhwc),
            naive_schedule,
            name="deformable_conv2d_nhwc.cuda",
        )
    else:
        raise RuntimeError("Layout %s is not supported in deformable conv2d on CUDA" % layout)
    return strategy


@conv2d_backward_weight_strategy.register(["cuda"])
def conv2d_backward_weight_strategy_cuda(attrs, inputs, out_type, target):
    """conv2d_backward_weight cuda strategy"""
    strategy = _op.OpStrategy()
    if target.kind.name == "cuda" and "cudnn" in target.libs:
        strategy.add_implementation(
            wrap_compute_conv2d_backward_weight(topi.cuda.conv2d_backward_weight_cudnn),
            wrap_topi_schedule(topi.generic.schedule_extern),
            name="conv2d_backward_weight_strategy.cudnn",
            plevel=15,
        )
    else:
        raise RuntimeError(
            "conv2d_backward_weight on cuda is currently only supported with cudnn. "
            "Please run Legalize pass to decompose this op into supported ops."
        )
    return strategy


@conv2d_transpose_strategy.register(["cuda", "gpu"])
def conv2d_transpose_strategy_cuda(attrs, inputs, out_type, target):
    """conv2d_transpose cuda strategy"""
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    assert dilation == (1, 1), "not support dilate now"
    strategy = _op.OpStrategy()
    num_strategies = 0

    if layout == "NCHW":
        strategy.add_implementation(
            wrap_compute_conv2d_transpose(topi.cuda.conv2d_transpose_nchw, has_groups=True),
            wrap_topi_schedule(topi.cuda.schedule_conv2d_transpose_nchw),
            name="conv2d_transpose_nchw.cuda",
        )
        num_strategies += 1

    if (
        target.kind.name == "cuda"
        and "cudnn" in target.libs
        and (
            (layout == "NCHW" and attrs.kernel_layout == "IOHW")
            or (layout == "NHWC" and attrs.kernel_layout == "IHWO")
        )
    ):
        strategy.add_implementation(
            wrap_compute_conv2d_transpose(
                topi.cuda.conv2d_transpose_cudnn, add_layout=True, has_groups=True
            ),
            wrap_topi_schedule(topi.generic.schedule_extern),
            name="conv2d_transpose.cudnn.cuda",
            plevel=25,
        )
        num_strategies += 1

    # TODO(masahi): Support conv2d_transpose NHWC for non-cudnn path.
    assert num_strategies > 0, "Unsupported conv2d_transpose workload, layout = %s, groups = %d" % (
        layout,
        groups,
    )
    return strategy


@conv3d_transpose_strategy.register(["cuda", "gpu"])
def conv3d_transpose_strategy_cuda(attrs, inputs, out_type, target):
    """conv3d_transpose cuda strategy"""
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    assert layout == "NCDHW", "only support ncdhw for now"
    assert dilation == (1, 1, 1), "not support dilate now"
    assert groups == 1, "only support groups == 1 for now"
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_conv3d_transpose(topi.cuda.conv3d_transpose_ncdhw),
        wrap_topi_schedule(topi.cuda.schedule_conv3d_transpose_ncdhw),
        name="conv3d_transpose_ncdhw.cuda",
    )
    return strategy


@conv3d_strategy.register(["cuda", "gpu"])
def conv3d_strategy_cuda(attrs, inputs, out_type, target):
    """conv3d cuda strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    layout = attrs.data_layout
    _, stride_h, stride_w = attrs.get_int_tuple("strides")
    _, dilation_h, dilation_w = attrs.get_int_tuple("dilation")
    assert layout in ["NCDHW", "NDHWC"], "Not support this layout {} yet".format(layout)
    if layout == "NCDHW":
        strategy.add_implementation(
            wrap_compute_conv3d(topi.cuda.conv3d_ncdhw),
            wrap_topi_schedule(topi.cuda.schedule_conv3d_ncdhw),
            name="conv3d_ncdhw.cuda",
            plevel=10,
        )
        _, _, _, kh, kw = get_const_tuple(kernel.shape)
        if (
            2 < kh < 8
            and 2 < kw < 8
            and kh == kw
            and stride_h == 1
            and stride_w == 1
            and dilation_h == 1
            and dilation_w == 1
            and attrs["groups"] == 1
        ):
            strategy.add_implementation(
                wrap_compute_conv3d(topi.cuda.conv3d_ncdhw_winograd),
                wrap_topi_schedule(topi.cuda.schedule_conv3d_ncdhw_winograd),
                name="conv3d_ncdhw_winograd.cuda",
                plevel=5,
            )
    else:  # layout == "NDHWC":
        strategy.add_implementation(
            wrap_compute_conv3d(topi.cuda.conv3d_ndhwc),
            wrap_topi_schedule(topi.cuda.schedule_conv3d_ndhwc),
            name="conv3d_ndhwc.cuda",
            plevel=10,
        )
        N, _, _, _, _ = get_const_tuple(data.shape)
        _, _, _, CI, CO = get_const_tuple(kernel.shape)
        if target.kind.name == "cuda":
            if nvcc.have_tensorcore(target=target):
                if (
                    (N % 16 == 0 and CI % 16 == 0 and CO % 16 == 0)
                    or (N % 8 == 0 and CI % 16 == 0 and CO % 32 == 0)
                    or (N % 32 == 0 and CI % 16 == 0 and CO % 8 == 0)
                ) and out_type == "float16":
                    strategy.add_implementation(
                        wrap_compute_conv3d(topi.cuda.conv3d_ndhwc_tensorcore),
                        wrap_topi_schedule(topi.cuda.schedule_conv3d_ndhwc_tensorcore),
                        name="conv3d_ndhwc_tensorcore.cuda",
                        plevel=20,
                    )

    if target.kind.name == "cuda" and "cudnn" in target.libs:
        strategy.add_implementation(
            wrap_compute_conv3d(topi.cuda.conv3d_cudnn, True),
            wrap_topi_schedule(topi.cuda.schedule_conv3d_cudnn),
            name="conv3d_cudnn.cuda",
            plevel=25,
        )
    return strategy


@conv3d_winograd_without_weight_transform_strategy.register(["cuda", "gpu"])
def conv3d_winograd_without_weight_transform_strategy_cuda(attrs, inputs, out_type, target):
    """conv3d_winograd_without_weight_transform cuda strategy"""
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    layout = attrs.data_layout
    assert dilation == (1, 1, 1), "Do not support dilate now"
    assert groups == 1, "Do not support arbitrary group number"
    strategy = _op.OpStrategy()
    if layout == "NCDHW":
        strategy.add_implementation(
            wrap_compute_conv3d(topi.cuda.conv3d_ncdhw_winograd_without_weight_transform),
            wrap_topi_schedule(topi.cuda.schedule_conv3d_ncdhw_winograd_without_weight_transform),
            name="conv3d_ncdhw_winograd_without_weight_transform.cuda",
        )
    else:
        raise RuntimeError(
            "Unsupported conv3d_winograd_without_weight_transform layout {}".format(layout)
        )
    return strategy


@conv1d_strategy.register(["cuda", "gpu"])
def conv1d_strategy_cuda(attrs, inputs, out_type, target):
    """conv1d cuda strategy"""
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    if dilation[0] < 1:
        raise ValueError("dilation should be a positive value")
    strategy = _op.OpStrategy()
    if attrs.groups == 1:
        if layout == "NCW":
            strategy.add_implementation(
                wrap_compute_conv1d(topi.cuda.conv1d_ncw),
                wrap_topi_schedule(topi.cuda.schedule_conv1d_ncw),
                name="conv1d_ncw.cuda",
            )
        elif layout == "NWC":
            strategy.add_implementation(
                wrap_compute_conv1d(topi.cuda.conv1d_nwc),
                wrap_topi_schedule(topi.cuda.schedule_conv1d_nwc),
                name="conv1d_nwc.cuda",
            )
        else:
            raise ValueError("Unsupported conv1d layout {}".format(layout))
    else:
        if layout == "NCW":
            strategy.add_implementation(
                wrap_compute_group_conv1d(topi.cuda.group_conv1d_ncw),
                wrap_topi_schedule(topi.cuda.schedule_group_conv1d_ncw),
                name="group_conv1d_ncw.cuda",
            )
        elif layout == "NWC":
            strategy.add_implementation(
                wrap_compute_group_conv1d(topi.cuda.group_conv1d_nwc),
                wrap_topi_schedule(topi.cuda.schedule_group_conv1d_nwc),
                name="group_conv1d_nwc.cuda",
            )
        else:
            raise ValueError("Unsupported conv1d layout {}".format(layout))
    return strategy


@conv1d_transpose_strategy.register(["cuda", "gpu"])
def conv1d_transpose_strategy_cuda(attrs, inputs, out_type, target):
    """conv1d_transpose cuda strategy"""
    strategy = _op.OpStrategy()
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    assert layout == "NCW", "conv1d_transpose ncw only supported"
    assert dilation == (1,), "conv1d_transpose dilation is not supported"
    assert groups == 1, "conv1d_transpose groups == 1 only supported"
    strategy.add_implementation(
        wrap_compute_conv1d_transpose(topi.cuda.conv1d_transpose_ncw),
        wrap_topi_schedule(topi.cuda.schedule_conv1d_transpose_ncw),
        name="conv1d_transpose_ncw.cuda",
    )
    return strategy


@matmul_strategy.register(["cuda", "gpu"])
def matmul_strategy_cuda(attrs, inputs, out_type, target):
    """Matmul cuda strategy."""
    strategy = _op.OpStrategy()

    if is_auto_scheduler_enabled():
        strategy.add_implementation(
            wrap_compute_matmul(topi.nn.matmul),
            naive_schedule,
            name="matmul.cuda",
        )
    elif is_meta_schedule_enabled():
        strategy.add_implementation(
            wrap_compute_matmul(topi.nn.matmul),
            naive_schedule,
            name="matmul.cuda",
        )
    else:
        logger.warning(
            "Matmul is not optimized for cuda. Recommend to use cublas for better performance."
        )
        # Temporary use this as a basic schedule
        strategy.add_implementation(
            wrap_compute_matmul(topi.gpu.matmul_default),
            wrap_topi_schedule(topi.gpu.schedule_matmul_default),
            name="matmul_default.gpu",
        )

    if target.kind.name == "cuda" and "cublas" in target.libs:
        strategy.add_implementation(
            wrap_compute_matmul(topi.cuda.matmul_cublas),
            wrap_topi_schedule(topi.cuda.schedule_matmul_cublas),
            name="matmul_cublas.cuda",
            plevel=25,
        )
    return strategy


@dense_strategy.register(["cuda", "gpu"])
def dense_strategy_cuda(attrs, inputs, out_type, target):
    """dense cuda strategy"""
    strategy = _op.OpStrategy()
    data, weights = inputs
    b, i = get_const_tuple(data.shape)
    o, _ = get_const_tuple(weights.shape)
    if (
        target.kind.name in ["cuda", "vulkan", "rocm"]
        and data.dtype == "int8"
        and weights.dtype == "int8"
        and out_type.dtype == "int32"
    ):
        strategy.add_implementation(
            wrap_compute_dense(topi.cuda.dense_int8),
            wrap_topi_schedule(topi.cuda.schedule_dense_int8),
            name="dense_int8.cuda",
        )
    else:
        # Some AMDGPU cards have accuracy issues with this schedule
        # See https://github.com/apache/tvm/issues/13666
        if target.kind.name != "rocm":
            strategy.add_implementation(
                wrap_compute_dense(topi.gpu.dense_small_batch),
                wrap_topi_schedule(topi.gpu.schedule_dense_small_batch),
                name="dense_small_batch.gpu",
            )

        with SpecializedCondition(target.kind.name == "rocm" or b >= 32):
            strategy.add_implementation(
                wrap_compute_dense(topi.gpu.dense_large_batch),
                wrap_topi_schedule(topi.gpu.schedule_dense_large_batch),
                name="dense_large_batch.gpu",
                plevel=5,
            )

    if target.kind.name == "cuda":
        if nvcc.have_tensorcore(target=target):
            if (
                (
                    data.dtype in ["float16", "int8", "uint8"]
                    and (
                        (i % 16 == 0 and b % 16 == 0 and o % 16 == 0)
                        or (i % 16 == 0 and b % 8 == 0 and o % 32 == 0)
                        or (i % 16 == 0 and b % 32 == 0 and o % 8 == 0)
                    )
                )
                or (data.dtype in ["int4", "uint4"] and i % 32 == 0 and b % 8 == 0 and o % 8 == 0)
                or (data.dtype in ["int1", "uint1"] and i % 128 == 0 and b % 8 == 0 and o % 8 == 0)
            ):
                strategy.add_implementation(
                    wrap_compute_dense(topi.cuda.dense_tensorcore),
                    wrap_topi_schedule(topi.cuda.schedule_dense_tensorcore),
                    name="dense_tensorcore.cuda",
                    plevel=20,
                )

    if target.kind.name == "cuda" and "cublas" in target.libs:
        strategy.add_implementation(
            wrap_compute_dense(topi.cuda.dense_cublas),
            wrap_topi_schedule(topi.cuda.schedule_dense_cublas),
            name="dense_cublas.cuda",
            plevel=25,
        )
    return strategy


@batch_matmul_strategy.register(["cuda", "gpu"])
def batch_matmul_strategy_cuda(attrs, inputs, out_type, target):
    """batch_matmul cuda strategy"""
    strategy = _op.OpStrategy()
    x, y = inputs
    if (
        x.dtype == "int8"
        and y.dtype == "int8"
        and out_type.dtype == "int32"
        and not attrs["transpose_a"]
        and attrs["transpose_b"]
    ):
        strategy.add_implementation(
            wrap_compute_batch_matmul(topi.cuda.batch_matmul_int8, need_out_dtype=True),
            wrap_topi_schedule(topi.cuda.schedule_batch_matmul_int8),
            name="batch_matmul_int8.cuda",
            plevel=10,
        )
    else:
        strategy.add_implementation(
            wrap_compute_batch_matmul(topi.cuda.batch_matmul, need_out_dtype=True),
            wrap_topi_schedule(topi.cuda.schedule_batch_matmul),
            name="batch_matmul.cuda",
            plevel=10,
        )
    if target.kind.name == "cuda" and "cublas" in target.libs:
        strategy.add_implementation(
            wrap_compute_batch_matmul(topi.cuda.batch_matmul_cublas, need_out_dtype=True),
            wrap_topi_schedule(topi.generic.schedule_extern),
            name="batch_matmul_cublas.cuda",
            plevel=30,
        )
    if (
        target.kind.name == "cuda"
        and nvcc.have_tensorcore(target=target)
        and not attrs["transpose_a"]
        and attrs["transpose_b"]
    ):
        x, y = inputs
        _, M, K = get_const_tuple(x.shape)
        _, N, K = get_const_tuple(y.shape)
        if (
            x.dtype in ["float16", "int8", "uint8"]
            and (
                (M % 8 == 0 and K % 16 == 0 and N % 32 == 0)
                or (M % 16 == 0 and K % 16 == 0 and N % 16 == 0)
                or (M % 32 == 0 and K % 16 == 0 and N % 8 == 0)
            )
        ) or (x.dtype in ["int4", "uint4"] and K % 32 == 0 and M % 8 == 0 and N % 8 == 0):
            strategy.add_implementation(
                wrap_compute_batch_matmul(topi.cuda.batch_matmul_tensorcore, need_out_dtype=True),
                wrap_topi_schedule(topi.cuda.schedule_batch_matmul_tensorcore),
                name="batch_matmul_tensorcore.cuda",
                plevel=20,
            )

    return strategy


@sparse_dense_strategy.register(["cuda", "gpu"])
def sparse_dense_strategy_cuda(attrs, inputs, out_type, target):
    """sparse dense cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_sparse_dense(topi.cuda.sparse_dense),
        wrap_topi_schedule(topi.cuda.schedule_sparse_dense),
        name="sparse_dense.cuda",
        plevel=10,
    )
    return strategy


@sparse_reshape_strategy.register(["cuda", "gpu"])
def sparse_reshape_strategy_cuda(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_sparse_reshape(topi.cuda.sparse_reshape),
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="sparse_reshape.cuda",
    )
    return strategy


@sparse_dense_padded_strategy.register(["cuda", "gpu", "rocm"])
def sparse_dense_padded_strategy_cuda(attrs, inputs, out_type, target):
    """sparse dense cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_sparse_dense(topi.cuda.sparse_dense_padded),
        wrap_topi_schedule(topi.cuda.schedule_sparse_dense_padded),
        name="sparse_dense_padded.cuda",
        plevel=10,
    )
    return strategy


@scatter_elements_strategy.register(["cuda", "gpu"])
def scatter_elements_cuda(attrs, inputs, out_type, target):
    """scatter elements cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_scatter_elements(topi.cuda.scatter_elements),
        wrap_topi_schedule(topi.cuda.schedule_extern),
        name="scatter_elements.cuda",
        plevel=10,
    )

    rank = len(inputs[0].shape)

    with SpecializedCondition(rank == 1 and attrs.reduction == "update"):
        if can_use_thrust(target, "tvm.contrib.thrust.stable_sort_by_key"):
            strategy.add_implementation(
                wrap_compute_scatter_elements(topi.cuda.scatter_via_sort),
                wrap_topi_schedule(topi.cuda.schedule_scatter_via_sort),
                name="scatter_via_sort.cuda",
                plevel=9,  # use the sequential version by default
            )
    return strategy


@scatter_nd_strategy.register(["cuda", "gpu"])
def scatter_nd_cuda(attrs, inputs, out_type, target):
    """scatter_nd cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_scatter_nd(topi.cuda.scatter_nd),
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="scatter_nd.cuda",
        plevel=10,
    )
    return strategy


@sort_strategy.register(["cuda", "gpu"])
def sort_strategy_cuda(attrs, inputs, out_type, target):
    """sort cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_sort(topi.cuda.sort),
        wrap_topi_schedule(topi.cuda.schedule_sort),
        name="sort.cuda",
    )
    if can_use_thrust(target, "tvm.contrib.thrust.sort"):
        strategy.add_implementation(
            wrap_compute_sort(topi.cuda.sort_thrust),
            wrap_topi_schedule(topi.cuda.schedule_sort),
            name="sort_thrust.cuda",
            plevel=15,
        )
    return strategy


@argsort_strategy.register(["cuda", "gpu"])
def argsort_strategy_cuda(attrs, inputs, out_type, target):
    """argsort cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_argsort(topi.cuda.argsort),
        wrap_topi_schedule(topi.cuda.schedule_argsort),
        name="argsort.cuda",
    )
    if can_use_thrust(target, "tvm.contrib.thrust.sort"):
        strategy.add_implementation(
            wrap_compute_argsort(topi.cuda.argsort_thrust),
            wrap_topi_schedule(topi.cuda.schedule_argsort),
            name="argsort_thrust.cuda",
            plevel=15,
        )
    return strategy


@topk_strategy.register(["cuda", "gpu"])
def topk_strategy_cuda(attrs, inputs, out_type, target):
    """topk cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_topk(topi.cuda.topk),
        wrap_topi_schedule(topi.cuda.schedule_topk),
        name="topk.cuda",
    )
    if can_use_thrust(target, "tvm.contrib.thrust.sort"):
        strategy.add_implementation(
            wrap_compute_topk(topi.cuda.topk_thrust),
            wrap_topi_schedule(topi.cuda.schedule_topk),
            name="topk_thrust.cuda",
            plevel=15,
        )
    return strategy


@searchsorted_strategy.register(["cuda", "gpu"])
def searchsorted_strategy_cuda(attrs, inputs, out_type, target):
    """searchsorted cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_searchsorted(topi.cuda.searchsorted),
        wrap_topi_schedule(topi.cuda.schedule_extern),
        name="searchsorted.cuda",
    )
    return strategy


@multibox_prior_strategy.register(["cuda", "gpu"])
def multibox_prior_strategy_cuda(attrs, inputs, out_type, target):
    """multibox_prior cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_multibox_prior(topi.cuda.multibox_prior),
        wrap_topi_schedule(topi.cuda.schedule_multibox_prior),
        name="multibox_prior.cuda",
    )
    return strategy


@multibox_transform_loc_strategy.register(["cuda", "gpu"])
def multibox_transform_loc_strategy_cuda(attrs, inputs, out_type, target):
    """multibox_transform_loc cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_multibox_transform_loc(topi.cuda.multibox_transform_loc),
        wrap_topi_schedule(topi.cuda.schedule_multibox_transform_loc),
        name="multibox_transform_loc.cuda",
    )
    return strategy


@get_valid_counts_strategy.register(["cuda", "gpu"])
def get_valid_counts_strategy_cuda(attrs, inputs, out_type, target):
    """get_valid_counts cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_get_valid_counts(topi.cuda.get_valid_counts),
        wrap_topi_schedule(topi.cuda.schedule_get_valid_counts),
        name="get_valid_counts.cuda",
    )
    return strategy


@nms_strategy.register(["cuda", "gpu"])
def nms_strategy_cuda(attrs, inputs, out_type, target):
    """nms cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_nms(topi.cuda.non_max_suppression),
        wrap_topi_schedule(topi.cuda.schedule_nms),
        name="nms.cuda",
    )
    return strategy


@all_class_nms_strategy.register(["cuda", "gpu"])
def all_class_nms_strategy_cuda(attrs, inputs, out_type, target):
    """all class nms cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_all_class_nms(topi.cuda.all_class_non_max_suppression),
        wrap_topi_schedule(topi.cuda.schedule_nms),
        name="all_class_nms.cuda",
    )
    return strategy


@roi_align_strategy.register(["cuda", "gpu"])
def roi_align_strategy_cuda(attrs, inputs, out_type, target):
    """roi_align cuda strategy"""
    strategy = _op.OpStrategy()
    layout = attrs.layout

    if layout == "NCHW":
        strategy.add_implementation(
            wrap_compute_roi_align(topi.vision.rcnn.roi_align_nchw),
            wrap_topi_schedule(topi.cuda.schedule_roi_align),
            name="roi_align_nchw.cuda",
        )
    else:
        assert layout == "NHWC", "layout must be NCHW or NHWC."
        strategy.add_implementation(
            wrap_compute_roi_align(topi.vision.rcnn.roi_align_nhwc),
            wrap_topi_schedule(topi.cuda.schedule_roi_align),
            name="roi_align_nhwc.cuda",
        )
    return strategy


@schedule_roi_pool.register(["cuda", "gpu"])
def schedule_roi_pool_cuda(attrs, outs, target):
    """schedule roi_pool for cuda"""
    with target:
        return topi.cuda.schedule_roi_pool(outs)


@proposal_strategy.register(["cuda", "gpu"])
def proposal_strategy_cuda(attrs, inputs, out_type, target):
    """proposal cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_proposal(topi.cuda.proposal),
        wrap_topi_schedule(topi.cuda.schedule_proposal),
        name="proposal.cuda",
    )
    return strategy


@correlation_strategy.register(["cuda", "gpu"])
def correlation_strategy_cuda(attrs, inputs, out_type, target):
    """correlation cuda strategy"""
    layout = attrs.layout
    assert layout == "NCHW", "Only support NCHW layout"
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_correlation(topi.cuda.correlation_nchw),
        wrap_topi_schedule(topi.cuda.schedule_correlation_nchw),
        name="correlation.cuda",
    )
    return strategy


@argwhere_strategy.register(["cuda", "gpu"])
def argwhere_strategy_cuda(attrs, inputs, out_type, target):
    """argwhere cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_argwhere(topi.cuda.argwhere),
        wrap_topi_schedule(topi.cuda.schedule_argwhere),
        name="argwhere.cuda",
    )
    return strategy


@cumsum_strategy.register(["cuda", "gpu"])
def cumsum_strategy_cuda(attrs, inputs, out_type, target):
    """cumsum cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_scanop(topi.cuda.cumsum),
        wrap_topi_schedule(topi.cuda.schedule_scan),
        name="cumsum.cuda",
    )
    return strategy


@cumprod_strategy.register(["cuda", "gpu"])
def cumprod_strategy_cuda(attrs, inputs, out_type, target):
    """cumprod cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_scanop(topi.cuda.cumprod),
        wrap_topi_schedule(topi.cuda.schedule_scan),
        name="cumprod.cuda",
    )
    return strategy


@unique_strategy.register(["cuda", "gpu"])
def unique_strategy_cuda(attrs, inputs, out_type, target):
    """unique cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_unique(topi.cuda.unique),
        wrap_topi_schedule(topi.cuda.schedule_scan),
        name="unique.cuda",
    )
    return strategy


@schedule_transpose.register(["cuda", "gpu", "rocm"])
def schedule_transpose_cuda(attrs, outs, target):
    """
    Transpose cuda strategy
    Dispatches to and optimized schedule if the transpose is standalone (not fused).
    """
    warp_size = int(Target.current(allow_none=False).thread_warp_size)
    if (
        isinstance(outs[0].op.input_tensors[0].op, te.PlaceholderOp)
        and len(outs[0].shape) == 2
        and (attrs.axes is None or (len(attrs.axes) == 2 and attrs.axes == [1, 0]))
        and isinstance(outs[0].shape[0], (int, IntImm))
        and outs[0].shape[0] >= warp_size
        and isinstance(outs[0].shape[1], (int, IntImm))
        and outs[0].shape[1] >= warp_size
    ):
        return topi.cuda.schedule_transpose(outs)
    return schedule_injective(attrs, outs, target)


@invert_permutation_strategy.register(["cuda", "gpu"])
def invert_permutation_strategy_cuda(attrs, inputs, out_type, target):
    """invert_permutation cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_invert_permutation(topi.cuda.invert_permutation),
        wrap_topi_schedule(topi.cuda.vision._default_schedule),
        name="invert_permutation.cuda",
    )
    return strategy


@einsum_strategy.register(["cuda", "gpu"])
def einsum_strategy_cuda(attrs, inputs, out_type, target):
    """einsum cuda strategy"""
    strategy = _op.OpStrategy()
    # TODO: Add cuda-specific op implementation for einsum
    strategy.add_implementation(
        wrap_compute_einsum(topi.einsum),
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="einsum.cuda",
    )
    return strategy


@stft_strategy.register(["cuda", "gpu"])
def stft_strategy_cuda(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_stft(topi.cuda.stft),
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="stft.cuda",
    )
    return strategy


@dft_strategy.register(["cuda", "gpu"])
def dft_strategy_cuda(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_dft(topi.cuda.dft),
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="dft.cuda",
    )
    return strategy

@layout_transform_strategy.register(["cuda", "gpu"])
def layout_transform_strategy(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_layout_transform(topi.layout_transform, schedule_rule="layout_transform"),
        wrap_topi_schedule(topi.cuda.schedule_injective),
        name="layout_transform.cuda",
    )
    return strategy


import tvm

@tvm.register_func("meta_schedule.cuda.layout_transform")
def cuda_layout_transform_schedule_rule(sch, block):
    # params: input_buffer, output_buffer
    params = sch.mod["main"].params
    input_buffer = sch.mod["main"].buffer_map[params[0]]
    output_buffer = sch.mod["main"].buffer_map[params[1]]

    input_shape = [int(dim) for dim in input_buffer.shape]
    output_shape = [int(dim) for dim in output_buffer.shape]

    src_layout = sch.get_sref(block).stmt.annotations["src_layout"]
    dst_layout = sch.get_sref(block).stmt.annotations["dst_layout"]

    import math
    from typing import List, Sequence, Tuple

    from tvm.tir.schedule import BlockRV, ExprRV, LoopRV

    def schedule_layout_transform_v4(
        sch: tvm.tir.Schedule,
        block_write: BlockRV,
        src_layout: str,
        dst_layout: str,
        input_shape: List[int],
        tile_size: ExprRV,
    ):
        ## Tiling block_read
        # Let N and M represent the dimensions of interest
        # N and M are the last dim of src_layout and dst_layout respectively.
        # Then the initial block's loop will look like
        # [i1, i2 ... M ... j1, j2 ... N]
        #
        # To guarantee contiguous read for N, we must group reads
        # so that loops which factor N, j_n, j_n-1 ... are the innermost dimension
        # Therefore our strategy for guaranteeing contiguous writes for N is by
        # continually splitting inner-most dimension in order of N, j_n, j_n-1...
        # by factors which divide into tile_size until we have the final tile_size.
        # e.g. if tile_size = 32. N = 2, j_n = 2, j_n-1 = 4, j_n-2 = 24
        # Then by combining N...j_n-1 we get a factor of 16, by spliiting up j_n-2 into
        # two loops of 2 and 12, we can combine with the new loop of factor 2 to get a
        # factor of 32. Note things don't divide evenly often time so we may have to pad
        # to properly factorize.
        #
        # Similarly with M, to have contiguous writes we must consider the dst_layout:
        # [a1, a2 ... N ... b1, b2 ... M]
        # So that loops which factor M, b_m, b_m-1 ... are the innermost dimension
        # Note that the dimension b_m, b_m-1 and j_n, j_n-1 may refer to the same dimension!
        # By factoring j_n, j_n-1 we may be contributing to the innermost write dimension for
        # M. However we note that both reads and writes must ideally have the same amount of
        # work per thread for layout transforms so we still must build out to factors up
        # to tile_size.

        def pad_dimension_to_at_least_number(loop: LoopRV, requested_size: int):
            """E.g. if loop has extant of 8 but we want 10, returns size 10 loop with padding"""
            l1, l2 = sch.split(loop, [None, requested_size])
            return sch.fuse(l1, l2)

        def pad_dimension_to_factor_of_tile_size(
            loop: LoopRV, initial_size: int, tile_size: int = tile_size
        ) -> Tuple[LoopRV, int]:
            """
            Pads loop of given size until it is divisble into tile_size.
            If the given size of the loop is greater than tile size. Do not pad.

            example, loop_size = 5, tile_size = 32. loop_size --> 8
                    loop_size = 5, tile_size = 36. loop_size --> 6
                    loop_size = 8, tile_size = 32. loop_size --> 8
                    loop_size = 33, tile_size = 32. loop_size --> 33

            Returns padded loopRV and the new size
            """
            if tile_size % initial_size == 0:
                return loop, int(initial_size)

            if initial_size > tile_size or initial_size == tile_size:
                return loop, int(initial_size)

            # if initial_size > tile_size return without change, factor = 1
            size = initial_size
            while (tile_size % size) % tile_size > 0:
                size += 1

            return pad_dimension_to_at_least_number(loop, size), int(size)

        def spin_out_factor(
            loops: List[LoopRV], loop_extants: List[int], index: int, factor_needed: int
        ) -> Tuple[List[LoopRV], List[int], int]:
            """
            Factor out loop extant to reach the requested factor. Updates the schedule in-place.

            E.g. say we want to factors which eventually multiply to 32 (factor_needed).

            Say we have the index we chose is a loop with an extant of 8.
            E.g. loops / loop_extants = [3, 32, 6, 8], factor_needed = 32, index = 3
                - 8 divides into 32 so we just split up the loop into two loops with extants 1 and 8.
                - we then keep the 1-loop in place and move the new 8-loop to back of the list of loops
                - ending loops / loop_extants = [3, 32, 6, 1, 8], remaining_factor_needed = 32 / 8 = 4

            E.g. loops / loop_extants = [3, 32, 6, 8], factor_needed=32, index = 0
                - 3 does not divide 32, so we pad until the extant divides 32, e.g. 4
                - we then split up the loop into extants 1 and 4, moving the 4 to the back
                - ending loops / loop_extants = [1, 32, 6, 8, 4], remaining_factor_needed = 32 / 4 = 8

            E.g. loops / loop_extants = [3, 32, 6, 8], factor_needed=5, index = 3
                - 8 is larger than 5 so we immediately do the splitting routine.
                - the 8 extant loop becomes loops with extants 2 and 5
                - ending loops / loop_extants = [1, 32, 6, 2, 5], remaining_factor_needed = 5 / 5 = 1

            After updating loop ordering in place, returns the new list of loops, extants, and the
            remaining factor needed.
            """
            cur_loop = loops[index]
            cur_extant = loop_extants[index]

            # Pad loops to divide evenly for factors needed, and split
            new_loop, new_size = pad_dimension_to_factor_of_tile_size(
                cur_loop, cur_extant, tile_size=factor_needed
            )

            split_factor = min(new_size, factor_needed)
            new_loop_split, factored_loop = sch.split(new_loop, [None, split_factor])
            factor_needed = factor_needed // split_factor

            # update caching
            loops[index] = new_loop_split
            loops.append(factored_loop)

            loop_extants[index] = math.ceil(new_size / split_factor)
            loop_extants.append(split_factor)

            sch.reorder(*loops)
            return loops, loop_extants, factor_needed

        def factor_dim_in_order(
            indices: Sequence[int],
            loops: List[LoopRV],
            cur_loop_extants: List[int],
            work_needed_inner_loop: int = tile_size,
        ):
            """TODO"""
            for i in indices:
                loops, cur_loop_extants, work_needed_inner_loop = spin_out_factor(
                    loops, cur_loop_extants, i, work_needed_inner_loop
                )
                if work_needed_inner_loop == 1:
                    break
            return loops, cur_loop_extants

        def get_high_level_loop_structure(block):
            """Runs the factorization described above."""
            # index 0 ... rank - 1 will always correspond to original loops
            # perhaps after they have been factored.
            loops = sch.get_loops(block)
            cur_loop_extants = list(input_shape)

            # Factor dim0 tile size and fuse things together
            loops, cur_loop_extants = factor_dim_in_order(
                range(rank - 1, -1, -1),
                loops,
                cur_loop_extants,
                work_needed_inner_loop=tile_size,
            )
            # The factors which multiply to tile_size are now in back of our
            # list of loops. However because we added them by traversing the inner
            # dimensions, they are actually reversed order to guarantee the best access
            # so reorder so reorder before fusing.
            loops = loops[:rank] + loops[rank:][::-1]
            cur_loop_extants = cur_loop_extants[:rank] + cur_loop_extants[rank::-1]
            sch.reorder(*loops)
            dim0_loop_tiled = sch.fuse(*loops[rank:])
            loops = loops[:rank]
            loops.append(dim0_loop_tiled)
            cur_loop_extants = cur_loop_extants[:rank]
            cur_loop_extants.append(tile_size)

            # Same thing with dim1
            # [:rank + 1], since we placed dim0_loop_tiled in the end which we want to keep
            loops, cur_loop_extants = factor_dim_in_order(
                (
                    src_layout.index(dst_layout[loop_index_dst])
                    for loop_index_dst in range(rank - 1, -1, -1)
                ),
                loops,
                cur_loop_extants,
                work_needed_inner_loop=tile_size,
            )
            loops = loops[: rank + 1] + loops[rank + 1 :][::-1]
            cur_loop_extants = cur_loop_extants[: rank + 1] + cur_loop_extants[rank + 1 :: -1]
            sch.reorder(*loops)
            dim1_loop_tiled = sch.fuse(*loops[rank + 1 :])
            loops = loops[: rank + 1]
            loops.append(dim1_loop_tiled)
            cur_loop_extants = cur_loop_extants[: rank + 1]
            cur_loop_extants.append(tile_size)

        rank = len(src_layout)

        # Outer loop structure of read block matches that of src_layout
        # E.g. if input_shape is [4, 6, 8]. Loops for read block will be
        # for i, j, k in T.grid(4, 6, 8):
        #     ...
        # Read block will read from global memory coalesced at the start
        # Assume write to output global memory is coalesced in block_write
        block_read = sch.cache_read(block_write, 0, "shared")

        # Here we have [loop1, loop2, loop3 ... dim0_tiled, dim1_tiled]
        get_high_level_loop_structure(block_read)

        loops = sch.get_loops(block_read)

        # If there are insufficient elements, than dim1_tiled or dim0_tiled might be too small
        # In all likelihood you should use a smaller tile, but I don't want things to crash.
        loops[-1] = pad_dimension_to_at_least_number(loops[-1], tile_size)
        loops[-2] = pad_dimension_to_at_least_number(loops[-2], tile_size)

        # We want the dim0 and dim1 parent loops to be the inner most. Right now dim1 is inner-msot
        # and we just need to move dim0 in (last dimension of dst).
        # Recall right now structure is at least [l1 l2 ... ln, dim0_tiled, dim1_tiled]
        # where n >= 2.
        dim0_loop_index = src_layout.index(dst_layout[-1])
        dim0_loop = loops.pop(dim0_loop_index)
        loops = loops[:-3] + [dim0_loop, loops[-3]] + loops[-2:]
        sch.reorder(*loops)

        # After this: [outer_loop (block), dim0_tiled, dim1_tiled]
        outer_loop = sch.fuse(*loops[:-2])

        # Now that we have the high level loop structure, we can use reverse_compute_at magic
        # To get the proper loop structure for writing! This is also as coalesced as possible
        # already.
        sch.reverse_compute_at(block_write, outer_loop)

        # Fuse all inner loops for the write into 2 loops, grab inner loops for both read
        # and write block which have locality (we will bind these to threadIdx)
        fused_write_loop = sch.fuse(*sch.get_loops(block_write)[1:])
        _, inner_write_loop = sch.split(fused_write_loop, [None, tile_size])
        inner_read_loop = sch.get_loops(block_read)[-2]

        sch.bind(loop=outer_loop, thread_axis="blockIdx.x")
        sch.bind(loop=inner_write_loop, thread_axis="threadIdx.x")
        sch.bind(loop=inner_read_loop, thread_axis="threadIdx.x")

    from collections import deque
    def auto_inline(start_block):
        # BFS from start block in a chain (no branches)
        fringe = deque([start_block])
        visited = set()
        while len(fringe) > 0:
            cur_block = fringe.popleft()
            if cur_block in visited:
                continue 
            else:
                visited.add(cur_block)
                
            consumer_blocks = sch.get_consumers(cur_block)
            
            if len(consumer_blocks) >= 1:
                fringe.extend(consumer_blocks)
                sch.compute_inline(cur_block)
            else:
                # consumer yay!
                return cur_block

    schedules = []
    
    # For each schedule we also want to inline each stage as would be done in normal circumstances
    # The block which producers the layout transform block seems 
    block = auto_inline(block)

    # Tile size 2,3,4...64
    # Tile size of 1 does not make sense...
    for tile_size in range(2, 65):
        cur_sch = sch.copy()
        schedule_layout_transform_v4(cur_sch, block, src_layout, dst_layout, input_shape, tile_size)
        schedules.append(cur_sch)

    # Also include the default schedules which will be handled via AutoBind schedule rule
    schedules.append(sch)
        
    return schedules
