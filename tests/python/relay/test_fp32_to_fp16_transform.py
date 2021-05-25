from collections import defaultdict
from typing import *

import numpy as np
import tvm
from tvm import relay
from tvm.relay.op.tensor import exp
from tvm.relay.testing import densenet, mobilenet, resnet, resnet_3d, squeezenet
from tvm.relay.transform import RewriteFP16


def run_module(mod, mod_params):
    dev = tvm.device("llvm", 0)
    intrp = relay.create_executor("debug", mod, device=dev, target="llvm")
    return intrp.evaluate()(**mod_params).asnumpy()


def verify_fp32_fp16_output_close(mod, mod_params, rtol=1e-3, atol=0):
    fp16_mod = RewriteFP16()(mod)
    result_fp16 = run_module(fp16_mod, mod_params)
    result_fp32 = run_module(mod, mod_params)

    # Ensure the results are close
    np.testing.assert_allclose(result_fp32, result_fp16, rtol=rtol, atol=atol)

    return fp16_mod


def test_resnet18():
    np.random.seed(4321)
    mod, mod_params = resnet.get_workload(1, 5, num_layers=18, image_shape=(1, 32, 32))
    mod_params["data"] = np.random.uniform(-10, 10, (1, 1, 32, 32)).astype("float32")

    verify_fp32_fp16_output_close(mod, mod_params)


def test_resnet18_3d():
    np.random.seed(3215)
    mod, mod_params = resnet_3d.get_workload(1, 5, num_layers=18, image_shape=(1, 3, 32, 32))
    mod_params["data"] = np.random.uniform(-10, 10, (1, 1, 3, 32, 32)).astype("float32")

    verify_fp32_fp16_output_close(mod, mod_params)


def test_mobilenet():
    np.random.seed(4615)

    mod, mod_params = mobilenet.get_workload(1, 5, image_shape=(1, 32, 32))
    mod_params["data"] = np.random.uniform(-10, 10, (1, 1, 32, 32)).astype("float32")

    verify_fp32_fp16_output_close(mod, mod_params)


def test_densenet():
    np.random.seed(3222)
    mod, mod_params = densenet.get_workload(classes=5, batch_size=1, image_shape=(1, 224, 224))
    mod_params["data"] = np.random.uniform(-10, 10, (1, 1, 224, 224)).astype("float32")

    verify_fp32_fp16_output_close(mod, mod_params)


def test_squeezenet():
    np.random.seed(5628)
    mod, mod_params = squeezenet.get_workload(1, 5, image_shape=(1, 32, 32))
    mod_params["data"] = np.random.uniform(-10, 10, (1, 1, 32, 32)).astype("float32")

    verify_fp32_fp16_output_close(mod, mod_params)


def test_convert_single_conv():
    """Conv is a green listed operation meaning it will always use fp16 workload.

    By default it accumulates to fp32 and outputs fp16.
    """
    np.random.seed(208)

    data_shape = (1, 3, 32, 32)
    weight_shape = (5, 3, 3, 3)
    data = relay.var("data", shape=data_shape, dtype="float32")
    weight = relay.var("weight", shape=weight_shape, dtype="float32")
    conv = relay.nn.conv2d(data, weight, strides=(1, 1), padding=(1, 1), out_dtype="float32")
    mod = tvm.IRModule.from_expr(conv)
    mod = tvm.relay.transform.InferType()(mod)

    mod_params = {
        "data": np.random.uniform(-1, 1, size=data_shape).astype("float32"),
        "weight": np.random.uniform(-1, 1, size=weight_shape).astype("float32"),
    }
    fp16_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.01, rtol=1e-3)

    expected_mod = tvm.IRModule.from_expr(
        relay.cast(
            relay.nn.conv2d(
                relay.cast(data, "float16"),
                relay.cast(weight, "float16"),
                strides=(1, 1),
                padding=(1, 1),
                out_dtype="float32",
            ),
            "float16",
        )
    )
    expected_mod = tvm.relay.transform.InferType()(expected_mod)

    assert not tvm.ir.structural_equal(fp16_mod, mod)
    assert tvm.ir.structural_equal(fp16_mod, expected_mod)


def test_do_not_convert_softmax():
    """Softmax is a red listed operation and therefore should never be fp16."""
    np.random.seed(209)
    shape = [1, 2, 3]
    a = relay.var("a", shape=shape)
    b = relay.nn.softmax(a)
    mod = tvm.IRModule.from_expr(b)
    mod = tvm.relay.transform.InferType()(mod)

    mod_params = {
        "a": np.random.uniform(-1, 1, size=shape).astype("float32"),
    }
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.0, rtol=0)
    assert tvm.ir.structural_equal(mod, output_mod)


def test_green_gray_propagates_simple():
    """Conv is a green listed operation, while addition is gray.

    When adjacent
    """
    np.random.seed(210)
    data_shape = (1, 3, 32, 32)
    weight_shape = (5, 3, 3, 3)
    data = relay.var("data", shape=data_shape, dtype="float32")
    weight = relay.var("weight", shape=weight_shape, dtype="float32")
    conv = relay.nn.conv2d(data, weight, strides=(1, 1), padding=(1, 1), out_dtype="float32")
    conv = conv + conv
    mod = tvm.IRModule.from_expr(conv)
    mod = tvm.relay.transform.InferType()(mod)

    mod_params = {
        "data": np.random.uniform(-1, 1, size=data_shape).astype("float32"),
        "weight": np.random.uniform(-1, 1, size=weight_shape).astype("float32"),
    }
    fp16_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.01, rtol=1e-3)

    conv_expr = relay.cast(
        relay.nn.conv2d(
            relay.cast(data, "float16"),
            relay.cast(weight, "float16"),
            strides=(1, 1),
            padding=(1, 1),
            out_dtype="float32",
        ),
        "float16",
    )
    expected_mod = tvm.IRModule.from_expr(conv_expr + conv_expr)
    expected_mod = tvm.relay.transform.InferType()(expected_mod)

    assert not tvm.ir.structural_equal(fp16_mod, mod)
    assert tvm.ir.structural_equal(fp16_mod, expected_mod)


def test_red_gray_propagates_simple():
    """Conv is a green listed operation, while addition is gray.

    When adjacent
    """
    np.random.seed(210)
    shape = [1, 2, 3]
    a = relay.var("a", shape=shape)
    b = relay.nn.softmax(a)
    c = b + b
    mod = tvm.IRModule.from_expr(c)
    mod = tvm.relay.transform.InferType()(mod)

    mod_params = {
        "a": np.random.uniform(-1, 1, size=shape).astype("float32"),
    }
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.0, rtol=0)

    assert tvm.ir.structural_equal(mod, output_mod)
