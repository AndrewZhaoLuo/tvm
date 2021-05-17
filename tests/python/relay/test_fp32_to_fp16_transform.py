from typing import *

import numpy as np
import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprVisitor
from tvm.relay.testing import resnet
from tvm.relay.transform.fp16_conversion import fp32_to_fp16

def run_module(mod, mod_params):
    dev = tvm.device("llvm", 0)
    intrp = relay.create_executor("debug", mod, device=dev, target="llvm")
    # in_data = [tvm.nd.array(value) for value in in_data.values()]
    return intrp.evaluate()(**mod_params).asnumpy()

def test_resnet18():
    np.random.seed(4321)
    mod, mod_params = resnet.get_workload(1, 5, num_layers=18, image_shape=(1, 32, 32))
    mod_params["data"] = np.random.uniform(-10, 10, (1, 1, 32, 32)).astype("float32")

    result_fp32 = run_module(mod, mod_params)
    output = fp32_to_fp16.quantize_to_fp16(mod["main"].body)

    fp16_mod = tvm.ir.IRModule.from_expr(output)
    result_fp16 = run_module(fp16_mod, mod_params)

    np.testing.assert_allclose(result_fp32, result_fp16, rtol=1e-3)
