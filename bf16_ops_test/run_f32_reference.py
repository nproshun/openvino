# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run each BF16 IR model with inference_precision=f32, save inputs+outputs to JSON."""

import json
import os
import sys

import numpy as np
from ml_dtypes import bfloat16
import openvino as ov
from openvino import op as ov_op
from openvino import opset13 as opset
import openvino.properties.hint as hints

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SEED = 42

# Ops that need special input handling
DIVIDE_OP = "Divide"
POWER_OP = "Power"

# Cache for converter infer requests, keyed by (src_type, dst_type, shape_tuple)
_converter_cache = {}


def _get_converter_request(core, src_type, dst_type, shape):
    """Get or create a compiled converter model for the given types and shape."""
    key = (src_type, dst_type, shape)
    if key not in _converter_cache:
        param = ov_op.Parameter(src_type, ov.Shape(list(shape)))
        convert = opset.convert(param, dst_type)
        model = ov.Model([convert], [param], "converter")
        compiled = core.compile_model(model, "CPU")
        _converter_cache[key] = compiled.create_infer_request()
    return _converter_cache[key]


def f32_to_bf16_tensor(core, f32_arr):
    """Convert f32 numpy array to a bf16 ov.Tensor using a separate converter model."""
    shape = tuple(f32_arr.shape)
    req = _get_converter_request(core, ov.Type.f32, ov.Type.bf16, shape)
    inp = ov.Tensor(ov.Type.f32, ov.Shape(list(shape)))
    np.copyto(inp.data, f32_arr)
    req.set_input_tensor(inp)
    req.infer()
    out = req.get_output_tensor(0)
    result = ov.Tensor(ov.Type.bf16, ov.Shape(list(shape)))
    np.copyto(result.data, out.data)
    return result


def bf16_tensor_to_f32(core, bf16_tensor):
    """Convert a bf16 ov.Tensor to f32 numpy array using a separate converter model."""
    shape = tuple(bf16_tensor.shape)
    req = _get_converter_request(core, ov.Type.bf16, ov.Type.f32, shape)
    inp = ov.Tensor(ov.Type.bf16, ov.Shape(list(shape)))
    np.copyto(inp.data, bf16_tensor.data)
    req.set_input_tensor(inp)
    req.infer()
    out = req.get_output_tensor(0)
    return np.array(out.data, copy=True)


def generate_input(rng, param, op_name):
    """Generate a random f32 numpy array for the given model parameter.

    For bf16 parameters, values are quantized to bf16 precision so both
    f32 and bf16 runs see identical input values.
    """
    shape = list(param.get_shape())
    et = param.get_element_type()

    if et == ov.Type.boolean:
        return rng.randint(0, 2, size=shape).astype(np.bool_)

    # Float types: generate as f32
    if op_name == DIVIDE_OP and "data2" in param.get_friendly_name():
        f32_arr = rng.uniform(0.5, 2.0, size=shape).astype(np.float32)
    elif op_name == POWER_OP:
        f32_arr = rng.uniform(0.5, 2.0, size=shape).astype(np.float32)
    else:
        f32_arr = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)

    # Quantize to bf16 precision so both f32 and bf16 runs use same actual values
    if et == ov.Type.bf16:
        f32_arr = f32_arr.astype(bfloat16).astype(np.float32)

    return f32_arr


def tensor_to_serializable(np_array):
    """Convert numpy array to a JSON-serializable dict."""
    return {
        "shape": list(np_array.shape),
        "dtype": str(np_array.dtype),
        "data": np_array.astype(np.float64).flatten().tolist(),
    }


def run_model_f32(core, model_path, op_name, rng):
    """Read model, run with f32 precision, return inputs and outputs."""
    model = core.read_model(model_path)

    # Generate inputs
    input_arrays = {}
    input_data = {}
    for param in model.get_parameters():
        name = param.get_friendly_name()
        arr = generate_input(rng, param, op_name)
        input_arrays[name] = arr
        input_data[name] = tensor_to_serializable(arr)

    # Compile with f32 precision
    compiled = core.compile_model(model, "CPU", {hints.inference_precision: ov.Type.f32})
    infer_request = compiled.create_infer_request()

    # Set inputs — use converter model for bf16 params
    for param in model.get_parameters():
        name = param.get_friendly_name()
        arr = input_arrays[name]
        et = param.get_element_type()
        if et == ov.Type.bf16:
            tensor = f32_to_bf16_tensor(core, arr)
        else:
            tensor = ov.Tensor(et, ov.Shape(list(arr.shape)))
            np.copyto(tensor.data, arr.reshape(tensor.data.shape))
        infer_request.set_tensor(param.output(0), tensor)

    # Run inference
    infer_request.infer()

    # Collect outputs — use converter model for bf16 outputs
    output_data = {}
    results = model.get_results()
    for i, result_node in enumerate(results):
        out_tensor = infer_request.get_output_tensor(i)
        if result_node.get_element_type() == ov.Type.bf16:
            out_arr = bf16_tensor_to_f32(core, out_tensor)
        else:
            out_arr = np.array(out_tensor.data, copy=True)
        output_data[f"output_{i}"] = tensor_to_serializable(out_arr)

    return input_data, output_data


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    core = ov.Core()

    rng = np.random.RandomState(SEED)

    # Find all model directories
    op_names = sorted(d for d in os.listdir(MODELS_DIR)
                      if os.path.isdir(os.path.join(MODELS_DIR, d)))

    print(f"Running f32 reference for {len(op_names)} models...")
    failed = []
    for op_name in op_names:
        model_path = os.path.join(MODELS_DIR, op_name, f"{op_name}.xml")
        if not os.path.exists(model_path):
            print(f"  [{op_name}] SKIP — model not found")
            continue

        print(f"  [{op_name}] ", end="", flush=True)
        try:
            input_data, output_data = run_model_f32(core, model_path, op_name, rng)
            result = {"op_name": op_name, "inputs": input_data, "outputs": output_data}
            json_path = os.path.join(RESULTS_DIR, f"{op_name}.json")
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)
            n_outputs = len(output_data)
            print(f"OK ({n_outputs} output{'s' if n_outputs > 1 else ''})")
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(op_name)

    print(f"\nDone. {len(op_names) - len(failed)}/{len(op_names)} succeeded.")
    if failed:
        print(f"Failed ops: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
