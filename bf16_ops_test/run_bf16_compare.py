# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run each BF16 IR model with inference_precision=bf16, compare against f32 JSON reference."""

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

ATOL = 1e-2
RTOL = 1e-2

# Ops whose outputs are integer and need exact comparison
EXACT_OPS = {"ShapeOf"}

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


def load_reference(op_name):
    """Load the JSON reference file for the given op."""
    json_path = os.path.join(RESULTS_DIR, f"{op_name}.json")
    with open(json_path, "r") as f:
        return json.load(f)


def reconstruct_array(entry):
    """Reconstruct a numpy array from a JSON dict."""
    shape = entry["shape"]
    dtype = entry["dtype"]
    data = np.array(entry["data"], dtype=np.float64)
    return data.astype(dtype).reshape(shape)


def compare_outputs(op_name, ref_outputs, actual_outputs):
    """Compare actual outputs against reference. Returns (passed, details_str)."""
    details = []
    all_ok = True

    for key in sorted(ref_outputs.keys()):
        ref_arr = reconstruct_array(ref_outputs[key])
        act_arr = actual_outputs[key]

        # Cast both to float64 for comparison
        ref_f64 = ref_arr.astype(np.float64)
        act_f64 = act_arr.astype(np.float64)

        if op_name in EXACT_OPS:
            if np.array_equal(ref_f64, act_f64):
                details.append(f"  {key}: EXACT MATCH")
            else:
                diff_mask = ref_f64 != act_f64
                n_diff = np.sum(diff_mask)
                details.append(f"  {key}: EXACT MISMATCH — {n_diff} elements differ")
                all_ok = False
        else:
            abs_diff = np.abs(ref_f64 - act_f64)
            max_abs = np.max(abs_diff) if abs_diff.size > 0 else 0.0
            mean_abs = np.mean(abs_diff) if abs_diff.size > 0 else 0.0

            # Relative error (avoid div by zero)
            denom = np.maximum(np.abs(ref_f64), 1e-12)
            rel_diff = abs_diff / denom
            max_rel = np.max(rel_diff) if rel_diff.size > 0 else 0.0

            ok = np.allclose(ref_f64, act_f64, atol=ATOL, rtol=RTOL)
            if not ok:
                all_ok = False

            status = "PASS" if ok else "FAIL"
            details.append(
                f"  {key}: {status}  max_abs={max_abs:.6e}  mean_abs={mean_abs:.6e}  "
                f"max_rel={max_rel:.6e}  shape={list(act_arr.shape)}"
            )

            if not ok:
                # Show top-5 biggest mismatches by absolute error
                flat_abs = abs_diff.flatten()
                flat_ref = ref_f64.flatten()
                flat_act = act_f64.flatten()
                n_show = min(5, len(flat_abs))
                worst_idx = np.argsort(flat_abs)[-n_show:][::-1]
                details.append(f"    Top {n_show} mismatches (flat index, ref, actual, abs_err, rel_err):")
                for idx in worst_idx:
                    nd_idx = np.unravel_index(idx, abs_diff.shape)
                    details.append(
                        f"      [{idx}] {nd_idx}  ref={flat_ref[idx]:.8e}  "
                        f"actual={flat_act[idx]:.8e}  abs={flat_abs[idx]:.8e}  "
                        f"rel={rel_diff.flatten()[idx]:.8e}"
                    )

    return all_ok, "\n".join(details)


def run_model_bf16(core, model_path, op_name, ref_data):
    """Run model with bf16 precision using the same inputs from reference."""
    model = core.read_model(model_path)

    # Compile with bf16 precision
    compiled = core.compile_model(model, "GPU", {hints.inference_precision: ov.Type.bf16})
    infer_request = compiled.create_infer_request()

    # Set inputs from reference data — use converter model for bf16 params
    for param in model.get_parameters():
        name = param.get_friendly_name()
        et = param.get_element_type()
        ref_input = ref_data["inputs"][name]
        arr = reconstruct_array(ref_input)

        if et == ov.Type.bf16:
            tensor = f32_to_bf16_tensor(core, arr.astype(np.float32))
        elif et == ov.Type.boolean:
            tensor = ov.Tensor(et, ov.Shape(list(arr.shape)))
            np.copyto(tensor.data, arr.astype(np.bool_).reshape(tensor.data.shape))
        else:
            tensor = ov.Tensor(et, ov.Shape(list(arr.shape)))
            np.copyto(tensor.data, arr.reshape(tensor.data.shape))

        infer_request.set_tensor(param.output(0), tensor)

    # Run inference
    infer_request.infer()

    # Collect outputs — use converter model for bf16 outputs
    outputs = {}
    results = model.get_results()
    for i, result_node in enumerate(results):
        out_tensor = infer_request.get_output_tensor(i)
        if result_node.get_element_type() == ov.Type.bf16:
            outputs[f"output_{i}"] = bf16_tensor_to_f32(core, out_tensor)
        else:
            outputs[f"output_{i}"] = np.array(out_tensor.data, copy=True)

    return outputs


def main():
    core = ov.Core()

    # Check CPU bf16 capability
    caps = core.get_property("CPU", "OPTIMIZATION_CAPABILITIES")
    print(f"CPU optimization capabilities: {caps}")
    if "BF16" not in caps:
        print("WARNING: CPU does not report BF16 capability. Results may not reflect true bf16 execution.")

    # Find all model directories
    op_names = sorted(d for d in os.listdir(MODELS_DIR)
                      if os.path.isdir(os.path.join(MODELS_DIR, d)))
    print(f"\nComparing bf16 vs f32 reference for {len(op_names)} models (atol={ATOL}, rtol={RTOL})...\n")

    passed_ops = []
    failed_ops = []
    skipped_ops = []

    for op_name in op_names:
        model_path = os.path.join(MODELS_DIR, op_name, f"{op_name}.xml")
        json_path = os.path.join(RESULTS_DIR, f"{op_name}.json")

        if not os.path.exists(model_path):
            print(f"[{op_name}] SKIP — model not found")
            skipped_ops.append(op_name)
            continue
        if not os.path.exists(json_path):
            print(f"[{op_name}] SKIP — reference JSON not found")
            skipped_ops.append(op_name)
            continue

        ref_data = load_reference(op_name)
        try:
            actual_outputs = run_model_bf16(core, model_path, op_name, ref_data)
            ok, details = compare_outputs(op_name, ref_data["outputs"], actual_outputs)
            status = "PASS" if ok else "FAIL"
            print(f"[{op_name}] {status}")
            print(details)
            if ok:
                passed_ops.append(op_name)
            else:
                failed_ops.append(op_name)
        except Exception as e:
            print(f"[{op_name}] ERROR: {e}")
            failed_ops.append(op_name)

    # Summary
    total = len(op_names)
    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(passed_ops)}/{total} PASSED, "
          f"{len(failed_ops)}/{total} FAILED, "
          f"{len(skipped_ops)}/{total} SKIPPED")
    print(f"Passed: {', '.join(passed_ops)}")
    if failed_ops:
        print(f"Failed: {', '.join(failed_ops)}")
    print(f"{'='*60}")

    return 1 if failed_ops else 0


if __name__ == "__main__":
    sys.exit(main())
