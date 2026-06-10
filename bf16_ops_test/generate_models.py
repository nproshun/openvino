# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generate minimal OpenVINO IR models (one per op) with BF16 precision."""

import os
import numpy as np
import openvino as ov
from openvino import op as ov_op
from openvino import opset13 as opset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def save(model, op_name):
    """Save model to models/<op_name>/<op_name>.xml."""
    out_dir = os.path.join(MODELS_DIR, op_name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{op_name}.xml")
    ov.save_model(model, path, compress_to_fp16=False)
    print(f"  Saved {path}")


def make_param(name, shape, dtype=ov.Type.bf16):
    p = ov_op.Parameter(dtype, ov.Shape(shape))
    p.friendly_name = name
    return p


def const_i64(name, values):
    c = ov_op.Constant(ov.Type.i64, ov.Shape([len(values)]), values)
    c.friendly_name = name
    return c


# ── Add ──────────────────────────────────────────────────────────────────────
def gen_add():
    d1 = make_param("data1", [1, 3, 4, 4])
    d2 = make_param("data2", [1, 3, 4, 4])
    node = opset.add(d1, d2)
    return ov.Model([node], [d1, d2], "Add")


# ── Clamp ────────────────────────────────────────────────────────────────────
def gen_clamp():
    d = make_param("data", [1, 3, 4, 4])
    node = opset.clamp(d, min_value=0.0, max_value=6.0)
    return ov.Model([node], [d], "Clamp")


# ── Concat ───────────────────────────────────────────────────────────────────
def gen_concat():
    d1 = make_param("data1", [1, 3, 4, 4])
    d2 = make_param("data2", [1, 2, 4, 4])
    node = opset.concat([d1, d2], axis=1)
    return ov.Model([node], [d1, d2], "Concat")


# ── Divide ───────────────────────────────────────────────────────────────────
def gen_divide():
    d1 = make_param("data1", [1, 3, 4, 4])
    d2 = make_param("data2", [1, 3, 4, 4])
    node = opset.divide(d1, d2)
    return ov.Model([node], [d1, d2], "Divide")


# ── Gather ───────────────────────────────────────────────────────────────────
def gen_gather():
    d = make_param("data", [1, 4, 4, 4])
    indices = const_i64("indices", [0, 2])
    axis = ov_op.Constant(ov.Type.i64, ov.Shape([]), [1])
    axis.friendly_name = "axis"
    node = opset.gather(d, indices, axis)
    return ov.Model([node], [d], "Gather")


# ── MatMul ───────────────────────────────────────────────────────────────────
def gen_matmul():
    d1 = make_param("data1", [1, 3, 4, 8])
    d2 = make_param("data2", [1, 3, 8, 4])
    node = opset.matmul(d1, d2, False, False)
    return ov.Model([node], [d1, d2], "MatMul")


# ── Multiply ─────────────────────────────────────────────────────────────────
def gen_multiply():
    d1 = make_param("data1", [1, 3, 4, 4])
    d2 = make_param("data2", [1, 3, 4, 4])
    node = opset.multiply(d1, d2)
    return ov.Model([node], [d1, d2], "Multiply")


# ── Power ────────────────────────────────────────────────────────────────────
def gen_power():
    d1 = make_param("data1", [1, 3, 4, 4])
    d2 = make_param("data2", [1, 3, 4, 4])
    node = opset.power(d1, d2)
    return ov.Model([node], [d1, d2], "Power")


# ── ReduceL2 ─────────────────────────────────────────────────────────────────
def gen_reduce_l2():
    d = make_param("data", [1, 3, 4, 4])
    axes = const_i64("axes", [2, 3])
    node = opset.reduce_l2(d, axes, keep_dims=True)
    return ov.Model([node], [d], "ReduceL2")


# ── ReduceMean ───────────────────────────────────────────────────────────────
def gen_reduce_mean():
    d = make_param("data", [1, 3, 4, 4])
    axes = const_i64("axes", [2, 3])
    node = opset.reduce_mean(d, axes, keep_dims=True)
    return ov.Model([node], [d], "ReduceMean")


# ── Reshape ──────────────────────────────────────────────────────────────────
def gen_reshape():
    d = make_param("data", [1, 3, 4, 4])
    shape = const_i64("target_shape", [1, 48])
    node = opset.reshape(d, shape, special_zero=False)
    return ov.Model([node], [d], "Reshape")


# ── ScaledDotProductAttention ────────────────────────────────────────────────
def gen_sdpa():
    q = make_param("query",          [1, 2, 4, 8])
    k = make_param("key",            [1, 2, 6, 8])
    v = make_param("value",          [1, 2, 6, 8])
    m = make_param("attention_mask", [1, 2, 4, 6])
    node = opset.scaled_dot_product_attention(q, k, v, attention_mask=m, causal=False)
    return ov.Model([node], [q, k, v, m], "ScaledDotProductAttention")


# ── Select ───────────────────────────────────────────────────────────────────
def gen_select():
    cond = make_param("condition", [1, 3, 4, 4], dtype=ov.Type.boolean)
    then_val = make_param("then_value", [1, 3, 4, 4])
    else_val = make_param("else_value", [1, 3, 4, 4])
    node = opset.select(cond, then_val, else_val)
    return ov.Model([node], [cond, then_val, else_val], "Select")


# ── ShapeOf ──────────────────────────────────────────────────────────────────
def gen_shape_of():
    d = make_param("data", [1, 3, 4, 4])
    node = opset.shape_of(d)
    return ov.Model([node], [d], "ShapeOf")


# ── Sigmoid ──────────────────────────────────────────────────────────────────
def gen_sigmoid():
    d = make_param("data", [1, 3, 4, 4])
    node = opset.sigmoid(d)
    return ov.Model([node], [d], "Sigmoid")


# ── Slice ────────────────────────────────────────────────────────────────────
def gen_slice():
    d = make_param("data", [1, 4, 4, 4])
    start = const_i64("start", [0])
    stop  = const_i64("stop",  [2])
    step  = const_i64("step",  [1])
    axes  = const_i64("axes",  [1])
    node = opset.slice(d, start, stop, step, axes)
    return ov.Model([node], [d], "Slice")


# ── Split ────────────────────────────────────────────────────────────────────
def gen_split():
    d = make_param("data", [1, 4, 4, 4])
    axis = ov_op.Constant(ov.Type.i64, ov.Shape([]), [1])
    axis.friendly_name = "axis"
    node = opset.split(d, axis, num_splits=2)
    return ov.Model(node.outputs(), [d], "Split")


# ── Tile ─────────────────────────────────────────────────────────────────────
def gen_tile():
    d = make_param("data", [1, 3, 4, 4])
    repeats = const_i64("repeats", [1, 2, 1, 1])
    node = opset.tile(d, repeats)
    return ov.Model([node], [d], "Tile")


# ── Transpose ────────────────────────────────────────────────────────────────
def gen_transpose():
    d = make_param("data", [1, 3, 4, 4])
    order = const_i64("order", [0, 2, 1, 3])
    node = opset.transpose(d, order)
    return ov.Model([node], [d], "Transpose")


# ── VariadicSplit ────────────────────────────────────────────────────────────
def gen_variadic_split():
    d = make_param("data", [1, 4, 4, 4])
    axis = ov_op.Constant(ov.Type.i64, ov.Shape([]), [1])
    axis.friendly_name = "axis"
    lengths = const_i64("split_lengths", [2, 2])
    node = opset.variadic_split(d, axis, lengths)
    return ov.Model(node.outputs(), [d], "VariadicSplit")


# ── Main ─────────────────────────────────────────────────────────────────────
GENERATORS = {
    "Add":                          gen_add,
    "Clamp":                        gen_clamp,
    "Concat":                       gen_concat,
    "Divide":                       gen_divide,
    "Gather":                       gen_gather,
    "MatMul":                       gen_matmul,
    "Multiply":                     gen_multiply,
    "Power":                        gen_power,
    "ReduceL2":                     gen_reduce_l2,
    "ReduceMean":                   gen_reduce_mean,
    "Reshape":                      gen_reshape,
    "ScaledDotProductAttention":    gen_sdpa,
    "Select":                       gen_select,
    "ShapeOf":                      gen_shape_of,
    "Sigmoid":                      gen_sigmoid,
    "Slice":                        gen_slice,
    "Split":                        gen_split,
    "Tile":                         gen_tile,
    "Transpose":                    gen_transpose,
    "VariadicSplit":                gen_variadic_split,
}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Generating {len(GENERATORS)} IR models into {MODELS_DIR} ...")
    for name, gen_fn in GENERATORS.items():
        print(f"[{name}]")
        model = gen_fn()
        save(model, name)
    print("Done.")


if __name__ == "__main__":
    main()
