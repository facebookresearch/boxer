#!/usr/bin/env python3
"""Append selected intermediate tensors to an ONNX model's graph outputs."""

import argparse
import os

import onnx
from onnx import TensorProto, helper


COARSE_TENSORS = [
    "/stem_2/stem_2.3/Relu_output_0",
    "/feature/dino/resize_layers.0/ConvTranspose_output_0",
    "/feature/dino/resize_layers.1/ConvTranspose_output_0",
    "/feature/deconv32_16/conv2/relu_1/Relu_output_0",
    "/feature/deconv16_8/conv2/relu_1/Relu_output_0",
    "/feature/deconv8_4/conv2/relu_1/Relu_output_0",
    "/corr_stem/corr_stem.3/relu_1/Relu_output_0",
    "/corr_feature_att/Mul_output_0",
    "/cost_agg/conv3/conv3.1/conv2/conv2.2/Relu_output_0",
    "/cost_agg/conv3_up/LeakyRelu_output_0",
    "/cost_agg/conv2_up/LeakyRelu_output_0",
    "/cost_agg/conv1_up/LeakyRelu_output_0",
    "/update_block/gru16/Add_output_0",
    "/update_block/gru08/Add_output_0",
    "/update_block/gru04/Add_output_0",
    "/update_block/disp_head/conv/conv.4/Conv_output_0",
    "disp",
]


def _iter_suffix(iter_idx: int) -> str:
    return "" if iter_idx == 0 else f"_{iter_idx}"


def _update_fine_tensors():
    tensors = []
    for iter_idx in range(16):
        suffix = _iter_suffix(iter_idx)
        tensors.extend(
            [
                f"/update_block/gru04/small_gru{suffix}/Sigmoid_output_0",
                f"/update_block/gru04/small_gru{suffix}/Sigmoid_1_output_0",
                f"/update_block/gru04/small_gru{suffix}/Tanh_output_0",
                f"/update_block/gru04/small_gru{suffix}/Add_output_0",
                f"/update_block/gru04/large_gru{suffix}/Sigmoid_output_0",
                f"/update_block/gru04/large_gru{suffix}/Sigmoid_1_output_0",
                f"/update_block/gru04/large_gru{suffix}/Tanh_output_0",
                f"/update_block/gru04/large_gru{suffix}/Add_output_0",
                f"/update_block/gru04{suffix}/Add_output_0",
                f"/update_block/disp_head/conv/conv.4{suffix}/Conv_output_0",
            ]
        )

    # These are the accumulated low-resolution coordinate/disparity updates.
    # They are unnamed top-level Add nodes in the exported ONNX graph.
    tensors.extend(f"/Add_{1252 + 5 * iter_idx}_output_0" for iter_idx in range(16))
    tensors.extend(
        [
            "/update_block/mask/mask.0/Conv_output_0",
            "/update_block/mask/mask.1/Relu_output_0",
            "/update_block/mask/mask.2/Conv_output_0",
            "/update_block/mask/mask.3/Relu_output_0",
            "/update_block_15/Mul_output_0",
            "disp",
        ]
    )
    return tensors


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input ONNX path")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument(
        "--preset",
        choices=["coarse", "update_fine"],
        default="coarse",
        help="Named tensor list to expose.",
    )
    parser.add_argument(
        "--tensor",
        action="append",
        default=[],
        help="Additional tensor name to expose. May be repeated.",
    )
    return parser.parse_args()


def _known_value_info(model):
    infos = {}
    for collection in (model.graph.input, model.graph.output, model.graph.value_info):
        for value in collection:
            infos[value.name] = value
    return infos


def main():
    args = parse_args()
    model = onnx.load(args.input)
    existing_outputs = {output.name for output in model.graph.output}
    produced_tensors = {output for node in model.graph.node for output in node.output}
    known_infos = _known_value_info(model)

    preset_tensors = {
        "coarse": COARSE_TENSORS,
        "update_fine": _update_fine_tensors(),
    }
    tensors = list(preset_tensors[args.preset])
    tensors.extend(args.tensor)

    added = []
    for tensor_name in dict.fromkeys(tensors):
        if tensor_name in existing_outputs:
            continue
        if tensor_name not in produced_tensors:
            raise ValueError(f"Tensor is not produced by the graph: {tensor_name}")
        value_info = known_infos.get(tensor_name)
        if value_info is None:
            value_info = helper.make_tensor_value_info(
                tensor_name, TensorProto.FLOAT, None
            )
        model.graph.output.append(value_info)
        existing_outputs.add(tensor_name)
        added.append(tensor_name)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    onnx.save(model, args.output, save_as_external_data=True)
    print(f"Added {len(added)} debug outputs:")
    for name in added:
        print(f"  {name}")


if __name__ == "__main__":
    main()
