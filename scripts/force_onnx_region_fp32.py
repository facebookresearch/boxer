#!/usr/bin/env python3
"""Force a named ONNX subgraph region to FP32 without whole-graph autocast."""

import argparse
import re
from collections import Counter

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


ACCUM_ADD_RE = (
    r"^/Add_(1252|1257|1262|1267|1272|1277|1282|1287|1292|1297|"
    r"1302|1307|1312|1317|1322|1327)$"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--pattern",
        action="append",
        default=[
            r"^/update_block/gru04",
            r"^/update_block/disp_head",
            r"^/update_block_15",
            ACCUM_ADD_RE,
        ],
        help="Regex for node names to keep in FP32. May be repeated.",
    )
    parser.add_argument(
        "--include-convtranspose",
        action="store_true",
        help="Also normalize ConvTranspose and InstanceNormalization weights/constants to FP32.",
    )
    parser.add_argument(
        "--boundary-tensor",
        action="append",
        default=[],
        help=(
            "Tensor name to cast to FP32 before matched consumers. May be repeated. "
            "This does not change non-matched consumers."
        ),
    )
    return parser.parse_args()


def node_matches(node, patterns, include_convtranspose):
    if include_convtranspose and node.op_type in {"ConvTranspose", "InstanceNormalization"}:
        return True
    return any(pattern.search(node.name) for pattern in patterns)


def tensor_to_fp32(tensor):
    array = numpy_helper.to_array(tensor).astype(np.float32)
    return numpy_helper.from_array(array, tensor.name)


def main():
    args = parse_args()
    patterns = [re.compile(pattern) for pattern in args.pattern]
    model = onnx.load(args.input)
    init_by_name = {init.name: init for init in model.graph.initializer}

    matched_nodes = [
        node for node in model.graph.node if node_matches(node, patterns, args.include_convtranspose)
    ]
    matched_names = {node.name for node in matched_nodes}

    cast_to_fp32 = 0
    constants_to_fp32 = 0
    initializer_names = set()
    op_counts = Counter(node.op_type for node in matched_nodes)
    matched_node_ids = {id(node) for node in matched_nodes}

    for node in matched_nodes:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT
                    cast_to_fp32 += 1
        elif node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT16:
                    attr.t.CopyFrom(tensor_to_fp32(attr.t))
                    constants_to_fp32 += 1

        for input_name in node.input:
            init = init_by_name.get(input_name)
            if init is not None and init.data_type == TensorProto.FLOAT16:
                initializer_names.add(input_name)

    boundary_casts = 0
    boundary_rewired = 0
    new_nodes = []
    for tensor_name in args.boundary_tensor:
        cast_output = f"{tensor_name}_force_fp32"
        cast_node_name = f"{tensor_name.strip('/').replace('/', '_')}_force_fp32"
        new_nodes.append(
            onnx.helper.make_node(
                "Cast",
                inputs=[tensor_name],
                outputs=[cast_output],
                name=cast_node_name,
                to=TensorProto.FLOAT,
            )
        )
        boundary_casts += 1
        for node in matched_nodes:
            for input_idx, input_name in enumerate(node.input):
                if input_name == tensor_name:
                    node.input[input_idx] = cast_output
                    boundary_rewired += 1

    initializers_to_fp32 = 0
    for idx, init in enumerate(model.graph.initializer):
        if init.name in initializer_names and init.data_type == TensorProto.FLOAT16:
            model.graph.initializer[idx].CopyFrom(tensor_to_fp32(init))
            initializers_to_fp32 += 1

    if new_nodes:
        rewritten_nodes = []
        inserted = set()
        for node in model.graph.node:
            rewritten_nodes.append(node)
            for new_node in new_nodes:
                if new_node.input[0] in node.output and new_node.name not in inserted:
                    rewritten_nodes.append(new_node)
                    inserted.add(new_node.name)
        for new_node in new_nodes:
            if new_node.name not in inserted:
                rewritten_nodes.append(new_node)
        del model.graph.node[:]
        model.graph.node.extend(rewritten_nodes)

    onnx.checker.check_model(model)
    onnx.save(model, args.output)

    print(f"matched_nodes={len(matched_nodes)}")
    print(f"matched_op_counts={dict(op_counts)}")
    print(f"cast_to_fp32={cast_to_fp32}")
    print(f"constants_to_fp32={constants_to_fp32}")
    print(f"initializers_to_fp32={initializers_to_fp32}")
    print(f"boundary_casts={boundary_casts}")
    print(f"boundary_rewired={boundary_rewired}")
    print(f"matched_names={len(matched_names)}")


if __name__ == "__main__":
    main()
