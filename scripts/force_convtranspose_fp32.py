#!/usr/bin/env python3
"""Force selected ONNX ops to FP32 with Cast nodes.

This is intended for TensorRT BF16 experiments where individual ops have strict
same-type input requirements.
"""

import argparse

import numpy as np
import onnx
import onnx_graphsurgeon as gs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input ONNX path")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    return parser.parse_args()


def _cast_to_fp32(graph: gs.Graph, tensor: gs.Tensor, name: str) -> gs.Variable:
    out = gs.Variable(name=f"{name}_fp32", dtype=np.float32, shape=tensor.shape)
    graph.nodes.append(
        gs.Node(
            op="Cast",
            name=f"{name}_cast_fp32",
            attrs={"to": onnx.TensorProto.FLOAT},
            inputs=[tensor],
            outputs=[out],
        )
    )
    return out


def main():
    args = parse_args()
    graph = gs.import_onnx(onnx.load(args.input))

    convtranspose_changed = 0
    instancenorm_changed = 0
    for node_idx, node in enumerate(list(graph.nodes)):
        if node.op == "ConvTranspose":
            for input_idx, input_tensor in enumerate(list(node.inputs)):
                if input_idx > 2:
                    continue
                if input_tensor is None:
                    continue
                node.inputs[input_idx] = _cast_to_fp32(
                    graph,
                    input_tensor,
                    f"{node.name or 'ConvTranspose'}_input{input_idx}_{node_idx}",
                )
            convtranspose_changed += 1
        elif node.op == "InstanceNormalization":
            for input_idx, input_tensor in enumerate(list(node.inputs)):
                if input_tensor is None:
                    continue
                node.inputs[input_idx] = _cast_to_fp32(
                    graph,
                    input_tensor,
                    f"{node.name or 'InstanceNormalization'}_input{input_idx}_{node_idx}",
                )
            instancenorm_changed += 1

    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), args.output)
    print(
        f"Forced {convtranspose_changed} ConvTranspose nodes and "
        f"{instancenorm_changed} InstanceNormalization nodes to FP32 inputs."
    )


if __name__ == "__main__":
    main()
