#!/usr/bin/env python3
"""Insert ONNX Identity nodes after ConvTranspose outputs."""

import argparse

import onnx
import onnx_graphsurgeon as gs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input ONNX path")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    return parser.parse_args()


def main():
    args = parse_args()
    graph = gs.import_onnx(onnx.load(args.input))

    inserted = 0
    for node in list(graph.nodes):
        if node.op != "ConvTranspose":
            continue
        for idx, old_output in enumerate(list(node.outputs)):
            identity_output = gs.Variable(
                name=f"{old_output.name}_identity",
                dtype=old_output.dtype,
                shape=old_output.shape,
            )
            identity = gs.Node(
                op="Identity",
                name=f"{node.name or 'ConvTranspose'}_identity_{idx}",
                inputs=[old_output],
                outputs=[identity_output],
            )
            graph.nodes.append(identity)

            for consumer in list(old_output.outputs):
                if consumer is identity:
                    continue
                for input_idx, input_tensor in enumerate(consumer.inputs):
                    if input_tensor is old_output:
                        consumer.inputs[input_idx] = identity_output
            inserted += 1

    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), args.output)
    print(f"Inserted {inserted} Identity nodes after ConvTranspose outputs.")


if __name__ == "__main__":
    main()
