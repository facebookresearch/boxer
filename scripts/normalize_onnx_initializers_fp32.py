#!/usr/bin/env python3

import argparse
from collections import Counter

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert floating point ONNX initializers to FP32."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--types",
        nargs="+",
        default=["fp16"],
        choices=["fp16"],
        help="Initializer types to convert to FP32.",
    )
    parser.add_argument(
        "--fp16-casts-to-fp32",
        action="store_true",
        help="Rewrite Cast(to=FLOAT16) nodes to Cast(to=FLOAT).",
    )
    parser.add_argument(
        "--fp16-constants-to-fp32",
        action="store_true",
        help="Rewrite Constant tensor values from FLOAT16 to FLOAT.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = onnx.load(args.input)
    before = Counter(init.data_type for init in model.graph.initializer)
    convert_fp16 = "fp16" in args.types

    converted = 0
    for idx, init in enumerate(model.graph.initializer):
        if convert_fp16 and init.data_type == TensorProto.FLOAT16:
            array = numpy_helper.to_array(init).astype(np.float32)
            model.graph.initializer[idx].CopyFrom(numpy_helper.from_array(array, init.name))
            converted += 1

    cast_converted = 0
    const_converted = 0
    for node in model.graph.node:
        if args.fp16_casts_to_fp32 and node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT
                    cast_converted += 1
        if args.fp16_constants_to_fp32 and node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT16:
                    array = numpy_helper.to_array(attr.t).astype(np.float32)
                    attr.t.CopyFrom(numpy_helper.from_array(array))
                    const_converted += 1

    after = Counter(init.data_type for init in model.graph.initializer)
    onnx.save(model, args.output)
    print(f"initializers_converted={converted}")
    print(f"casts_converted={cast_converted}")
    print(f"constants_converted={const_converted}")
    print(f"before={dict(before)}")
    print(f"after={dict(after)}")


if __name__ == "__main__":
    main()
