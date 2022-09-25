'''
Author: Peng Bo
Date: 2022-09-25 10:44:46
LastEditTime: 2022-09-25 10:54:29
Description: 

'''
import argparse
import onnxoptimizer  # pip install onnxoptimizer
import onnx


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input model")
    parser.add_argument("--output", required=True, help="output model")
    args = parser.parse_args()
    return args


def remove_initializer_from_input():
    args = get_args()

    model = onnx.load(args.input)
    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
    optimized_model = onnxoptimizer.optimize(model, passes)
    if model.ir_version < 4:
        print("Model with ir_version below 4 requires to include initilizer in graph input")
        return

    inputs = optimized_model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in optimized_model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])
    
    onnx.save(optimized_model, args.output)


if __name__ == "__main__":
    remove_initializer_from_input()