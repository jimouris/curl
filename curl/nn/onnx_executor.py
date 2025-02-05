# IMPORTANT:
# This file has modified versions from a pure cleartext implementation.
# Note that:
# - The original implementation would not clamp the inputs and weights.
# - The original implementation would not perform a range check before Softmax.

import logging

import numpy as np
import scipy.special
import torch
from onnx import numpy_helper

UPPER_BOUND = 1 << 16
LOWER_BOUND = -UPPER_BOUND


# Helper function to clamp int64 values to avoid overflow
def clamp_int64(x):
    return x
    max_value = UPPER_BOUND
    min_value = LOWER_BOUND
    return np.maximum(min_value, np.minimum(max_value, x))


class ONNXExecutor:
    """
    Executes an ONNX model node by node. Can be used to load inputs, initialize weights,
    and execute operations as defined in the ONNX graph.

    Example:
    model = ONNXExecutor(onnx_model)
    output = model.forward(inputs)

    Attributes:
        onnx_model: The ONNX model to execute.
        tensor_store: A dictionary to store intermediate tensor results.
    """

    def __init__(self, onnx_model):
        """
        Initialize the ONNXExecutor with the given ONNX model.

        Args:
            onnx_model: The ONNX model object.
        """
        self.onnx_model = onnx_model  # Store the ONNX model
        self.tensor_store = {}  # Store tensors (weights, inputs, intermediate results)
        self.initialize()  # Initialize with constant tensors (weights)
        self.function_ranges = {}

    def initialize(self):
        """
        Load the initializer values (constant tensors, like model weights) from the ONNX model
        into the tensor store.
        """
        # Load initializers (constants like model weights)
        for initializer in self.onnx_model.graph.initializer:
            tensor_name = initializer.name  # Get the name of the tensor
            tensor_value = numpy_helper.to_array(initializer)  # Convert to NumPy array
            self.tensor_store[tensor_name] = (
                tensor_value  # Store the tensor in tensor_store
            )

    @staticmethod
    def check_range(x, min_value, max_value):
        """
        Check if all values in the array x are within the specified range.
        """
        return np.logical_and(x >= min_value, x < max_value).all()

    # def check_range_bitwidth(self, x, function: str):
    #     """
    #     Check if all values in the array x are within the specified bitwidth.
    #     """

    #     lut_config = LookupTables().get_config()
    #     if lut_config[function]["method"] in ["bior", "haar", "split"]:
    #         bitwidth = lut_config[function]["lut_max_bits"]
    #         range_bitwidth = 1 << (bitwidth)
    #         if function not in self.function_ranges:
    #             self.function_ranges[function] = (
    #                 x.min(),
    #                 x.max(),
    #                 np.ceil(np.log2(np.abs(x).max() + 1)),
    #             )
    #         else:
    #             self.function_ranges[function] = (
    #                 min(self.function_ranges[function][0], x.min()),
    #                 max(self.function_ranges[function][1], x.max()),
    #                 max(
    #                     self.function_ranges[function][2],
    #                     np.ceil(np.log2(np.abs(x).max() + 1)),
    #                 ),
    #             )
    #         assert ONNXExecutor.check_range(
    #             x, -range_bitwidth, range_bitwidth
    #         ), f"Function: [{function}] values exceed bitwidth {bitwidth} with range {self.function_ranges[function][0]} to {self.function_ranges[function][1]}. Min required bitwidth: {self.function_ranges[function][2]}"
    #         logging.info(f"[ONNX EXECUTOR] {function} bitwidth: {bitwidth}")

    def load_inputs(self, inputs):
        """
        Load input tensors into the tensor store for execution.

        Args:
            inputs: A dictionary of input tensors (as NumPy arrays), keyed by input names.
        """
        # Store input tensors by their names
        for input_name in self.onnx_model.graph.input:
            tensor_name = input_name.name  # Get the input name from the model
            self.tensor_store[tensor_name] = inputs[
                tensor_name
            ].numpy()  # Load input tensor

    def forward(self, inputs, output_name="logits"):
        """
        Execute the model on the given inputs.

        Args:
            inputs: A dictionary of input tensors (as NumPy arrays), keyed by input names.

        Returns:
            NumPy array: The final output tensor (in this case, 'logits').
        """
        self.load_inputs(inputs)  # Load inputs into the tensor store

        # Execute each node in the graph
        for i, node in enumerate(self.onnx_model.graph.node):
            logging.debug(f"[{i}] {node.op_type}")
            self.execute_node(node)

        logging.info("LUT Ranges: ", self.function_ranges)
        # Return the final output tensor (assuming 'logits' is the output)
        return self.tensor_store[output_name]

    def execute_node_by_name(self, node_name):
        for node in self.onnx_model.graph.node:
            if node_name in node.output:
                self.execute_node(node)
                return self.tensor_store[node.output[0]]
        raise ValueError(f"Node with name '{node_name}' not found in the model")

    def get_node_inputs_by_name(self, node_name):
        for node in self.onnx_model.graph.node:
            if node_name in node.output:
                inputs = [self.tensor_store[input_name] for input_name in node.input]
                return inputs
        raise ValueError(f"Node with name '{node_name}' not found in the model")

    def execute_node(self, node):
        """
        Execute a single ONNX node and store its result.

        Args:
            node: The ONNX node to execute.
        """
        # Get the operation type of the node (e.g., 'Add', 'MatMul')
        op_type = node.op_type

        # Get the input tensors by their names from the tensor store
        inputs = [self.tensor_store[input_name] for input_name in node.input]

        for i, input_tensor in enumerate(inputs):
            logging.info(f"[{op_type}] Input {i}: {input_tensor.shape}")
        # Perform the operation based on the node's type
        if op_type == "Add":
            result = np.add(*inputs)
        elif op_type == "MatMul":
            result = np.matmul(*inputs)
        elif op_type == "Relu":
            result = np.maximum(inputs[0], 0)
        elif op_type == "Constant":
            result = clamp_int64(
                numpy_helper.to_array(node.attribute[0].t)
            )  # Load the constant tensor
        elif op_type == "Gemm":
            # Process Gemm attributes (e.g., alpha, beta, transpositions)
            alpha, beta, transA, transB = 1.0, 1.0, 0, 0
            for attr in node.attribute:
                if attr.name == "alpha":
                    alpha = attr.f
                elif attr.name == "beta":
                    beta = attr.f
                elif attr.name == "transA":
                    transA = attr.i
                elif attr.name == "transB":
                    transB = attr.i

            A = inputs[0].T if transA else inputs[0]  # Apply transposition if required
            B = inputs[1].T if transB else inputs[1]
            C = inputs[2] if len(inputs) > 2 else 0  # Bias term is optional

            # Perform Gemm (general matrix multiplication)
            result = alpha * np.dot(A, B) + beta * C
        elif op_type == "Sub":
            result = np.subtract(*inputs)
        elif op_type == "Mul":
            result = np.multiply(*inputs)
        elif op_type == "Div":
            result = np.divide(*inputs)
        elif op_type == "Neg":
            result = np.negative(inputs[0])
        elif op_type == "Abs":
            result = np.abs(inputs[0])
        elif op_type == "Max":
            result = np.maximum.reduce(inputs)
        elif op_type == "Min":
            result = np.minimum.reduce(inputs)
        elif op_type == "Exp":
            ##self.check_range_bitwidth(inputs[0], "exp")
            result = np.exp(inputs[0])
        elif op_type == "Log":
            #self.check_range_bitwidth(inputs[0], "log")
            result = np.log(inputs[0])
        elif op_type == "Sqrt":
            #self.check_range_bitwidth(inputs[0], "sqrt")
            result = np.sqrt(inputs[0])
        elif op_type == "Reciprocal":
            #self.check_range_bitwidth(inputs[0], "reciprocal")
            result = np.reciprocal(inputs[0])
        elif op_type == "Sigmoid":
            #self.check_range_bitwidth(inputs[0], "exp")
            result = 1 / (1 + np.exp(-inputs[0]))
        elif op_type == "Softmax":
            # Softmax computation over a specified axis
            axis = -1  # Default axis for softmax
            for attr in node.attribute:
                if attr.name == "axis":
                    axis = attr.i

            a = np.where(inputs[0] >= LOWER_BOUND, inputs[0], np.array(-32))
            #self.check_range_bitwidth(a, "exp")

            exp_values = np.exp(a - np.max(a, axis=axis, keepdims=True))
            # exp_values = np.exp(a)
            #self.check_range_bitwidth(exp_values, "reciprocal")
            result = exp_values / np.sum(exp_values, axis=axis, keepdims=True)
            result = torch.nn.functional.softmax(torch.tensor(a), dim=axis).numpy()
        elif op_type == "Flatten":
            # Flatten tensor starting from the specified axis
            axis = 1
            for attr in node.attribute:
                if attr.name == "axis":
                    axis = attr.i
            result = inputs[0].reshape(inputs[0].shape[:axis] + (-1,))
        elif op_type == "Concat":
            # Concatenate tensors along a specified axis
            axis = 0
            for attr in node.attribute:
                if attr.name == "axis":
                    axis = attr.i
            result = np.concatenate(inputs, axis=axis)
        elif op_type == "Reshape":
            # Reshape the tensor to the given shape
            result = np.reshape(inputs[0], inputs[1])
        elif op_type == "Transpose":
            # Transpose tensor with optional permutation of axes
            perm = None
            for attr in node.attribute:
                if attr.name == "perm":
                    perm = attr.ints
            if perm is None:
                perm = list(range(len(inputs[0].shape)))[
                    ::-1
                ]  # Reverse axis order by default
            result = np.transpose(inputs[0], axes=perm)
        elif op_type == "Pad":
            # Pad the tensor based on the provided padding scheme
            pads, mode, value = None, "constant", 0
            for attr in node.attribute:
                if attr.name == "pads":
                    pads = attr.ints
                elif attr.name == "mode":
                    mode = attr.s.decode("utf-8")
                elif attr.name == "value":
                    value = attr.f
            if pads is None:
                raise ValueError("Pads attribute is required for Pad operation")
            pad_width = [
                (pads[i], pads[i + len(pads) // 2]) for i in range(len(pads) // 2)
            ]
            result = (
                np.pad(inputs[0], pad_width, mode=mode, constant_values=value)
                if mode == "constant"
                else np.pad(inputs[0], pad_width, mode=mode)
            )
        elif op_type == "ReduceMean":
            # Reduce mean along specified axes
            axes, keepdims = None, 1
            for attr in node.attribute:
                if attr.name == "axes":
                    axes = attr.ints
                elif attr.name == "keepdims":
                    keepdims = attr.i
            result = np.mean(
                inputs[0], axis=tuple(axes) if axes else None, keepdims=bool(keepdims)
            )
        elif op_type == "ReduceSum":
            # Reduce sum along specified axes
            axes, keepdims = None, 1
            for attr in node.attribute:
                if attr.name == "axes":
                    axes = attr.ints
                elif attr.name == "keepdims":
                    keepdims = attr.i
            result = np.sum(
                inputs[0], axis=tuple(axes) if axes else None, keepdims=bool(keepdims)
            )
        elif op_type == "LeakyRelu":
            # Apply LeakyReLU with given alpha
            alpha = 0.01
            for attr in node.attribute:
                if attr.name == "alpha":
                    alpha = attr.f
            result = np.where(inputs[0] > 0, inputs[0], alpha * inputs[0])
        elif op_type == "Tanh":
            result = np.tanh(inputs[0])
        elif op_type == "Identity":
            # Pass input as output
            result = inputs[0]
        elif op_type == "Shape":
            # Return the shape of the tensor as a 1D array
            result = np.array(inputs[0].shape, dtype=np.int64)
        elif op_type == "Gather":
            data = inputs[0]
            indices = inputs[1]
            axis = 0
            for attr in node.attribute:
                if attr.name == "axis":
                    axis = attr.i
            indices = indices.astype(np.int64)
            result = np.take(data, indices, axis=axis)
        elif op_type == "Unsqueeze":
            data = inputs[0]
            axes = inputs[1]

            for axis in sorted(axes):
                data = np.expand_dims(data, axis=axis)
            result = data
        elif op_type == "Squeeze":
            data = inputs[0]
            axes = inputs[1]

            # Apply squeeze at the specified axes
            result = np.squeeze(data, axis=tuple(axes))
        elif op_type == "Slice":
            data = inputs[0]
            starts = inputs[1] if len(inputs) > 1 else None
            ends = inputs[2] if len(inputs) > 2 else None
            axes = inputs[3] if len(inputs) > 3 else None
            steps = inputs[4] if len(inputs) > 4 else None

            # If starts, ends, axes, or steps are not provided as inputs, check attributes
            if starts is None:
                starts = [
                    attr.ints for attr in node.attribute if attr.name == "starts"
                ][0]
            if ends is None:
                ends = [attr.ints for attr in node.attribute if attr.name == "ends"][0]
            if axes is None:
                axes = (
                    [attr.ints for attr in node.attribute if attr.name == "axes"][0]
                    if any(attr.name == "axes" for attr in node.attribute)
                    else range(len(starts))
                )
            if steps is None:
                steps = (
                    [attr.ints for attr in node.attribute if attr.name == "steps"][0]
                    if any(attr.name == "steps" for attr in node.attribute)
                    else [1] * len(starts)
                )

            # Convert to lists if they're numpy arrays
            starts = starts.tolist() if isinstance(starts, np.ndarray) else starts
            ends = ends.tolist() if isinstance(ends, np.ndarray) else ends
            axes = axes.tolist() if isinstance(axes, np.ndarray) else axes
            steps = steps.tolist() if isinstance(steps, np.ndarray) else steps

            # Prepare slicing tuple
            slices = [slice(None)] * data.ndim
            for i, axis in enumerate(axes):
                start = starts[i]
                end = ends[i]
                step = steps[i]

                # Handle negative indices
                if start < 0:
                    start = max(0, data.shape[axis] + start)
                if end < 0:
                    end = max(0, data.shape[axis] + end)

                # Handle special end cases
                if end >= 2**63 - 1:
                    end = data.shape[axis]

                slices[axis] = slice(start, end, step)

            # Perform slicing
            result = data[tuple(slices)]
        elif op_type == "ConstantOfShape":
            # Get the input shape
            shape = inputs[0]

            # Default value is 0.0 (float32)
            value = np.array([0.0], dtype=np.float32)

            # Check if there's a 'value' attribute
            for attr in node.attribute:
                if attr.name == "value":
                    value = numpy_helper.to_array(attr.t)

            # Ensure shape is a 1D array of integers
            if isinstance(shape, np.ndarray):
                shape = shape.astype(np.int64).flatten().tolist()
            else:
                shape = [int(dim) for dim in shape]

            # Create the constant tensor
            result = clamp_int64(np.full(shape, value.item(), dtype=value.dtype))
        elif op_type == "Equal":
            result = np.equal(*inputs)

        elif op_type == "Expand":
            shape = inputs[1]
            result = np.broadcast_to(inputs[0], shape)

        elif op_type == "Pow":
            result = np.power(*inputs)

        elif op_type == "Cast":
            to_type = None
            for attr in node.attribute:
                if attr.name == "to":
                    to_type = attr.i
            # Taken from: https://onnx.ai/onnx/intro/concepts.html#element-type
            numpy_type = {
                1: np.float32,
                2: np.uint8,
                3: np.int8,
                4: np.uint16,
                5: np.int16,
                6: np.int32,
                7: np.int64,
                9: np.bool_,
                10: np.float16,
                11: np.double,
                12: np.uint32,
                13: np.uint64,
            }.get(to_type)
            logging.info("[CT] to_type: ", numpy_type)
            if numpy_type is None:
                raise ValueError(f"Unsupported 'to' type in Cast operation: {to_type}")
            result = inputs[0].astype(numpy_type)

        elif op_type == "Where":
            condition, x, y = inputs

            # x = clamp_int64(x)
            # y = clamp_int64(y)

            result = np.where(condition, x, y)
        elif op_type == "LayerNormalization":
            X = inputs[0]  # Input tensor
            scale = inputs[1] if len(inputs) > 1 else np.ones(X.shape[-1])
            bias = inputs[2] if len(inputs) > 2 else np.zeros(X.shape[-1])

            # Default values
            axis = -1
            epsilon = 1e-5

            # Check for attributes
            for attr in node.attribute:
                if attr.name == "axis":
                    axis = attr.i
                elif attr.name == "epsilon":
                    epsilon = attr.f

            # Compute mean and variance along the specified axis
            mean = np.mean(X, axis=axis, keepdims=True)
            variance = np.var(X, axis=axis, keepdims=True)

            # Normalize
            X_normalized = (X - mean) / np.sqrt(variance + epsilon)

            # Scale and shift
            result = scale * X_normalized + bias

            # # Store results
            # self.tensor_store[node.output[0]] = Y

            # # If the node has more than one output, it's expecting mean and variance as well
            # if len(node.output) > 1:
            #     self.tensor_store[node.output[1]] = mean
            # if len(node.output) > 2:
            #     self.tensor_store[node.output[2]] = variance
        elif op_type == "Erf":
            result = scipy.special.erf(inputs[0])
        elif op_type == "Range":
            start, end, step = (x.item() if isinstance(x, np.ndarray) else x for x in inputs)
            result = torch.arange(start, end, step)
        else:
            raise NotImplementedError(f"Operation {op_type} is not implemented")

        logging.debug(f"[{op_type}] Output: {result.shape}")
        # Store the result for the output
        for output_name in node.output:
            self.tensor_store[output_name] = result
