#!/usr/bin/env python3

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import curl
import curl.communicator as comm
import torch

from curl.common.rng import generate_random_ring_element
from curl.common.util import torch_cat

'''
Fission Architecture

N = world size
E = eval size

0: MPC Party 0
1: MPC Party 1
...
N-1: MPC Party N-1
N: TTP
N+1: Evaluator 0
N+2: Evaluator 1
...
N+E: Evaluator E-1
'''

class EvaluatorClient:
    __instance = None

    class ___EvaluatorClient:
        """Singleton class"""

        def __init__(self):
            # Initialize connection
            self.eval_group = comm.get().eval_group
            logging.info(f"EvaluatorClient {comm.get().get_rank()} initialized")

        def evaluator_request(self, func_name, tensor, *args, **kwargs):
            world_size = comm.get().get_world_size()
            assert (
                comm.get().get_rank() < world_size
            ), "Only MPC parties communicate with the EvaluatorServers"
            n = comm.get().get_evaluators_size()  # Number of processes
            dim = tensor.share.ndim - 1  # Dimension to split along
            # Split the tensor along the specified dimension
            split_size = tensor.size(dim) // n
            # This is for corner cases where the last dimension is 1: e.g., [[1], [2], ...]
            if split_size == 0:
                dim -= 1
                split_size = tensor.size(dim) // n
            split = tensor.split(split_size, dim=dim)

            # Initialize local results with the correct split sizes
            results = [0] * n
            # Process each split sequentially
            for i in range(n):
                print(f"Processing split {i}... {split[i]}")
                evaluator_rank = world_size + 1 + i

                message = {
                    "function": func_name,
                    "tensor": split[i].share,
                    "args": args,
                    "kwargs": kwargs,
                }
                if comm.get().get_rank() == 0:
                    message["precision"] = tensor.encoder.precision_bits

                comm.get().send_obj(message, evaluator_rank, self.eval_group)

            for i in range(n):
                evaluator_rank = world_size + 1 + i
                result = torch.empty(tensor.size(), dtype=torch.long, device=tensor.device)
                results[i] = comm.get().recv(result, evaluator_rank, self.eval_group)
                curl.print(f'EvaluatorClient: received {results[i]=}')
            tensor.share = torch_cat(results, dim=dim)
            curl.print(f'EvaluatorClient: tensor {tensor=}')
            return tensor
        
    @staticmethod
    def _init():
        """Initializes a Evaluator client that sends requests"""
        if EvaluatorClient.__instance is None:
            EvaluatorClient.__instance = EvaluatorClient.___EvaluatorClient()

    @staticmethod
    def uninit():
        """Uninitializes an Evaluator client"""
        del EvaluatorClient.__instance
        EvaluatorClient.__instance = None

    @staticmethod
    def get():
        """Returns the instance of the EvaluatorClient"""
        if EvaluatorClient.__instance is None:
            raise RuntimeError("EvaluatorClient is not initialized")

        return EvaluatorClient.__instance

    
class EvaluatorServer:
    TERMINATE = -1

    def __init__(self):
        """Initializes an Evaluator server that receives requests"""
        self.generator = torch.Generator()

        # Initialize connection
        logging.info("EvaluatorServer: Initializing...")

        env_vars = {}
        for key in ["distributed_backend", "rendezvous", "world_size", "rank"]:
            if key.upper() not in os.environ:
                raise ValueError("Environment variable %s must be set." % key)
            env_vars[key.lower()] = os.environ[key.upper()]

        logging.info("EvaluatorServer: before crypten init.")
        logging.info(f"EvaluatorServer: env: {env_vars}")
        curl.init()
        logging.info("EvaluatorServer: crypten init done.")

        self.eval_group = comm.get().eval_group
        self.device = "cpu"
        logging.info("EvaluatorServer Initialized")
        evaluator_rank = comm.get().get_rank()
        world_size = comm.get().get_world_size()
        try:
            while True:
                # Wait for next request from client

                # TODO(memo, jimouris): loop here to receive from everyone
                messages = []
                for mpc_node in range(world_size):
                    message = comm.get().recv_obj(mpc_node, self.eval_group)
                    print(f'Evaluator Server({evaluator_rank - world_size - 1}): received {message=}')
                    messages.append(message)

                logging.info("Messages received: %s" % messages)

                message = messages[0]
                if message == "terminate":
                    logging.info("Evaluator Server({evaluator_rank - world_size - 1}) shutting down.")
                    return

                function = str(message["function"])
                print(f'Evaluator Server({evaluator_rank - world_size - 1}): {function=}')
                precision = message["precision"]
                args = message["args"]
                kwargs = message["kwargs"]

                # Reconstruct
                tensor = sum([message["tensor"] for message in messages])
                print(f'Evaluator Server({evaluator_rank - world_size - 1}): reconstructed {tensor=}')
                tensor = tensor.float() / 2**precision
                print(f'Evaluator Server({evaluator_rank - world_size - 1}): decoded {tensor=}')

                match function:
                    case "exp":
                        result = torch.exp(tensor)
                    case "log":
                        result = torch.log(tensor)
                    case "reciprocal":
                        result = torch.reciprocal(tensor)
                    case "inv_sqrt":
                        result = torch.rsqrt(tensor)
                    case "sqrt":
                        result = torch.sqrt(tensor)
                    case "cos":
                        result = torch.cos(tensor)
                    case "sin":
                        result = torch.sin(tensor)
                    case "sigmoid":
                        result = torch.sigmoid(tensor)
                    case "tanh":
                        result = torch.tanh(tensor)
                    case "erf":
                        result = torch.erf(tensor)
                    case "gelu":
                        result = torch.nn.functional.gelu(tensor)
                    case "silu":
                        result = torch.nn.functional.silu(tensor)
                    case "softmax":
                        result = torch.softmax(tensor, dim=-1)
                    case "log_softmax":
                        result = torch.log_softmax(tensor, dim=-1)
                    case "relu":
                        result = torch.nn.functional.relu(tensor)
                    case _:
                        raise ValueError("Unsupported function %s" % function)

                print(f'Evaluator Server({evaluator_rank - world_size - 1}): result {result=}')
                # Secret share the result back to the MPC nodes.
                result = (result * 2**precision).long()
                print(f'Evaluator Server({evaluator_rank - world_size - 1}): result {result=}')
                for mpc_node in range(1, world_size):
                    share = generate_random_ring_element(result.size(), generator=self.generator)
                    comm.get().send(share, mpc_node, self.eval_group)
                    result -= share
                    print(f'Evaluator Server {mpc_node}: share {share=}')
                # Send the last share to MPC party 0.
                print(f'Evaluator Server({evaluator_rank - world_size - 1}): result {result=}')
                comm.get().send(result, 0, self.eval_group)
        except RuntimeError as err:
            logging.info("Encountered Runtime error. Evaluator Server shutting down:")
            logging.info(f"{err}")

