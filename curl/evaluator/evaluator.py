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
            communicator = comm.get()
            self.eval_group = communicator.eval_group
            self.eval_comm_group = communicator.eval_comm_group
            logging.info(f"EvaluatorClient {communicator.get_rank()} initialized")

        def evaluator_request(self, func_name, tensor, dim=None, *args, **kwargs):
            communicator = comm.get()
            world_size = communicator.get_world_size()
            evaluators_size = communicator.get_evaluators_size()  # Number of processes
            assert (
                communicator.get_rank() < world_size
            ), "Only MPC parties communicate with the EvaluatorServers"
            if dim is None:
                dim = tensor.share.ndim - 1 # Dimension to split along
                # Split the tensor along the specified dimension
                split_size = tensor.size(dim) // evaluators_size
                # This is for corner cases where the last dimension is 1: e.g., [[1], [2], ...]
                while split_size == 0 or dim > 0:
                    dim -= 1
                    split_size = tensor.size(dim) // evaluators_size
            # Scatter: Divide data into chunks for workers
            chunks = tensor.chunk(evaluators_size, dim=dim)

            if communicator.get_rank() == 0:
                message = {
                    "function": func_name,
                    "precision": tensor.encoder.precision_bits,
                }
                for i in range(evaluators_size):
                    evaluator_rank = world_size + 1 + i
                    message["tensor_size"]= chunks[i].size()
                    communicator.send_obj(message, evaluator_rank, self.eval_comm_group)
                    logging.debug(f"Sent to Evaluator [{evaluator_rank}]")

            # Process each split asynchronously
            requests = [None] * evaluators_size
            for i in range(evaluators_size):
                evaluator_rank = world_size + 1 + i
                requests[i] = communicator.isend(chunks[i].share.contiguous(), evaluator_rank, self.eval_group)
            # Wait for all async requests to complete and retrieve messages
            for req in requests:
                req.wait()

            # Initialize local results with the correct split sizes
            results = [torch.empty_like(chunks[i]._tensor.share) for i in range(evaluators_size)]
            requests = [None] * evaluators_size
            for i in range(evaluators_size):
                evaluator_rank = world_size + 1 + i
                requests[i] = communicator.irecv(results[i], evaluator_rank, self.eval_group)
            # Wait for all async requests to complete and retrieve messages
            for req in requests:
                req.wait()

            tensor.share = torch_cat(results, dim=dim)
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

        communicator = comm.get()
        self.eval_group = communicator.eval_group
        self.device = "cpu"
        logging.info("EvaluatorServer Initialized")
        evaluator_rank = communicator.get_rank()
        world_size = communicator.get_world_size()

        # Operations supported by Fission
        fission_operations = {
            "exp": torch.exp,
            "log": torch.log,
            "reciprocal": torch.reciprocal,
            "inv_sqrt": torch.rsqrt,
            "sqrt": torch.sqrt,
            "cos": torch.cos,
            "sin": torch.sin,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "erf": torch.erf,
            "gelu": torch.nn.functional.gelu,
            "silu": torch.nn.functional.silu,
            "softmax": lambda x: torch.softmax(x, dim=-1),
            "log_softmax": lambda x: torch.log_softmax(x, dim=-1),
            "relu": torch.nn.functional.relu
        }
        try:
            while True:
                # Wait for next request from client
                # Receive the function to evaluate
                message = communicator.recv_obj(0, communicator.eval_comm_group)
                logging.debug(f"Evaluator [{evaluator_rank}] Message received: %s" % message)

                if message == "terminate":
                    logging.info("Evaluator Server {evaluator_rank - world_size - 1} shutting down.")
                    exit()
                function = str(message["function"])
                precision = message["precision"]
                tensor_size = message["tensor_size"]

                # Receive data from all the MPC nodes
                results = [torch.empty(tensor_size, dtype=torch.long) for _ in range(world_size)]
                requests = [None] * world_size
                for mpc_node in range(world_size):
                    requests[mpc_node] = communicator.irecv(results[mpc_node].contiguous(), mpc_node, self.eval_group)
                # Wait for all async requests to complete and retrieve messages
                for req in requests:
                    req.wait()

                # Reconstruct
                tensor = sum(results)
                tensor = tensor.float() / 2**precision
                if function not in fission_operations:
                    raise ValueError(f"Unsupported function {function}")
                result = fission_operations[function](tensor)

                # Secret share the result back to the MPC nodes.
                result = (result * 2**precision).long()
                requests = [None] * world_size
                for mpc_node in range(1, world_size):
                    share = generate_random_ring_element(result.size(), generator=self.generator)
                    requests[mpc_node] = communicator.isend(share, mpc_node, self.eval_group)
                    result -= share
                # Send the last share to MPC party 0.
                requests[0] = communicator.isend(result, 0, self.eval_group)
                # Wait for all async requests to complete and retrieve messages
                for req in requests:
                    req.wait()
        except RuntimeError as err:
            logging.info("Encountered Runtime error. Evaluator Server shutting down:")
            logging.info(f"{err}")
