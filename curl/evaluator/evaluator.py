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
from curl.config import cfg

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
            self.eval_groups = communicator.eval_groups
            self.eval_comm_group = communicator.eval_comm_group
            logging.info(f"[Party {communicator.get_rank()}] Evaluator Client initialized")

        def evaluator_request(self, func_name, tensor, *args, **kwargs):
            communicator = comm.get()
            world_size = communicator.get_world_size()
            evaluators_size = communicator.get_evaluators_size()  # Number of processes
            mpc_party_rank = communicator.get_rank()
            assert (
                communicator.get_rank() < world_size
            ), "Only MPC parties communicate with the EvaluatorServers"
            if communicator.ttp_initialized:
                world_size += 1

            # Scatter: Divide data into chunks for workers
            chunks = tensor.chunk(evaluators_size)
            chunks = list(chunks) + [curl.cryptensor([]) for _ in range(evaluators_size - len(chunks))]
            if mpc_party_rank == 0:
                message = {
                    "function": func_name,
                    "scale": tensor.encoder.scale,
                }
                for i in range(evaluators_size):
                    evaluator_rank = world_size + i
                    message["tensor_size"] = chunks[i].size()
                    communicator.send_obj(message, evaluator_rank, self.eval_comm_group)

            # Process each split asynchronously
            communicator.broadcast_parallel(chunks, [mpc_party_rank]*len(chunks), self.eval_groups[mpc_party_rank])

            # Initialize local results with the correct split sizes
            results = [torch.empty_like(chunks[i]._tensor.share, device=tensor.device) for i in range(evaluators_size)]
            communicator.broadcast_parallel(results, [world_size + i for i in range(evaluators_size)], self.eval_groups[mpc_party_rank])

            tensor.share = torch_cat(results)
            tensor.encoder._precision_bits = cfg.encoder.precision_bits
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
        self.cfg_file = curl.cfg.get_default_config_path()
        evaluator_rank = comm.get().get_rank()

        torch.set_num_threads(1)

        # Initialize connection
        logging.info(f"[Evaluator {evaluator_rank}]: Initializing...")
        env_vars = {}
        for key in ["distributed_backend", "rendezvous", "world_size", "rank"]:
            if key.upper() not in os.environ:
                raise ValueError("Environment variable %s must be set." % key)
            env_vars[key.lower()] = os.environ[key.upper()]
        communicator = comm.get()
        self.eval_groups = communicator.eval_groups

        # Determine device
        self.device = "cpu"
        logging.info(f"[Evaluator {evaluator_rank}] Initialized with device: {self.device}")
        evaluator_rank = communicator.get_rank()
        world_size = communicator.get_world_size()
        ttp = 0
        if communicator.ttp_initialized:
            ttp = 1

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
                    logging.info(f"Evaluator Server {evaluator_rank - world_size} shutting down.")
                    exit()
                function = str(message["function"])
                scale = message["scale"]
                tensor_size = message["tensor_size"]

                # Receive data from all the MPC nodes
                results = [torch.empty(tensor_size, dtype=torch.long) for _ in range(world_size)]
                communicator.broadcast_parallel(
                    results,
                    [mpc_node for mpc_node in range(world_size)],
                    [ self.eval_groups[mpc_node][evaluator_rank-world_size-ttp ]
                    for mpc_node in range(world_size)]
                )

                # Reconstruct
                tensor = sum(results)
                tensor = tensor.float() / scale
                if function == "layernorm":
                    mean = tensor.mean(-1, keepdims=True)
                    variance = tensor.var(-1, keepdims=True)
                    inv_var = 1.0 / torch.sqrt(variance + 1e-05)
                    inv_var = inv_var.reshape(mean.shape)
                    # compute z-scores:
                    result = (tensor - mean) * inv_var
                elif function in fission_operations:
                    result = fission_operations[function](tensor)
                else:
                    raise ValueError(f"Unsupported function {function}")

                # Secret share the result back to the MPC nodes.
                results = [(result * 2**cfg.encoder.precision_bits).long()]
                for mpc_node in range(1, world_size):
                    share = generate_random_ring_element(results[0].size(), generator=self.generator)
                    results[0] -= share
                    results.append(share)

                communicator.broadcast_parallel(
                    results,
                    [evaluator_rank]*len(results),
                    [self.eval_groups[mpc_node][evaluator_rank-world_size-ttp]
                    for mpc_node in range(world_size)]
                )


        except RuntimeError as err:
            logging.info("Encountered Runtime error. Evaluator Server shutting down:")
            logging.info(f"{err}")
