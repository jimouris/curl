#!/usr/bin/env python3

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import curl
import curl.communicator as comm
import torch

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
            assert (
                comm.get().get_rank() < comm.get().get_world_size()
            ), "Only MPC parties communicate with the EvaluatorServers"
            message = {
                "function": func_name,
                "tensor": tensor.share,
                "precision": tensor.encoder._precision_bits,
                "args": args,
                "kwargs": kwargs,
            }

            ttp_rank = comm.get().get_ttp_rank()
            evaluators_size = comm.get().get_evaluators_size()
            # for evaluator in range(ttp_rank + 1, ttp_rank + 1 + evaluators_size):
            #     pass

            assert evaluators_size == 1
            evaluator_rank = ttp_rank + 1
            comm.get().send_obj(message, evaluator_rank, self.eval_group)

            curl.print(f'before empty: {tensor.size()=}, {tensor.device=}')
            result = torch.empty(tensor.size(), dtype=torch.long, device=tensor.device)

            # TODO: this is the shares, do not broadcast
            comm.get().broadcast(result, evaluator_rank, self.eval_group)

            return result
        
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

        try:
            while True:
                # Wait for next request from client

                # TODO: loop here to receive from everyone
                messages = []
                for mpc_node in range(comm.get().get_world_size()):
                    message = comm.get().recv_obj(mpc_node, self.eval_group)
                    print(f'EvaluatorServer: received {message=}')
                    messages.append(message)

                logging.info("Messages received: %s" % messages)

                message = messages[0]
                if message == "terminate":
                    logging.info("EvaluatorServer shutting down.")
                    return

                function = str(message["function"])
                print(f'EvaluatorServer: {function=}')
                precision = message["precision"]
                args = message["args"]
                kwargs = message["kwargs"]

                tensor = sum([message["tensor"] for message in messages])
                print(f'EvaluatorServer: reconstructed {tensor=}')
                tensor = tensor.float() / 2**precision
                print(f'EvaluatorServer: decoded {tensor=}')

                if function == "exp":
                    result = torch.exp(tensor)
                elif function == "log":
                    result = torch.log(tensor)
                elif function == "reciprocal":
                    result = torch.reciprocal(tensor)
                elif function == "inv_sqrt":
                    result = torch.rsqrt(tensor)
                elif function == "sqrt":
                    result = torch.sqrt(tensor)
                elif function == "cos":
                    result = torch.cos(tensor)
                elif function == "sin":
                    result = torch.sin(tensor)
                elif function == "sigmoid":
                    result = torch.sigmoid(tensor)
                elif function == "tanh":
                    result = torch.tanh(tensor)
                elif function == "erf":
                    result = torch.erf(tensor)
                elif function == "gelu":
                    result = torch.nn.functional.gelu(tensor)
                elif function == "silu":
                    result = torch.nn.functional.silu(tensor)
                elif function == "softmax":
                    result = torch.softmax(tensor, dim=-1)
                elif function == "log_softmax":
                    result = torch.log_softmax(tensor, dim=-1)
                elif function == "relu":
                    result = torch.nn.functional.relu(tensor)
                else:
                    raise ValueError("Unsupported function %s" % function)
                result = (result * 2**precision).long()
                comm.get().broadcast(result, evaluator_rank, self.eval_group)
        except RuntimeError as err:
            logging.info("Encountered Runtime error. EvaluatorServer shutting down:")
            logging.info(f"{err}")

