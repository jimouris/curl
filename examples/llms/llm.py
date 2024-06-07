#!/usr/bin/env python3

import logging
import timeit
from collections import namedtuple
import numpy as np
import pandas as pd
import torch
import functools

import crypten
import torch
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from crypten.config import cfg
import crypten.communicator as comm

Runtime = namedtuple("Runtime", "mid q1 q3")

def time_me(func=None, n_loops=1):
    """Decorator returning average runtime in seconds over n_loops

    Args:
        func (function): invoked with given args / kwargs
        n_loops (int): number of times to invoke function for timing

    Returns: tuple of (time in seconds, inner quartile range, function return value).
    """
    if func is None:
        return functools.partial(time_me, n_loops=n_loops)

    @functools.wraps(func)
    def timing_wrapper(*args, **kwargs):
        return_val = func(*args, **kwargs)
        times = []
        for _ in range(n_loops):
            start = timeit.default_timer()
            func(*args, **kwargs)
            times.append(timeit.default_timer() - start)
        mid_runtime = np.quantile(times, 0.5)
        q1_runtime = np.quantile(times, 0.25)
        q3_runtime = np.quantile(times, 0.75)
        runtime = Runtime(mid_runtime, q1_runtime, q3_runtime)
        return runtime, return_val

    return timing_wrapper


class LLMs:
    """LLM benchmarks runtime and error of crypten functions against PyTorch

    Args:
        tensor_size (int or tuple): size of tensor for benchmarking runtimes
    """

    def __init__(self, model, tensor_size, device="cpu", full=True):
        from examples.llms.gpt import GPT2, GPTNeo
        from examples.llms.bert import BertTiny, BertBase, BertLarge

        all_models = {
            'gpt2': GPT2,
            'gptneo': GPTNeo,
            'berttiny': BertTiny,
            'bertbase': BertBase,
            'bertlarge': BertLarge
        }

        self.device = torch.device(device)
        self.tensor_size = tensor_size
        self.df = None
        self.full = full
        model = model.lower()
        if model is None or model == 'all':
            self.models = [m(seq_len=tensor_size[1], full=full).encrypt(src=0) for m in all_models.values()]
        elif model in all_models:
            self.models = [all_models[model](seq_len=tensor_size[1], full=full).encrypt(src=0)]
        else:
            raise ValueError(f"Invalid model name: {model}. Choose from: {', '.join(all_models.keys())}")

    def __repr__(self):
        if self.df is not None:
            return self.df.to_string(index=False, justify="left")
        return "No Function Benchmarks"

    @staticmethod
    @time_me
    def time_llm(x, model):
        return model(x)

    def get_runtimes(self):
        """Returns plain text and crypten runtimes"""

        runtimes, runtimes_enc = [], []
        for llm in self.models:
            if self.full:
                x = torch.rand(self.tensor_size, device=self.device)
            else:
                x = torch.rand(self.tensor_size[0] * self.tensor_size[1] * llm.embed_dim, device=self.device).reshape(self.tensor_size[0], self.tensor_size[1], llm.embed_dim)
            x_enc = crypten.cryptensor(x)

            llm.eval()

            # runtime, _ = LLMs.time_llm(x, llm)
            # runtimes.append(runtime)

            runtime_enc, _ = LLMs.time_llm(x_enc, llm)
            runtimes_enc.append(runtime_enc)

        return runtimes, runtimes_enc

    def run(self):
        """Runs and stores benchmarks in self.df"""
        _runtimes, runtimes_enc = self.get_runtimes()

        self.df = pd.DataFrame.from_dict(
            {
                "function": self.models,
                "runtime": [r.mid for r in runtimes_enc],
            }
        )

def run_llm(cfg_file, tensor_size, party_name, model, with_cache=False, verbose=False, full=True):
    device = torch.device("cpu")
    logging.info("Tensor size '{}'".format(tensor_size))

    # First cold run.
    crypten.init(cfg_file, party_name=party_name)
    if verbose:
        comm.get().set_verbosity(True)

    functions_data = cfg.config.get('functions', {})
    filtered_data = {key: value for key, value in functions_data.items() if '_method' in key}
    logging.info("\t'{}'".format(filtered_data))
    if with_cache:
        crypten.trace()

    logging.info(f"="*22 + " Without Cache " + "="*22)

    benches = LLMs(model, tensor_size, device=device, full=full)
    benches.run()
    logging.info("'\n{}\n'".format(benches))
    logging.info("="*60)

    if verbose:
        comm.get().print_communication_stats()

    if with_cache:
        if verbose:
            comm.get().reset_communication_stats()

        # Populate the cache.
        crypten.fill_cache()
        provider = crypten.mpc.get_default_provider()
        provider.save_cache()
        provider.load_cache()
        crypten.trace(False)

        # Run with the cache.
        logging.info(f"="*24 + " With Cache " + "="*24)
        benches = LLMs(model, tensor_size, device=device, full=full)
        benches.run()
        logging.info("'\n{}\n'".format(benches))
        if verbose:
            comm.get().print_communication_stats()
