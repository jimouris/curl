#!/usr/bin/env python3

import logging
import timeit
from collections import namedtuple
import numpy as np
import pandas as pd
import functools

import curl
import torch
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from curl.config import cfg
import curl.communicator as comm

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
        times = []
        for _ in range(n_loops):
            start = timeit.default_timer()
            return_val = func(*args, **kwargs)
            times.append(timeit.default_timer() - start)
        mid_runtime = np.quantile(times, 0.5)
        q1_runtime = np.quantile(times, 0.25)
        q3_runtime = np.quantile(times, 0.75)
        runtime = Runtime(mid_runtime, q1_runtime, q3_runtime)
        return runtime, return_val

    return timing_wrapper


class LLMs:
    """LLM benchmarks runtime and error of curl functions against PyTorch

    Args:
        tensor_size (int or tuple): size of tensor for benchmarking runtimes
    """

    def __init__(self, model, tensor_size, device="cpu", full=True):
        from examples.llms.gpt import GPT2, GPTNeo
        from examples.llms.bert import BertTiny, BertBase, BertLarge
        from examples.llms.llama import Llama1B

        all_models = {
            'gpt2': GPT2,
            'gptneo': GPTNeo,
            'berttiny': BertTiny,
            'bertbase': BertBase,
            'bertlarge': BertLarge,
            'llama': Llama1B
        }

        self.device = torch.device(device)
        self.tensor_size = tensor_size
        self.df = None
        self.full = full
        model = model.lower()
        if model in all_models:
            m_clear = all_models[model](seq_len=tensor_size[1], full=full)
            if hasattr(m_clear, "to"):
                m_clear = m_clear.to(self.device)
            self.models = [m_clear.encrypt(src=0)]
        else:
            raise ValueError(f"Invalid model name: {model}. Choose from: {', '.join(all_models.keys())}")

    def __repr__(self):
        if self.df is not None:
            return " ".join(self.df.astype(str).values.flatten())
        return "No Function Benchmarks"

    @staticmethod
    @time_me
    def time_llm(x, model):
        return model(x)

    def get_runtimes(self):
        from examples.llms.llama import Llama1B
        """Returns plain text and curl runtimes"""
        runtimes_enc = []
        for llm in self.models:
            if self.full:
                x = torch.rand(self.tensor_size, device=self.device)
            else:
                x = torch.rand(self.tensor_size[0] * self.tensor_size[1] * llm.embed_dim, device=self.device).reshape(self.tensor_size[0], self.tensor_size[1], llm.embed_dim)
            if isinstance(llm, Llama1B):
                x = (torch.cat([torch.tensor([1]), torch.rand(self.tensor_size[1])]) * 128_000).long().to(self.device)
            x_enc = curl.cryptensor(x)

            llm.eval()

            runtime_enc, _ = LLMs.time_llm(x_enc, llm)
            runtimes_enc.append(runtime_enc)

        return runtimes_enc

    def run(self):
        """Runs and stores benchmarks in self.df"""
        runtimes_enc = self.get_runtimes()

        self.df = pd.DataFrame.from_dict(
            {
                "function": self.models,
                "runtime": [r.mid for r in runtimes_enc],
            }
        )

def run_llm(tensor_size, model, fill_cache=False, communication=False, full=True, device=None):
    rank = comm.get().get_rank()
    logging.info(f"[Party {rank}][Device] running in {device}")
    logging.info(f"[Party {rank}] Tensor size {tensor_size}")

    # First cold run.
    if communication:
        comm.get().set_verbosity(True)

    functions_data = cfg.config.get('functions', {})
    filtered_data = {
        key: dict(value)['method'] for key, value in functions_data.items() if 'method' in dict(value)
    }
    logging.info(f"[Party {rank}] Config: {filtered_data}")

    provider = curl.mpc.get_default_provider()
    if fill_cache:
        logging.info(f"[Party {rank}] Tracing requests for the cache " + "=" * 20)
        provider.trace_once()
    else:
        provider.load_cache()

    benches = LLMs(model, tensor_size, device=device, full=full)
    benches.run()

    logging.info(f"[Party {rank}] {benches}")

    if fill_cache:
        logging.info(f"[Party {rank}] Filling the cache " + "=" * 20)
        provider.fill_cache()

    if communication:
        comm.get().print_communication_stats()
        exit(0)

    logging.info(f"[Party {rank}] Done")
