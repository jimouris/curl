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

Runtime = namedtuple("Runtime", "mid q1 q3")


def time_me(func=None, n_loops=10):
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


class FuncBenchmarks:
    """Benchmarks runtime and error of crypten functions against PyTorch

    Args:
        tensor_size (int or tuple): size of tensor for benchmarking runtimes
    """

    UNARY = [
        "sin",
        "cos",
        "erf",
        "exp",
        "gelu",
        "log",
        "reciprocal",
        "sigmoid",
        "silu",
        "tanh",
        "sqrt",
        "inv_sqrt",
    ]

    DOMAIN = torch.arange(start=1.01, end=10, step=0.01)
    # for exponential, sin, and cos
    TRUNCATED_DOMAIN = torch.arange(start=0.001, end=10, step=0.001)

    def __init__(self, tensor_size, device="cpu"):
        self.device = torch.device(device)
        self.tensor_size = tensor_size

        # dataframe for benchmarks
        self.df = None

    def __repr__(self):
        if self.df is not None:
            return self.df.to_string(index=False, justify="left")
        return "No Function Benchmarks"

    @staticmethod
    @time_me
    def time_func(x, func):
        """Invokes func as a method of x"""
        if func == "gelu":
            gelu = lambda x: x * (1 + (x / torch.sqrt(torch.tensor(2))).erf()) / 2
            return gelu(x)
        elif func == "silu":
            silu = lambda x: x * x.sigmoid()
            return silu(x)
        elif func == "inv_sqrt":
            inv_sqrt = lambda x: x.sqrt().reciprocal()
            return inv_sqrt(x)

        return getattr(x, func)()

    def get_runtimes(self):
        """Returns plain text and crypten runtimes"""
        x = torch.rand(self.tensor_size, device=self.device)*5 + 1
        x_enc = crypten.cryptensor(x)

        runtimes, runtimes_enc = [], []

        for func in FuncBenchmarks.UNARY:
            runtime, _ = FuncBenchmarks.time_func(x, func)
            runtimes.append(runtime)

            runtime_enc, _ = FuncBenchmarks.time_func(x_enc, func)
            runtimes_enc.append(runtime_enc)

        return runtimes, runtimes_enc

    @staticmethod
    def calc_abs_error(ref, out):
        """Computes total absolute error"""
        ref, out = ref.cpu(), out.cpu()
        if ref.dtype == torch.bool:
            errors = (out != ref).numpy().sum()
            return errors
        errors = torch.abs(out - ref).numpy()
        return errors.sum()

    @staticmethod
    def calc_relative_error(ref, out):
        """Computes average relative error"""
        ref, out = ref.cpu(), out.cpu()
        if ref.dtype == torch.bool:
            errors = (out != ref).numpy().sum() // ref.nelement()
            return errors
        errors = torch.abs((out - ref) / ref)
        # remove inf due to division by tiny numbers
        errors = errors[errors != float("inf")].numpy()
        return errors.mean()

    def call_function_on_domain(self, func):
        """Call plain text and CrypTen function on given function
        Uses DOMAIN or TRUNCATED_DOMAIN

        Returns: tuple of (plain text result, encrypted result)
        """
        DOMAIN, TRUNCATED_DOMAIN = (
            FuncBenchmarks.DOMAIN,
            FuncBenchmarks.TRUNCATED_DOMAIN,
        )
        if hasattr(DOMAIN, "to") and hasattr(TRUNCATED_DOMAIN, "to"):
            DOMAIN, TRUNCATED_DOMAIN = (
                DOMAIN.to(device=self.device),
                TRUNCATED_DOMAIN.to(device=self.device),
            )
        y = torch.rand(DOMAIN.shape, device=self.device)
        DOMAIN_enc, _ = crypten.cryptensor(DOMAIN), crypten.cryptensor(y)
        TRUNCATED_DOMAIN_enc = crypten.cryptensor(TRUNCATED_DOMAIN)

        if func in ["exp", "cos", "sin"]:
            ref, out_enc = (
                getattr(TRUNCATED_DOMAIN, func)(),
                getattr(TRUNCATED_DOMAIN_enc, func)(),
            )
        elif func in ["gelu"]:
            gelu = lambda x: x * (1 + (x / torch.sqrt(torch.tensor(2))).erf()) / 2
            ref, out_enc = gelu(DOMAIN), getattr(DOMAIN_enc, func)()
        elif func in ["silu"]:
            silu = lambda x: x * x.sigmoid()
            ref, out_enc = silu(DOMAIN), getattr(DOMAIN_enc, func)()
        elif func in ["inv_sqrt"]:
            inv_sqrt = lambda x: x.sqrt().reciprocal()
            ref, out_enc = inv_sqrt(DOMAIN), getattr(DOMAIN_enc, func)()
        elif func in FuncBenchmarks.UNARY:
            ref, out_enc = getattr(DOMAIN, func)(), getattr(DOMAIN_enc, func)()
        else:
            raise ValueError(f"{func} not supported")

        return ref, out_enc

    def get_errors(self):
        """Computes the total error of approximations"""
        abs_errors, relative_errors = [], []

        for func in FuncBenchmarks.UNARY:
            ref, out_enc = self.call_function_on_domain(func)
            out = out_enc.get_plain_text()

            abs_error = FuncBenchmarks.calc_abs_error(ref, out)
            abs_errors.append(abs_error)

            relative_error = FuncBenchmarks.calc_relative_error(ref, out)
            relative_errors.append(relative_error)

        return abs_errors, relative_errors

    def run(self):
        """Runs and stores benchmarks in self.df"""
        _runtimes, runtimes_enc = self.get_runtimes()

        abs_errors, relative_errors = self.get_errors()

        self.df = pd.DataFrame.from_dict(
            {
                "function": FuncBenchmarks.UNARY,
                "runtime": [r.mid for r in runtimes_enc],
                "total abs err.": abs_errors,
                "avg relative err.": relative_errors,
            }
        )

def run_benches(cfg_file, tensor_size, party_name):
    device = torch.device("cpu")
    logging.info("Tensor size '{}'".format(tensor_size))

    # First cold run.
    crypten.init(cfg_file, party_name=party_name)
    functions_data = cfg.config.get('functions', {})
    filtered_data = {key: value for key, value in functions_data.items() if '_method' in key}
    logging.info("\t'{}'".format(filtered_data))
    crypten.trace()

    logging.info(f"="*22 + " Without Cache " + "="*22)

    benches = FuncBenchmarks(tensor_size, device=device)
    benches.run()
    logging.info("'\n{}\n'".format(benches))
    logging.info("="*60)

    # Populate the cache.
    crypten.fill_cache()
    provider = crypten.mpc.get_default_provider()
    provider.save_cache()
    provider.load_cache()
    crypten.trace(False)

    # Run with the cache.
    logging.info(f"="*24 + " With Cache " + "="*24)
    benches = FuncBenchmarks(tensor_size, device=device)
    benches.run()
    logging.info("'\n{}\n'".format(benches))
