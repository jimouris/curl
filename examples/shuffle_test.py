#!/usr/bin/env python3

"""
python examples/shuffle_test.py --multiprocess
"""

import argparse
import logging
import os

import curl
from curl.evaluator.evaluator import EvaluatorClient
from multiprocess_launcher import MultiProcessLauncher

def get_args():
    parser = argparse.ArgumentParser(description="Curl LLM Inference")
    parser.add_argument(
        "--world_size",
        type=int,
        default=2,
        help="The number of parties to launch. Each party acts as its own process",
    )
    parser.add_argument(
        "--evaluator_size",
        "-es",
        type=int,
        default=0,
        help="The number of eval parties to launch. Each party acts as its own process",
    )
    parser.add_argument(
        "--multiprocess",
        default=False,
        action="store_true",
        help="Run example in multiprocess mode",
    )
    parser.add_argument(
        "--device",
        "-d",
        required=False,
        default="cpu",
        help="the device to run the benchmarks",
    )
    parser.add_argument(
        "--multi-gpu",
        "-mg",
        required=False,
        default=False,
        action="store_true",
        help="use different gpu for each party. Will override --device if selected",
    )
    args = parser.parse_args()
    return args

def get_config(args):
    cfg_file = curl.cfg.get_default_config_path()
    return cfg_file

def run_shuffle(cfg_file, device=None):
    curl.init(cfg_file, device=device)

    tests = [
        curl.cryptensor([1, 2, 3, 4]),
        curl.cryptensor([[1, 2, 3, 4]]),
        curl.cryptensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
        curl.cryptensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        curl.cryptensor([[1], [2], [3], [4], [5], [6], [7], [8]]),
    ]
    for x_enc in tests:
        curl.print("x_enc :", x_enc.get_plain_text())
        y_enc = x_enc.gelu()
        curl.print("y_enc :", y_enc)
        curl.print("y :", y_enc.get_plain_text())
        curl.print(y_enc.share.shape, x_enc.shape)
        assert y_enc.share.shape == x_enc.shape
        curl.print("----")

def _run_experiment(args):
    # Only Rank 0 will display logs.
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)

    cfg_file = get_config(args)
    run_shuffle(cfg_file)
    curl.print('Done')

def main():
    args = get_args()
    cfg_file = get_config(args)
    curl.cfg.load_config(cfg_file)

    if args.multiprocess:
        launcher = MultiProcessLauncher(args.world_size, args.evaluator_size, _run_experiment, args, cfg_file)
        launcher.start()
        launcher.join()
        launcher.terminate()
    else:
        _run_experiment(args)

if __name__ == "__main__":
    main()
