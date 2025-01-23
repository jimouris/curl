#!/usr/bin/env python3

"""
python examples/llms/launcher.py --world_size 2 --tensor_size 1000,10 --multiprocess
"""

import argparse
import logging
import os

import curl
from curl.config import cfg
from examples.multiprocess_launcher import MultiProcessLauncher
from curl.config import cfg
import curl.communicator as comm

def get_args():
    parser = argparse.ArgumentParser(description="Curl LLM Inference")
    parser.add_argument(
        "--world_size",
        type=int,
        default=2,
        help="The number of parties to launch. Each party acts as its own process",
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
    # First cold run.
    curl.init(cfg_file, device=device)

    x_enc = curl.cryptensor([1, 2, 3, 4, 5, 6])
    print("x_enc :", x_enc.get_plain_text())

    y_enc, p = x_enc.shuffle()
    print("y_enc :", y_enc.get_plain_text())

    z_enc = y_enc.unshuffle(p)
    print("z_enc :", z_enc.get_plain_text())


def _run_experiment(args):
    # Only Rank 0 will display logs.
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)

    cfg_file = get_config(args)
    run_shuffle(cfg_file)
    print('Done')

def main():
    args = get_args()
    cfg_file = get_config(args)
    curl.cfg.load_config(cfg_file)

    if args.multiprocess:
        launcher = MultiProcessLauncher(args.world_size, _run_experiment, args, cfg_file)
        launcher.start()
        launcher.join()
        launcher.terminate()
    else:
        _run_experiment(args)

if __name__ == "__main__":
    main()
