#!/usr/bin/env python3

import argparse
import logging
import os

import curl
from curl.config import cfg
from examples.multiprocess_launcher import MultiProcessLauncher


'''
python examples/llms/lambada_launcher.py --world_size 2 --multiprocess --evaluator_size 2
'''

def get_args():
    def tuple_type(s):
        try:
            # Split the string into integers
            elements = tuple(map(int, s.split(',')))
            return elements
        except ValueError:
            # Raise an error if parsing fails
            raise argparse.ArgumentTypeError("Tuple format must be integers separated by commas")


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
        "-s",
        "--tensor_size",
        type=tuple_type,
        default=(10, 10),
        help="The size of the tensors as a tuple of integers separated by commas (e.g., '100,100,50')",
    )
    parser.add_argument(
        "--multiprocess",
        default=False,
        action="store_true",
        help="Run example in multiprocess mode",
    )
    parser.add_argument(
        "--communication",
        default=False,
        action="store_true",
        help="Print communication statistics",
    )
    parser.add_argument(
        "--fill-cache",
        default=False,
        action="store_true",
        help="Populate the cache and run with it",
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
    if args.evaluator_size:
        logging.info("Using Fission config")
        cfg_file = cfg_file.replace("default", "fission")
    else:
        logging.info("Using LUTs Config:")
    return cfg_file

def _run_experiment(args):
    # only import here to initialize curl within the subprocesses
    from examples.llms.llms_lambada import evaluate_lambada_curl

    # Only Rank 0 will display logs.
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)

    cfg_file = get_config(args)
    evaluate_lambada_curl(cfg_file, args.fill_cache, args.communication, args.device)

    print('Done')

def main():
    args = get_args()
    cfg_file = get_config(args)
    curl.cfg.load_config(cfg_file)

    if args.communication and cfg.mpc.provider == "TTP":
        raise ValueError("Communication statistics are not available for TTP provider")

    if args.multiprocess:
        launcher = MultiProcessLauncher(args.world_size, args.evaluator_size, _run_experiment, args, cfg_file)
        launcher.start()
        launcher.join()
        launcher.terminate()
    else:
        _run_experiment(args)

def main_clear():
    from llms_lambada import evaluate_lambada_clear
    evaluate_lambada_clear()

if __name__ == "__main__":
    main()
    # main_clear()
