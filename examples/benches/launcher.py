#!/usr/bin/env python3

"""
python examples/benches/launcher.py --world_size 2 --tensor_size 1000,10 --multiprocess
"""

import argparse
import logging
import os

from examples.multiprocess_launcher import MultiProcessLauncher


def get_args():
    def tuple_type(s):
        try:
            # Split the string into integers
            elements = tuple(map(int, s.split(',')))
            return elements
        except ValueError:
            # Raise an error if parsing fails
            raise argparse.ArgumentTypeError("Tuple format must be integers separated by commas")


    parser = argparse.ArgumentParser(description="CrypTen Cifar Training")
    parser.add_argument(
        "--world_size",
        type=int,
        default=2,
        help="The number of parties to launch. Each party acts as its own process",
    )
    parser.add_argument(
        "-s",
        "--tensor_size",
        type=tuple_type,
        default=(100, 100),
        help="The size of the tensors as a tuple of integers separated by commas (e.g., '100,100,50')",
    )
    parser.add_argument(
        "--multiprocess",
        default=False,
        action="store_true",
        help="Run example in multiprocess mode",
    )
    args = parser.parse_args()
    return args

def _run_experiment(args):
    # only import here to initialize crypten within the subprocesses
    from examples.benches.benches import run_benches

    # Only Rank 0 will display logs.
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)

    run_benches(args.tensor_size)

    print('Done')

def main(run_experiment):
    args = get_args()

    if args.multiprocess:
        launcher = MultiProcessLauncher(args.world_size, run_experiment, args)
        launcher.start()
        launcher.join()
        launcher.terminate()
    else:
        run_experiment(args)


if __name__ == "__main__":
    main(_run_experiment)