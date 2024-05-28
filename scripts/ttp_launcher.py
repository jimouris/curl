#!/usr/bin/env python3

import crypten
# import argparse

# parser = argparse.ArgumentParser(description="Curl TTP launcher")
# parser.add_argument(
#     "--world_size",
#     type=int,
#     default=2,
#     help="The number of parties to launch. Each party acts as its own process",
# )
# parser.add_argument(
#     "--rendezvous",
#     default="file://",
#     type=str,
#     help="rendezvous file",
# )

def main():
    # args = parser.parse_args()
    crypten.mpc.provider.TTPServer()

if __name__ == "__main__":
    main()

