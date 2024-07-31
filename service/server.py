import socket
import json
import subprocess


import argparse
import logging
import os

import torch
import curl
from curl.config import cfg
from examples.multiprocess_launcher import MultiProcessLauncher


def handle_client_connection(client_socket):
    # TODO: The server crashes when a second client arrives with: `Server
    # response: An error occurred: context has already been set`.

    try:
        request = client_socket.recv(1024).decode('utf-8')
        args = json.loads(request)

        # Ensure necessary keys are in the args dictionary
        required_keys = ["tensor_size", "communication", "device", "world_size"]
        for key in required_keys:
            if key not in args:
                raise ValueError(f"Missing required argument: {key}")

        if args["communication"] and crypten.cfg.mpc.provider == "TTP":
            raise ValueError("Communication statistics are not available for TTP provider")
        world_size = args["world_size"]
        args = argparse.Namespace(**args)
        launcher = MultiProcessLauncher(world_size, example, args, cfg_file=None)

        launcher.start()
        launcher.join()

        launcher.terminate()

        response = "Benchmark completed successfully."
    except Exception as e:
        response = f"An error occurred: {str(e)}"

    client_socket.send(response.encode('utf-8'))
    client_socket.close()


def example(_args):
    # Only Rank 0 will display logs.
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)

    # TODO: Replace this with an actual program
    x_enc = curl.cryptensor([1.0, 2.0, 3.0])
    y = 2.0
    y_enc = curl.cryptensor(2.0)
    z_enc1 = x_enc + y      # Public
    z_enc2 = x_enc + y_enc  # Private
    curl.print("\nPublic  addition:", z_enc1.get_plain_text())
    curl.print("Private addition:", z_enc2.get_plain_text())
    print('Done')


if __name__ == "__main__":
    port = 9999
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', port))
    server.listen(5)
    print(f"Server listening on port {port}")

    while True:
        client_sock, address = server.accept()
        print(f"Accepted connection from {address}")
        handle_client_connection(client_sock)
