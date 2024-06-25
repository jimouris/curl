#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import multiprocessing
import os
import uuid

import crypten
import torch

class MultiProcessLauncher:

    # run_process_fn will be run in subprocesses.
    def __init__(self, world_size, run_process_fn, fn_args=None, cfg_file=None):
        env = os.environ.copy()
        env["WORLD_SIZE"] = str(world_size)
        multiprocessing.set_start_method("spawn")
        device = torch.device(fn_args.device)

        # Use random file so multiple jobs can be run simultaneously
        INIT_METHOD = "file:///tmp/crypten-rendezvous-{}".format(uuid.uuid1())
        env["RENDEZVOUS"] = INIT_METHOD

        # Using multiple GPUs
        if fn_args.multi_gpu:
            assert (
                fn_args.world_size <= torch.cuda.device_count()
            ), f"Got {fn_args.world_size} parties, but only {torch.cuda.device_count()} GPUs found"

        self.processes = []
        for rank in range(world_size):
            if fn_args.multi_gpu:
                device = torch.device(f"cuda:{rank}")
                new_args = copy.deepcopy(fn_args)
                new_args.device = device
                print(f'Running party {rank} in {device}')
            else:
                new_args = fn_args

            process_name = "process " + str(rank)
            process = multiprocessing.Process(
                target=self.__class__._run_process,
                name=process_name,
                args=(rank, world_size, env, run_process_fn, new_args, cfg_file, device),
            )
            self.processes.append(process)

        if crypten.mpc.ttp_required():
            if fn_args.multi_gpu:
                ttp_device = torch.device(f"cuda:0")
            else:
                ttp_device = device
            new_args = copy.deepcopy(fn_args)
            new_args.device = ttp_device

            ttp_process = multiprocessing.Process(
                target=self.__class__._run_process,
                name="TTP",
                args=(
                    world_size,
                    world_size,
                    env,
                    crypten.mpc.provider.TTPServer,
                    None,
                    cfg_file,
                    ttp_device,
                ),
            )
            self.processes.append(ttp_process)

    @classmethod
    def _run_process(cls, rank, world_size, env, run_process_fn, fn_args, cfg_file=None, device=None):
        for env_key, env_value in env.items():
            os.environ[env_key] = env_value
        os.environ["RANK"] = str(rank)
        orig_logging_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.INFO)
        crypten.init(cfg_file, device=device)
        logging.getLogger().setLevel(orig_logging_level)
        if fn_args is None:
            run_process_fn()
        else:
            run_process_fn(fn_args)

    def start(self):
        for process in self.processes:
            process.start()

    def join(self):
        for process in self.processes:
            process.join()
            assert (
                process.exitcode == 0
            ), f"{process.name} has non-zero exit code {process.exitcode}"

    def terminate(self):
        for process in self.processes:
            process.terminate()
