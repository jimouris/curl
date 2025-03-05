#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import curl
import curl.communicator as comm
import torch
import warnings


class TupleProvider:
    TRACEABLE_FUNCTIONS = [
        "generate_additive_triple",
        "square",
        "generate_binary_triple",
        "wrap_rng",
        "B2A_rng",
        "generate_one_hot",
        "egk_trunc_pr_rng"
    ]

    _DEFAULT_CACHE_PATH = os.path.normpath(os.path.join(__file__, "../tuple_cache/"))
    CACHE_SAVE_BATCH_SIZE = 1000  # Save cache every CACHE_SAVE_BATCH_SIZE requests

    def __init__(self, device=None):
        self.tracing = False
        self.request_cache = []
        self.tuple_cache = {}
        self.device = device

    @property
    def rank(self):
        return comm.get().get_rank()

    def _get_request_path(self, prefix=None):
        if prefix is None:
            prefix = self._DEFAULT_CACHE_PATH
        return prefix + f"/request_cache-{self.rank}"

    def _get_tuple_path(self, prefix=None):
        if prefix is None:
            prefix = self._DEFAULT_CACHE_PATH
        return prefix + f"/tuple_cache-{self.rank}"

    def trace(self, tracing=True):
        """Sets tracing attribute.

        When tracing is True, provider caches all tuple requests.
        When tracing is False, provider attempts to load tuples from cache.
        """
        self.tracing = tracing

    def trace_once(self):
        """Sets tracing attribute True only if the request cache is empty.
        If `trace_once()` is called again, it sets tracing attribute to False
        """
        untraced = len(self.request_cache) == 0
        self.trace(tracing=untraced)

    def _save_requests(self, filepath=None):
        if len(self.request_cache) == 0:
            curl.log("Request cache not saved - cache is empty")
            return
        filepath = self._get_request_path(prefix=filepath)
        torch.save(self.request_cache, filepath)
        self.request_cache = []

    def _load_requests(self, filepath=None):
        filepath = self._get_request_path(prefix=filepath)
        if os.path.exists(filepath):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                self.request_cache = torch.load(filepath, weights_only=False)
        else:
            curl.log(f"Cache requests not loaded - File `{filepath}` not found")

    def _save_tuples(self, filepath=None):
        """Saves each batch of tuple cache to a separate file."""
        if len(self.tuple_cache) == 0:
            curl.log("Tuple cache not saved - cache is empty")
            return
        filepath = self._get_tuple_path(prefix=filepath)
        # Ensure directory exists
        os.makedirs(filepath, exist_ok=True)

        existing_files = [f for f in os.listdir(filepath) if f.startswith("tuple_batch_")]
        next_index = len(existing_files)  # New file index
        batch_file = os.path.join(filepath, f"tuple_batch_{next_index}.pt")

        # Convert generators to lists before saving
        tensor_cache = {}
        for key, value in self.tuple_cache.items():
            tensor_cache[key] = list(value)
        torch.save(tensor_cache, batch_file)
        curl.log(f"Tuple cache batch saved to {batch_file}")
        self.tuple_cache.clear()  # Clear memory after saving

    def _load_tuples(self, filepath=None):
        """Loads all batch files and reconstructs the tuple cache."""
        filepath = self._get_tuple_path(prefix=filepath)
        if not os.path.exists(filepath):
            curl.log(f"Tuple cache directory `{filepath}` not found")
            return

        batch_files = sorted([f for f in os.listdir(filepath) if f.startswith("tuple_batch_")])
        self.tuple_cache = {}
        for batch_file in batch_files:
            batch_path = os.path.join(filepath, batch_file)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    batch_data = torch.load(batch_path, weights_only=False)
                for key, values in batch_data.items():
                    # curl.log(f"Loading cache key: {key}")
                    if key not in self.tuple_cache:
                        self.tuple_cache[key] = values
            except Exception as e:
                curl.log(f"Error loading {batch_path}: {e}")

        curl.log(f"Loaded {len(batch_files)} tuple cache batches with {len(self.tuple_cache)} total entries")

    def _save_cache(self, filepath=None):
        """Saves request and tuple cache to a file.

        args:
            filepath - base filepath for cache folder (default: "provider/tuple_cache/")
        """
        self._save_requests(filepath=filepath)
        self._save_tuples(filepath=filepath)

    def load_cache(self, filepath=None):
        """Loads request and tuple cache from a file.

        args:
            filepath - base filepath for cache folder (default: "provider/tuple_cache/")
        """
        self._load_requests(filepath=filepath)
        self._load_tuples(filepath=filepath)

    def __getattribute__(self, func_name):
        """Deals with caching logic"""
        if func_name not in TupleProvider.TRACEABLE_FUNCTIONS:
            return object.__getattribute__(self, func_name)

        # Trace requests while tracing
        if self.tracing:
            def func_with_trace(*args, **kwargs):
                request = (func_name, args, kwargs)  # Save full request for tracing
                self.request_cache.append(request)
                return object.__getattribute__(self, func_name)(*args, **kwargs)
            return func_with_trace

        # If the cache is empty, call function directly
        if len(self.tuple_cache) == 0:
            return object.__getattribute__(self, func_name)

        # Return results from cache if available
        def func_from_cache(*args, **kwargs):
            request = (func_name, args)  # Ignore kwargs for cache lookup
            # curl.log(f"Checking cache for request: {request}")
            # curl.log(f"Available cache keys: {list(self.tuple_cache.keys())}")
            # Read from cache
            if request in self.tuple_cache.keys():
                # Move cached ArithmeticSharedTensor to appropriate device
                device = kwargs.get('device', self.device)
                if device is None:
                    device = "cpu"
                return (
                    r.to(device) for r in self.tuple_cache[request]
                )
            # Cache miss
            return object.__getattribute__(self, func_name)(*args, **kwargs)
        return func_from_cache

    def remove_cache(self):
        # Remove previous request cache if it exists
        filepath = self._get_request_path()
        if os.path.exists(filepath):
            os.remove(filepath)
            curl.log(f"Removed previous request cache: {filepath}")
        else:
            curl.log(f"Request cache not removed - File `{filepath}` not found")

        # Remove tuple cache files iteratively
        filepath = self._get_tuple_path()
        if os.path.exists(filepath):
            batch_files = sorted([f for f in os.listdir(filepath) if f.startswith("tuple_batch_")])
            for batch_file in batch_files:
                batch_path = os.path.join(filepath, batch_file)
                try:
                    os.remove(batch_path)
                    curl.log(f"Removed tuple cache file: {batch_path}")
                except Exception as e:
                    curl.log(f"Error removing {batch_path}: {e}")
        else:
            curl.log(f"Tuple cache not removed - File `{filepath}` not found")

        curl.log(f"Completed tuple cache cleanup.")

    # TODO: parallelize / async this
    def fill_cache(self):
        """Fills tuple_cache with tuples requested in the request_cache and saves in batches."""
        self.remove_cache()
        batch_count = 0
        for request in self.request_cache:
            func_name, args, kwargs = request  # Unpack full request
            result = object.__getattribute__(self, func_name)(*args, **kwargs)

            hashable_request = (func_name, args)  # Ignore kwargs for cache key
            # curl.log(f"Saving to cache with key: {hashable_request}")
            if hashable_request not in self.tuple_cache.keys():
                self.tuple_cache[hashable_request] = result
            # Save in batches to avoid excessive memory use
            batch_count += 1
            if batch_count >= self.CACHE_SAVE_BATCH_SIZE:
                self._save_tuples()
                batch_count = 0  # Reset counter

        # Final save if anything remains in cache
        if len(self.tuple_cache) > 0:
            self._save_tuples()
        # Finally, save the requests.
        self._save_requests()

    def generate_additive_triple(self, size0, size1, op, device=None, *args, **kwargs):
        """Generate multiplicative triples of given sizes"""
        raise NotImplementedError(
            "TupleProvider generate_additive_triple not implemented."
        )

    def square(self, size, device=None):
        """Generate square double of given size"""
        raise NotImplementedError("TupleProvider square not implemented.")

    def generate_binary_triple(self, size0, size1, device=None):
        """Generate xor triples of given size"""
        raise NotImplementedError(
            "TupleProvider generate_binary_triple not implemented."
        )

    def wrap_rng(self, size, device=None):
        """Generate random shared tensor of given size and sharing of its wraps"""
        raise NotImplementedError("TupleProvider wrap_rng not implemented.")

    def B2A_rng(self, size, device=None):
        """Generate random bit tensor as arithmetic and binary shared tensors"""
        raise NotImplementedError("TupleProvider B2A_rng not implemented.")

    def generate_one_hot(self, tensor_size, lut_size, device=None):
        """Generate random values and their corresponding one hot vectors of
        given size."""
        raise NotImplementedError(
            "TupleProvider generate_one_hot not implemented."
        )

    def egk_trunc_pr_rng(self, size, l, m, device=None):
        """Generate random values for the [EGK+20] probabilistic truncation protocol."""
        raise NotImplementedError(
            "TupleProvider egk_trunc_pr_rng not implemented."
        )

    def generate_permutation(self, tensor_size, device=None):
        """Generate random permutation."""
        raise NotImplementedError(
            "TupleProvider generate_permutation not implemented."
        )
