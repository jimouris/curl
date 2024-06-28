#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import itertools
import logging
import math
import random
import unittest
from collections import defaultdict

import curl
import curl.communicator as comm
import torch
import torch.nn.functional as F
from curl.common import serial
from curl.common.tensor_types import is_float_tensor
from curl.config import cfg
from test.multiprocess_test_case import get_random_test_tensor, MultiProcessTestCase
from torch import nn


class TestCrypten(MultiProcessTestCase):
    """
    This class tests all member functions of crypten package
    """

    def setUp(self):
        super().setUp()
        if self.rank >= 0:
            curl.init()
            curl.set_default_cryptensor_type("mpc")

    def _check(self, encrypted_tensor, reference, msg, tolerance=None):
        if tolerance is None:
            tolerance = getattr(self, "default_tolerance", 0.05)
        tensor = encrypted_tensor.get_plain_text()

        # Check sizes match
        self.assertTrue(tensor.size() == reference.size(), msg)

        self.assertTrue(is_float_tensor(reference), "reference must be a float")
        diff = (tensor - reference).abs_()
        norm_diff = diff.div(tensor.abs() + reference.abs()).abs_()
        test_passed = norm_diff.le(tolerance) + diff.le(tolerance * 0.1)
        test_passed = test_passed.gt(0).all().item() == 1
        if not test_passed:
            logging.info(msg)
            logging.info("Result %s" % tensor)
            logging.info("Result = %s;\nreference = %s" % (tensor, reference))
        self.assertTrue(test_passed, msg=msg)

    def test_przs_generators(self):
        """Tests that przs generators are initialized independently"""
        # Check that each party has two unique generators for next and prev seeds
        for device in curl.generators["prev"].keys():
            t0 = torch.randint(
                -(2**63),
                2**63 - 1,
                (1,),
                device=device,
                generator=curl.generators["prev"][device],
            )
            t1 = torch.randint(
                -(2**63),
                2**63 - 1,
                (1,),
                device=device,
                generator=curl.generators["next"][device],
            )
            self.assertNotEqual(t0.item(), t1.item())

            # Check that generators are sync'd as expected
            for rank in range(self.world_size):
                receiver = rank
                sender = (rank + 1) % self.world_size
                if self.rank == receiver:
                    sender_value = comm.get().recv_obj(sender)
                    receiver_value = curl.generators["next"][device].initial_seed()
                    self.assertEqual(sender_value, receiver_value)
                elif self.rank == sender:
                    sender_value = curl.generators["prev"][device].initial_seed()
                    comm.get().send_obj(sender_value, receiver)

    def test_global_generator(self):
        """Tests that global generator is generated properly"""
        # Check that all seeds are the same
        for device in curl.generators["global"].keys():
            this_generator = curl.generators["global"][device].initial_seed()
            generator0 = comm.get().broadcast_obj(this_generator, 0)
            self.assertEqual(this_generator, generator0)

    def test_manual_seeds(self):
        """
        Tests that user-supplied seeds replaces auto-generated seeds
        and tests that the seed values match the expected values
        """

        # Store auto-generated seeds
        orig_seeds = defaultdict(dict)
        seed_names = ["prev", "next", "local", "global"]
        for seed_name in seed_names:
            for device in curl.generators[seed_name].keys():
                orig_seeds[seed_name][device] = curl.generators[seed_name][
                    device
                ].initial_seed()

        # User-generated seeds
        next_seed = random.randint(0, 2**63 - 1)
        local_seed = random.randint(0, 2**63 - 1)
        global_seed = random.randint(0, 2**63 - 1)

        # Store expected seeds
        expected_seeds = {}
        expected_seeds["next"] = next_seed
        expected_seeds["local"] = local_seed

        # Set user-generated seeds in crypten
        cfg.debug.debug_mode = True
        curl.manual_seed(next_seed, local_seed, global_seed)

        # Check that user-generated seeds are not equal to the auto-generated ones
        for seed_name in seed_names:
            for device in curl.generators[seed_name].keys():
                self.assertNotEqual(
                    curl.generators[seed_name][device].initial_seed(),
                    orig_seeds[seed_name][device],
                )

                # Check if seeds match the expected seeds
                if seed_name in expected_seeds.keys():
                    self.assertEqual(
                        curl.generators[seed_name][device].initial_seed(),
                        expected_seeds[seed_name],
                    )

        # Run the tests to validate prev and global are intialized correctly
        self.test_przs_generators()
        self.test_global_generator()

    def test_cat_stack(self):
        """Tests concatenation and stacking of tensors"""
        tensor1 = get_random_test_tensor(size=(5, 5, 5, 5), is_float=True)
        tensor2 = get_random_test_tensor(size=(5, 5, 5, 5), is_float=True)
        encrypted1 = curl.cryptensor(tensor1)
        encrypted2 = curl.cryptensor(tensor2)

        for module in [crypten, torch]:  # torch.cat on CrypTensor runs curl.cat
            for op in ["cat", "stack"]:
                reference = getattr(torch, op)([tensor1, tensor2])
                encrypted_out = getattr(module, op)([encrypted1, encrypted2])
                self._check(encrypted_out, reference, "%s failed" % op)

                for dim in range(4):
                    reference = getattr(torch, op)([tensor1, tensor2], dim=dim)
                    encrypted_out = getattr(module, op)(
                        [encrypted1, encrypted2], dim=dim
                    )
                    self._check(encrypted_out, reference, "%s failed" % op)

    def test_print_log(self):
        """Tests curl.print and curl.log logging functions."""
        curl.print("test")
        curl.log("test")

    def test_rand(self):
        """Tests uniform random variable generation on [0, 1)"""
        for size in [(10,), (10, 10), (10, 10, 10)]:
            randvec = curl.rand(*size)
            self.assertTrue(randvec.size() == size, "Incorrect size")
            tensor = randvec.get_plain_text()
            self.assertTrue(
                (tensor >= 0).all() and (tensor < 1).all(), "Invalid values"
            )

        randvec = curl.rand(int(1e6)).get_plain_text()
        mean = torch.mean(randvec)
        var = torch.var(randvec)
        self.assertTrue(torch.isclose(mean, torch.tensor([0.5]), rtol=1e-3, atol=1e-3))
        self.assertTrue(
            torch.isclose(var, torch.tensor([1.0 / 12]), rtol=1e-3, atol=1e-3)
        )

    def test_bernoulli(self):
        for size in [(10,), (10, 10), (10, 10, 10)]:
            probs = torch.rand(size)
            randvec = curl.bernoulli(probs)
            self.assertTrue(randvec.size() == size, "Incorrect size")
            tensor = randvec.get_plain_text()
            self.assertTrue(((tensor == 0) + (tensor == 1)).all(), "Invalid values")

        probs = torch.Tensor(int(1e4)).fill_(0.2)
        randvec = curl.bernoulli(probs).get_plain_text()
        frac_zero = float((randvec == 0).sum()) / randvec.nelement()
        self.assertTrue(math.isclose(frac_zero, 0.8, rel_tol=1e-1, abs_tol=1e-1))

    def test_cryptensor_registration(self):
        """Tests the registration mechanism for custom `CrypTensor` types."""

        # perform tests:
        cryptensor_name = "my_cryptensor"
        self.assertEqual(curl.get_default_cryptensor_type(), "mpc")
        with self.assertRaises(ValueError):
            curl.set_default_cryptensor_type(cryptensor_name)
        tensor = curl.cryptensor(torch.zeros(1, 3))
        self.assertEqual(curl.get_cryptensor_type(tensor), "mpc")

        # register new tensor type:
        @curl.register_cryptensor(cryptensor_name)
        class MyCrypTensor(curl.CrypTensor):
            """Dummy `CrypTensor` type."""

            def __init__(self, *args, **kwargs):
                self.is_custom_type = True

        # test that registration was successful:
        self.assertEqual(curl.get_default_cryptensor_type(), "mpc")
        curl.set_default_cryptensor_type(cryptensor_name)
        self.assertEqual(curl.get_default_cryptensor_type(), cryptensor_name)
        tensor = curl.cryptensor(torch.zeros(1, 3))
        self.assertTrue(getattr(tensor, "is_custom_type", False))
        self.assertEqual(curl.get_cryptensor_type(tensor), cryptensor_name)

    def test_cryptensor_instantiation(self):
        """Tests that CrypTensors cannot be instantiated."""
        tensor = get_random_test_tensor()
        with self.assertRaises(TypeError):
            encrypted_tensor = curl.CrypTensor(tensor)
        encrypted_tensor = curl.mpc.MPCTensor(tensor)
        self.assertIsInstance(encrypted_tensor, curl.CrypTensor)

    def test_save_load(self):
        """Test that curl.save and curl.load properly save and load
        shares of cryptensors"""
        import io
        import pickle

        def custom_load_function(f):
            obj = pickle.load(f)
            return obj

        def custom_save_function(obj, f):
            pickle.dump(obj, f)

        all_save_fns = [torch.save, custom_save_function]
        all_load_fns = [torch.load, custom_load_function]

        tensor = get_random_test_tensor()
        cryptensor1 = curl.cryptensor(tensor)

        for i, save_closure in enumerate(all_save_fns):
            load_closure = all_load_fns[i]
            f = [
                io.BytesIO() for i in range(curl.communicator.get().get_world_size())
            ]
            curl.save(cryptensor1, f[self.rank], save_closure=save_closure)
            f[self.rank].seek(0)
            cryptensor2 = curl.load(f[self.rank], load_closure=load_closure)
            # test whether share matches
            self.assertTrue(cryptensor1.share.allclose(cryptensor2.share))
            # test whether tensor matches
            self.assertTrue(
                cryptensor1.get_plain_text().allclose(cryptensor2.get_plain_text())
            )
            attributes = [
                a
                for a in dir(cryptensor1)
                if not a.startswith("__")
                and not callable(getattr(cryptensor1, a))
                and a not in ["share", "_tensor", "ctx"]
            ]
            for a in attributes:
                attr1, attr2 = getattr(cryptensor1, a), getattr(cryptensor2, a)
                if a == "encoder":
                    self.assertTrue(attr1._scale == attr2._scale)
                    self.assertTrue(attr1._precision_bits == attr2._precision_bits)
                elif torch.is_tensor(attr1):
                    self.assertTrue(attr1.eq(attr2).all())
                else:
                    self.assertTrue(attr1 == attr2)

    def test_plaintext_save_load_from_party(self):
        """Test that curl.save_from_party and curl.load_from_party
        properly save and load plaintext tensors"""
        import tempfile

        import numpy as np

        def custom_load_function(f):
            np_arr = np.load(f)
            tensor = torch.from_numpy(np_arr)
            return tensor

        def custom_save_function(obj, f):
            np_arr = obj.numpy()
            np.save(f, np_arr)

        comm = curl.communicator
        filename = tempfile.NamedTemporaryFile(delete=True).name
        all_save_fns = [torch.save, custom_save_function]
        all_load_fns = [torch.load, custom_load_function]
        all_file_completions = [".pth", ".npy"]
        all_test_load_fns = [torch.load, np.load]
        for dimensions in range(1, 5):
            # Create tensors with different sizes on each rank
            size = [self.rank + 1] * dimensions
            size = tuple(size)
            tensor = torch.randn(size=size)

            for i, save_closure in enumerate(all_save_fns):
                load_closure = all_load_fns[i]
                test_load_fn = all_test_load_fns[i]
                complete_file = filename + all_file_completions[i]
                for src in range(comm.get().get_world_size()):
                    curl.save_from_party(
                        tensor, complete_file, src=src, save_closure=save_closure
                    )

                    # the following line will throw an error if an object saved with
                    # torch.save is attempted to be loaded with np.load
                    if self.rank == src:
                        test_load_fn(complete_file)

                    encrypted_load = curl.load_from_party(
                        complete_file, src=src, load_closure=load_closure
                    )

                    reference_size = tuple([src + 1] * dimensions)
                    self.assertEqual(encrypted_load.size(), reference_size)

                    size_out = [src + 1] * dimensions
                    reference = (
                        tensor if self.rank == src else torch.empty(size=size_out)
                    )
                    comm.get().broadcast(reference, src)
                    self._check(encrypted_load, reference, "curl.load() failed")

                    # test for invalid load_closure
                    with self.assertRaises(TypeError):
                        curl.load_from_party(
                            complete_file, src=src, load_closure=(lambda f: None)
                        )

                    # test pre-loaded
                    encrypted_preloaded = curl.load_from_party(
                        src=src, preloaded=tensor
                    )
                    self._check(
                        encrypted_preloaded,
                        reference,
                        "curl.load() failed using preloaded",
                    )

    def test_plaintext_save_load_module_from_party(self) -> None:
        """Test that curl.save_from_party and curl.load_from_party
        properly save and load plaintext modules"""
        import tempfile

        comm = curl.communicator
        for model_type in [TestModule, NestedTestModule]:
            # Create models with different parameter values on each rank
            rank = comm.get().get_rank()

            test_model = model_type(200, 10)
            test_model.set_all_parameters(rank)
            serial.register_safe_class(model_type)

            filename = tempfile.NamedTemporaryFile(delete=True).name
            for src in range(comm.get().get_world_size()):
                curl.save_from_party(test_model, filename, src=src)

                result = curl.load_from_party(filename, src=src)
                if src == rank:
                    for param in result.parameters(recurse=True):
                        self.assertTrue(
                            param.eq(rank).all().item(), "Model load failed"
                        )
                self.assertEqual(result.src, src)

    def test_where(self) -> None:
        """Test that curl.where properly conditions"""
        sizes = [(10,), (5, 10), (1, 5, 10)]
        y_types = [lambda x: x, curl.cryptensor]

        for size, y_type in itertools.product(sizes, y_types):
            tensor1 = get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor1 = curl.cryptensor(tensor1)
            tensor2 = get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor2 = y_type(tensor2)

            condition_tensor = (
                get_random_test_tensor(max_value=1, size=size, is_float=False) + 1
            )
            condition_encrypted = curl.cryptensor(condition_tensor)
            condition_bool = condition_tensor.bool()

            reference_out = torch.where(condition_bool, tensor1, tensor2)

            encrypted_out = curl.where(
                condition_bool, encrypted_tensor1, encrypted_tensor2
            )

            y_is_private = curl.is_encrypted_tensor(tensor2)
            self._check(
                encrypted_out,
                reference_out,
                f"{'private' if y_is_private else 'public'} y "
                "where failed with public condition",
            )

            encrypted_out = encrypted_tensor1.where(
                condition_encrypted, encrypted_tensor2
            )
            self._check(
                encrypted_out,
                reference_out,
                f"{'private' if y_is_private else 'public'} y "
                "where failed with private condition",
            )

    @unittest.skip("Test is flaky, with successes, failures and timeouts as outcomes")
    def test_is_initialized(self) -> None:
        """Tests that the is_initialized flag is set properly"""
        comm = curl.communicator

        self.assertTrue(curl.is_initialized())
        self.assertTrue(comm.is_initialized())

        curl.uninit()
        self.assertFalse(curl.is_initialized())
        self.assertFalse(comm.is_initialized())

        # note that uninit() kills the TTP process, so we need to restart it:
        if self.rank == self.MAIN_PROCESS_RANK and curl.mpc.ttp_required():
            self.processes += [self._spawn_ttp()]

        curl.init()
        self.assertTrue(curl.is_initialized())
        self.assertTrue(comm.is_initialized())


# Modules used for testing saveing / loading of modules
class TestModule(nn.Module):
    def __init__(self, input_features, output_features):
        super(TestModule, self).__init__()
        self.fc1 = nn.Linear(input_features, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, output_features)

    def forward(self, input):
        out = F.relu(self.fc1(input))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def set_all_parameters(self, value):
        self.fc1.weight.data.fill_(value)
        self.fc1.bias.data.fill_(value)
        self.fc2.weight.data.fill_(value)
        self.fc2.bias.data.fill_(value)
        self.fc3.weight.data.fill_(value)
        self.fc3.bias.data.fill_(value)


class NestedTestModule(nn.Module):
    def __init__(self, input_features, output_features):
        super(NestedTestModule, self).__init__()
        self.fc1 = nn.Linear(input_features, input_features)
        self.nested = TestModule(input_features, output_features)

    def forward(self, input):
        out = F.relu(self.fc1(input))
        out = self.nested(out)

    def set_all_parameters(self, value):
        self.fc1.weight.data.fill_(value)
        self.fc1.bias.data.fill_(value)
        self.nested.set_all_parameters(value)


# This code only runs when executing the file outside the test harness
if __name__ == "__main__":
    unittest.main()
