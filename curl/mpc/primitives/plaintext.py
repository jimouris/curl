#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import curl
import curl.communicator as comm
import jax
import jax.numpy as jnp
import torch

from curl.common.util import count_wraps
from curl.config import cfg
from jax.lib import xla_bridge

from .util import IgnoreEncodings

jax.config.update("jax_enable_x64", True)
xla_bridge.get_backend().platform


def __plaintext_protocol(op, x, y, *args, **kwargs):
    """Performs plaintext protocol for additively secret-shared tensors x and y

    1. Open ([epsilon] = [x]) and ([delta] = [y])
    2. Return [z] = (epsilon * delta)
    """
    assert op in {
        "mul",
        "matmul",
        "conv1d",
        "conv2d",
        "conv_transpose1d",
        "conv_transpose2d",
    }
    if x.device != y.device:
        raise ValueError(f"x lives on device {x.device} but y on device {y.device}")

    from .arithmetic import ArithmeticSharedTensor

    if cfg.mpc.jax:
        device = jax.devices("cpu")[0]
        if x.device.type == "cuda":
            device = jax.devices("cuda")[x.device.index]

    epsilon, delta = ArithmeticSharedTensor.reveal_batch([x, y])
    if cfg.mpc.jax and op == "matmul":
        epsilon = jnp.array(epsilon.data, dtype=jnp.int64, device=device)
        delta = jnp.array(delta.data, dtype=jnp.int64, device=device)
        inner = jnp.matmul(epsilon, delta)
        inner = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(inner))
    elif cfg.mpc.jax and op == "mul":
        epsilon = jnp.array(epsilon.data, dtype=jnp.int64, device=device)
        delta = jnp.array(delta.data, dtype=jnp.int64, device=device)
        inner = jnp.multiply(epsilon, delta)
        inner = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(inner))
    else:
        inner = getattr(torch, op)(epsilon, delta, *args, **kwargs)
    z = ArithmeticSharedTensor(inner, precision=0, src=0)
    z.encoder._precision_bits = x.encoder.precision_bits + y.encoder.precision_bits
    return z


def mul(x, y):
    return __plaintext_protocol("mul", x, y)


def matmul(x, y):
    return __plaintext_protocol("matmul", x, y)


def conv1d(x, y, **kwargs):
    return __plaintext_protocol("conv1d", x, y, **kwargs)


def conv2d(x, y, **kwargs):
    return __plaintext_protocol("conv2d", x, y, **kwargs)


def conv_transpose1d(x, y, **kwargs):
    return __plaintext_protocol("conv_transpose1d", x, y, **kwargs)


def conv_transpose2d(x, y, **kwargs):
    return __plaintext_protocol("conv_transpose2d", x, y, **kwargs)


def square(x):
    """Computes the square of `x` for additively secret-shared tensor `x`

    1. Obtain uniformly random sharings [r] and [r2] = [r * r]
    2. Additively hide [x] with appropriately sized [r]
    3. Open ([epsilon] = [x] - [r])
    4. Return z = [r2] + 2 * epsilon * [r] + epsilon ** 2
    """
    from .arithmetic import ArithmeticSharedTensor

    epsilon = ArithmeticSharedTensor.reveal(x)
    c = ArithmeticSharedTensor(epsilon * epsilon, precision=0, src=0)
    c.encoder._precision_bits = 2 * x.encoder.precision_bits
    return c


def wraps(x):
    """Privately computes the number of wraparounds for a set a shares

    To do so, we note that:
        [theta_x] = theta_z + [beta_xr] - [theta_r] - [eta_xr]

    Where [theta_i] is the wraps for a variable i
          [beta_ij] is the differential wraps for variables i and j
          [eta_ij]  is the plaintext wraps for variables i and j

    Note: Since [eta_xr] = 0 with probability 1 - |x| / Q for modulus Q, we
    can make the assumption that [eta_xr] = 0 with high probability.
    """
    provider = curl.mpc.get_default_provider()
    r, theta_r = provider.wrap_rng(x.size(), device=x.device)
    beta_xr = theta_r.clone()
    beta_xr._tensor = count_wraps([x._tensor, r._tensor])

    with IgnoreEncodings([x, r]):
        z = x + r
    theta_z = comm.get().gather(z._tensor, 0)
    theta_x = beta_xr - theta_r

    # TODO: Incorporate eta_xr
    if x.rank == 0:
        theta_z = count_wraps(theta_z)
        theta_x._tensor += theta_z
    return theta_x


def truncate(x, y):
    """Protocol to divide an ArithmeticSharedTensor `x` by a constant integer `y`"""
    wrap_count = wraps(x)
    x.share = x.share.div_(y, rounding_mode="trunc")
    # NOTE: The multiplication here must be split into two parts
    # to avoid long out-of-bounds when y <= 2 since (2 ** 63) is
    # larger than the largest long integer.
    correction = wrap_count * 4 * (int(2**62) // y)
    x.share -= correction.share
    return x


def egk_trunc_pr(x, l, m):
    """
    Evaluates probabilistic truncation with no correctness error using [EGK+20]
    protocol.

    Reference: "Improved Primitives for MPC over Mixed Arithmetic-Binary Circuits"
    Figure: 10
    Link: https://eprint.iacr.org/2020/338.pdf

    Args:
        x (torch.Tensor): Input tensor.
        l (int): Max bit size of input tensor, i.e., 0 <= x < 2**l.
        m (int): number of bits to truncate.

    Returns:
        torch.Tensor: Result tensor after applying the LUT.
    """

    provider = curl.mpc.get_default_provider()
    k = 64
    two_to_l = torch.tensor(2**l, dtype=torch.int64, device=x.device) # to prevent overflow
    tensor_size = x.size()

    # Preprocessing
    r, r_p, b = provider.egk_trunc_pr_rng(tensor_size, l, m, device=x.device)
    with IgnoreEncodings([x, b]):
        # Step 1
        a_p = x + 2**(l-1) # allowing negative numbers
        rpp = 2**m * r + r_p
        enc_c = 2**(k - l - 1) * (a_p + two_to_l * b + rpp)
        c = enc_c.reveal()
        c_p = c >> (k - l - 1)
        # Step 2
        c_pl = (c_p >> l) & 1 # c'_l, the l-th (last) bit of c'
        v = b + c_pl - 2 * b * c_pl
        # Step 3
        y = 2**(l-m) * v - r - 2**(l-m-1) + ((c_p % two_to_l) // 2**m)

    return y


def evaluate_lut(x, lut):
    """Evaluates a Look-Up Table (LUT) using an input tensor x.

    Args:
        x (Cryptensor): Input tensor.
        lut (torch.Tensor): Look-Up Table tensor.

    Returns:
        Cryptensor: Result tensor after applying the LUT.
    """
    from .arithmetic import ArithmeticSharedTensor

    result = x.reveal() % lut.size()[0]
    result = ArithmeticSharedTensor(lut[result], precision=0, src=0)
    result.encoder._precision_bits = x.encoder.precision_bits
    return result


def evaluate_bior_lut(x, luts, scale, bias):
    """Evaluates a Look-Up Table (LUT) using an input tensor x.

    Args:
        x (Cryptensor): Input tensor.
        luts (torch.Tensor): Look-Up Table tensors.
        scale (torch.Tensor): Scaling factor for the lookups.
        bias (int): Bias for the LUT.

    Returns:
        Cryptensor: Result tensor after applying the LUT.
    """
    from .arithmetic import ArithmeticSharedTensor

    result = x.reveal() % luts[0].shape[0]
    lut0 = luts[0][result]
    lut1 = luts[1][result]
    with IgnoreEncodings([scale]):
        result = (lut1 - lut0) * scale + 2**bias * lut0
    result = ArithmeticSharedTensor(result, precision=0, src=0)
    result.encoder._precision_bits = x.encoder.precision_bits
    return result


def evaluate_embed(x, embed):
    """Evaluates an embedding using an input tensor x.

    Args:
        x (torch.Tensor): Input tensor.
        embed (Cryptensor): Embedding tensor.

    Returns:
        Cryptensor: Result tensor after applying the LUT.
    """
    from .arithmetic import ArithmeticSharedTensor

    embed = ArithmeticSharedTensor.from_shares(embed, precision=0, device=x.device)
    embed = embed.reveal()
    result = x.reveal() % embed.shape[0]
    x.share = embed[result]
    return x


def shuffle(x):
    provider = curl.mpc.get_default_provider()
    permutation, inv_permutation = provider.generate_permutation(x.size(0), device=x.device)
    result = x[permutation]
    return result, inv_permutation


def unshuffle(x, inv_permutation):
    return x[inv_permutation]
