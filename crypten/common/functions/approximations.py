#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
import pywt

import crypten
import torch
from crypten.config import cfg


__all__ = [
    "exp",
    "log",
    "reciprocal",
    "inv_sqrt",
    "sqrt",
    "_eix",
    "cossin",
    "cos",
    "sin",
    "sigmoid",
    "tanh",
    "erf",
    "gelu",
    "silu",
    "softmax",
    "log_softmax",
]

class LookupTables:

    LUTs = {}

    """Use to create a singleton"""
    def __new__(cls, *args, **kwds):
        """
        >>> s = Singleton()
        >>> p = Singleton()
        >>> id(s) == id(p)
        True
        """
        it_id = "__it__"
        # getattr will dip into base classes, so __dict__ must be used
        it = cls.__dict__.get(it_id, None)
        if it is not None:
            return it
        it = object.__new__(cls)
        setattr(cls, it_id, it)
        it.init(*args, **kwds)
        it.initialize_luts()
        return it

    def init(self, *args, **kwds):
        pass

    @classmethod
    def generate_haar(cls, max_bits, lut_bits, function, name):
        scale = 2**cfg.encoder.precision_bits
        max_element = 2**max_bits
        depth = max_bits + cfg.encoder.precision_bits - lut_bits
        full = function(np.linspace(1.0/scale, max_element, max_element * scale))
        coeffs, *_ = pywt.wavedec(full, 'haar', level=depth)
        cls.LUTs[name] = torch.tensor(coeffs * 2**(-depth/2) * scale).long()

    @classmethod
    def generate_bior(cls, max_bits, lut_bits, function, name):
        scale = 2**cfg.encoder.precision_bits
        max_element = 2**max_bits
        depth = max_bits + cfg.encoder.precision_bits - lut_bits
        full = function(np.linspace(1.0/scale, max_element, max_element * scale))
        coeffs, *_ = pywt.wavedec(full, 'bior2.2', level=depth)
        coeffs = np.stack([np.roll(coeffs, -2)[:2**lut_bits], np.roll(coeffs, -3)[:2**lut_bits]])
        cls.LUTs[name] = torch.tensor((coeffs * scale) * 2**(depth*0.5)).long()

    @classmethod
    def initialize_luts(cls):
        r"""Initialize LUTs for different approximation functions:
            * exp: Exponential
            * log: Logarithm
            * reciprocal: Reciprocal
            * sqrt: Square root
            * inv_sqrt: Inverse square root
            * sin: Sine
            * cos: Cosine
            * sigmoid: Sigmoid
            * tanh: hyperbolic tangent function
            * erf: Error function
            * gelu: Gaussian Error Linear Units
            * silu: Sigmoid Linear Units
        """
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        relu = lambda x: x * (x > 0)

        """Exp LUT"""
        if cfg.functions.exp_method in ("haar", "bior"):
            scale = 2**cfg.encoder.precision_bits
            max_element = 2**cfg.functions.exp_lut_max_bits
            # HAAR
            depth = 1 + cfg.functions.exp_lut_max_bits + cfg.encoder.precision_bits - cfg.functions.exp_haar_size_bits
            full = np.exp(np.linspace(-max_element, max_element-1.0/scale, 2 * max_element * scale))
            coeffs, *_ = pywt.wavedec(full, 'haar', level=depth)
            cls.LUTs["exp_haar"] = torch.tensor(coeffs * 2**(-depth/2) * scale).long()
            # BIOR
            depth = 1 + cfg.functions.exp_lut_max_bits + cfg.encoder.precision_bits - cfg.functions.exp_bior_size_bits
            coeffs, *_ = pywt.wavedec(full, 'bior2.2', level=depth)
            coeffs = coeffs[:2**cfg.functions.exp_bior_size_bits]
            coeffs = np.stack([np.roll(coeffs, -2), np.roll(coeffs, -3)])
            cls.LUTs["exp_bior"] = torch.tensor(coeffs * scale).long()
            # NEXP
            size = cfg.functions.exp_neg_lut_size
            full = np.exp(-np.linspace(1.0/size, 1/2**4, size))
            cls.LUTs['nexp_low'] = torch.tensor(full * scale).long()
            full = np.exp(-np.linspace(1.0*2**4/size, 2**4, size))
            cls.LUTs['nexp_high'] = torch.tensor(full * scale).long()
            # NEXP-BIOR
            cls.generate_haar(cfg.functions.exp_lut_max_bits,
                              cfg.functions.exp_haar_size_bits,
                              lambda x: np.exp(-x),
                              "nexp_haar")
            # NEXP-BIOR
            cls.generate_bior(cfg.functions.exp_lut_max_bits,
                              cfg.functions.exp_bior_size_bits,
                              lambda x: np.exp(-x),
                              "nexp_bior")

        """Logarithm LUT"""
        if cfg.functions.log_method in ("haar", "bior"):
            cls.generate_haar(cfg.functions.log_lut_max_bits,
                              cfg.functions.log_haar_size_bits,
                              np.log,
                              "log_haar")
            cls.generate_bior(cfg.functions.log_lut_max_bits,
                              cfg.functions.log_bior_size_bits,
                              np.log,
                              "log_bior")

        """Reciprocal LUT"""
        if cfg.functions.reciprocal_method in ("haar", "bior"):
            cls.generate_haar(cfg.functions.reciprocal_lut_max_bits,
                              cfg.functions.reciprocal_haar_size_bits,
                              np.reciprocal,
                              "reciprocal_haar")
            cls.generate_bior(cfg.functions.reciprocal_lut_max_bits,
                              cfg.functions.reciprocal_bior_size_bits,
                              np.reciprocal,
                              "reciprocal_bior")

        """Sqrt LUT"""
        if cfg.functions.sqrt_method in ("haar", "bior"):
            cls.generate_haar(cfg.functions.sqrt_lut_max_bits,
                              cfg.functions.sqrt_haar_size_bits,
                              np.sqrt,
                              "sqrt_haar")
            cls.generate_bior(cfg.functions.sqrt_lut_max_bits,
                              cfg.functions.sqrt_bior_size_bits,
                              np.sqrt,
                              "sqrt_bior")

        """Inv Sqrt LUT"""
        if cfg.functions.inv_sqrt_method in ("haar", "bior"):
            cls.generate_haar(cfg.functions.inv_sqrt_lut_max_bits,
                              cfg.functions.inv_sqrt_haar_size_bits,
                              lambda x: np.reciprocal(np.sqrt(x)),
                              "inv_sqrt_haar")
            cls.generate_bior(cfg.functions.inv_sqrt_lut_max_bits,
                              cfg.functions.inv_sqrt_bior_size_bits,
                              lambda x: np.reciprocal(np.sqrt(x)),
                              "inv_sqrt_bior")

        """Trigonometry LUTs: Sin, Cos"""
        if cfg.functions.trigonometry_method in ("haar", "bior"):
            cls.generate_haar(3,
                              cfg.functions.trigonometry_haar_size_bits,
                              lambda x: np.sin(x/np.pi/2),
                              "sin_haar")
            cls.generate_bior(3,
                              cfg.functions.trigonometry_bior_size_bits,
                              lambda x: np.sin(x/np.pi/2),
                              "sin_bior")
            cls.generate_haar(3,
                              cfg.functions.trigonometry_haar_size_bits,
                              lambda x: np.cos(x/np.pi/2),
                              "cos_haar")
            cls.generate_bior(3,
                              cfg.functions.trigonometry_bior_size_bits,
                              lambda x: np.cos(x/np.pi/2),
                              "cos_bior")

        """Sigmoid & Tanh LUT"""
        if cfg.functions.sigmoid_tanh_method in ("haar", "bior"):
            cls.generate_haar(cfg.functions.sigmoid_lut_max_bits,
                              cfg.functions.sigmoid_tanh_haar_size_bits,
                              sigmoid,
                              "sigmoid_haar")
            cls.generate_bior(cfg.functions.sigmoid_lut_max_bits,
                              cfg.functions.sigmoid_tanh_bior_size_bits,
                              sigmoid,
                              "sigmoid_bior")
            cls.generate_haar(cfg.functions.tanh_lut_max_bits,
                              cfg.functions.sigmoid_tanh_haar_size_bits,
                              np.tanh,
                              "tanh_haar")
            cls.generate_bior(cfg.functions.tanh_lut_max_bits,
                              cfg.functions.sigmoid_tanh_bior_size_bits,
                              np.tanh,
                              "tanh_bior")

        """Erf LUT"""
        if cfg.functions.erf_method in ("haar", "bior"):
            cls.generate_haar(cfg.functions.erf_lut_max_bits,
                              cfg.functions.erf_haar_size_bits,
                              lambda x: np.array([math.erf(x_) for x_ in x]),
                              "erf_haar")
            cls.generate_bior(cfg.functions.erf_lut_max_bits,
                              cfg.functions.erf_bior_size_bits,
                              lambda x: np.array([math.erf(x_) for x_ in x]),
                              "erf_bior")

        """Gelu LUT"""
        if cfg.functions.gelu_method in ("haar", "bior"):
            gelu = lambda x: x * (1 + np.array([math.erf(x_/math.sqrt(2)) for x_ in x])) / 2
            cls.generate_haar(cfg.functions.gelu_lut_max_bits,
                              cfg.functions.gelu_haar_size_bits,
                              lambda x: relu(x) - gelu(x),
                              "gelu_haar")
            cls.generate_bior(cfg.functions.gelu_lut_max_bits,
                              cfg.functions.gelu_bior_size_bits,
                              lambda x: relu(x) - gelu(x),
                              "gelu_bior")

        """Silu LUT"""
        if cfg.functions.silu_method in ("haar", "bior"):
            silu = lambda x: x * sigmoid(x)
            cls.generate_haar(cfg.functions.silu_lut_max_bits,
                              cfg.functions.silu_haar_size_bits,
                              lambda x: relu(x) - silu(x),
                              "silu_haar")
            cls.generate_bior(cfg.functions.silu_lut_max_bits,
                              cfg.functions.silu_bior_size_bits,
                              lambda x: relu(x) - silu(x),
                              "silu_bior")

def _nexp_lut(self, method):
    r"""Approximates the negative exponential function using a limit approximation"""
    luts = LookupTables()
    precision = 2**cfg.encoder.precision_bits
    size = cfg.functions.exp_neg_lut_size

    if method == "split":
        x = self.div(precision/2**4/size)
        d = x < 1
        bits = x.encoder._precision_bits
        x.encoder._precision_bits = 0
        c = d * x + (1-d) * (precision-1)
        x.encoder._precision_bits = bits
        c0 = c # c0 = c.mod(size)
        c1 = c.div(size)
        t0 = c0.evaluate_lut(luts.LUTs["nexp_low"])
        t1 = c1.evaluate_lut(luts.LUTs["nexp_high"])
        return t0 * t1
    elif method == "haar":
        check = self < 2**cfg.functions.exp_lut_max_bits
        truncation = cfg.functions.exp_lut_max_bits + cfg.encoder.precision_bits - cfg.functions.exp_bior_size_bits
        msb = self.div(2**truncation)
        lut = msb.evaluate_lut(luts.LUTs["nexp_haar"])
        return check * lut
    elif method == "bior":
        check = self < 2**cfg.functions.exp_lut_max_bits
        truncation = cfg.functions.exp_lut_max_bits + cfg.encoder.precision_bits - cfg.functions.exp_bior_size_bits
        msb, lsb = self.divmod(2**truncation)
        lut = msb.evaluate_bior_lut(luts.LUTs["nexp_bior"], lsb, truncation)
        return check * lut
    else:
        raise ValueError(f"Invalid method {method} given for nexp function")

# Iterative methods:
def exp(self):
    r"""Approximates the exponential function using a limit approximation:

    .. math::

        exp(x) = \lim_{n \\rightarrow \\infty} (1 + x / n) ^ n

    Here we compute exp by choosing n = 2 ** d for some large d equal to
    `iterations`. We then compute (1 + x / n) once and square `d` times.

    Set the number of iterations for the limit approximation with
    config.exp_iterations.
    """  # noqa: W605
    method = cfg.functions.exp_method

    if method in ("split", "haar", "bior"):
        if cfg.functions.exp_all_neg:
            return _nexp_lut(-self, method)
        luts = LookupTables()
        if method == "haar":
            truncation = cfg.functions.exp_lut_max_bits + cfg.encoder.precision_bits - cfg.functions.exp_haar_size_bits
            msb = self.div(2**truncation)
            return msb.evaluate_lut(luts.LUTs["exp_haar"])
        elif method == "bior":
            truncation = cfg.functions.exp_lut_max_bits + cfg.encoder.precision_bits - cfg.functions.exp_bior_size_bits
            msb, lsb = self.divmod(2**truncation)
            return msb.evaluate_bior_lut(luts.LUTs["exp_bior"], lsb, truncation)
    elif method == "limit":
        iters = cfg.functions.exp_iterations
        result = 1 + self.div(2**iters)
        for _ in range(iters):
            result = result.square()
        return result
    else:
        raise ValueError(f"Invalid method {method} given for exp function")


def log(self, input_in_01=False, use_lut=False):
    r"""
    Approximates the natural logarithm using 8th order modified
    Householder iterations. This approximation is accurate within 2% relative
    error on [0.0001, 250].

    Iterations are computed by: :math:`h = 1 - x * exp(-y_n)`

    .. math::

        y_{n+1} = y_n - \sum_k^{order}\frac{h^k}{k}

    Args:
        input_in_01 (bool) : Allows a user to indicate that the input is in the domain [0, 1],
            causing the function optimize for this domain. This is useful for computing
            log-probabilities for entropy functions.

            We shift the domain of convergence by a constant :math:`a` using the following identity:

            .. math::

                \ln{u} = \ln {au} - \ln{a}

            Since the domain of convergence for CrypTen's log() function is approximately [1e-4, 1e2],
            we can set :math:`a=100`.

    Configuration parameters:
        iterations (int): number of Householder iterations for the approximation
        exp_iterations (int): number of iterations for limit approximation of exp
        order (int): number of polynomial terms used (order of Householder approx)
    """
    if input_in_01:
        return log(self.mul(100)) - 4.605170

    # Initialization to a decent estimate (found by qualitative inspection):
    #                ln(x) = x/120 - 20exp(-2x - 1.0) + 3.0
    iterations = cfg.functions.log_iterations
    exp_iterations = cfg.functions.log_exp_iterations
    order = cfg.functions.log_order
    method = cfg.functions.log_method

    if method in ("haar", "bior"):
        luts = LookupTables()
        if method == "haar":
            log_truncation = cfg.functions.log_lut_max_bits + cfg.encoder.precision_bits - cfg.functions.log_haar_size_bits
            msb = self.div(2**log_truncation)
            return msb.evaluate_lut(luts.LUTs["log_haar"])
        elif method == "bior":
            log_truncation = cfg.functions.log_lut_max_bits + cfg.encoder.precision_bits - cfg.functions.log_bior_size_bits
            msb, lsb = self.divmod(2**log_truncation)
            return msb.evaluate_bior_lut(luts.LUTs["log_bior"], lsb, log_truncation)
    elif method == "iter":
        term1 = self.div(120)
        term2 = exp(self.mul(2).add(1.0).neg()).mul(20)
        y = term1 - term2 + 3.0

        # 8th order Householder iterations
        with cfg.temp_override({"functions.exp_iterations": exp_iterations}):
            for _ in range(iterations):
                h = 1 - self * exp(-y)
                y -= h.polynomial([1 / (i + 1) for i in range(order)])
        return y
    else:
        raise ValueError(f"Invalid method {method} given for log function")

def reciprocal(self, input_in_01=False):
    r"""
    Args:
        input_in_01 (bool) : Allows a user to indicate that the input is in the range [0, 1],
                    causing the function optimize for this range. This is useful for improving
                    the accuracy of functions on probabilities (e.g. entropy functions).

    Methods:
        'NR' : `Newton-Raphson`_ method computes the reciprocal using iterations
                of :math:`x_{i+1} = (2x_i - self * x_i^2)` and uses
                :math:`3*exp(1 - 2x) + 0.003` as an initial guess by default

        'log' : Computes the reciprocal of the input from the observation that:
                :math:`x^{-1} = exp(-log(x))`

    Configuration params:
        reciprocal_method (str):  One of 'NR' or 'log' or 'lut'.
        reciprocal_nr_iters (int):  determines the number of Newton-Raphson iterations to run
                        for the `NR` method
        reciprocal_log_iters (int): determines the number of Householder
            iterations to run when computing logarithms for the `log` method
        reciprocal_all_pos (bool): determines whether all elements of the
            input are known to be positive, which optimizes the step of
            computing the sign of the input.
        reciprocal_initial (tensor): sets the initial value for the
            Newton-Raphson method. By default, this will be set to :math:
            `3*exp(-(x-.5)) + 0.003` as this allows the method to converge over
            a fairly large domain

    .. _Newton-Raphson:
        https://en.wikipedia.org/wiki/Newton%27s_method
    """
    pos_override = {"functions.reciprocal_all_pos": True}
    if input_in_01:
        with cfg.temp_override(pos_override):
            rec = reciprocal(self.mul(64)).mul(64)
        return rec

    # Get config options
    method = cfg.functions.reciprocal_method
    all_pos = cfg.functions.reciprocal_all_pos
    initial = cfg.functions.reciprocal_initial

    if not all_pos:
        sgn = self.sign()
        pos = sgn * self
        with cfg.temp_override(pos_override):
            return sgn * reciprocal(pos)

    if method in ("haar", "bior"):
        luts = LookupTables()
        if method == "haar":
            reciprocal_truncation = cfg.functions.reciprocal_lut_max_bits + cfg.encoder.precision_bits - cfg.functions.reciprocal_haar_size_bits
            msb = self.div(2**reciprocal_truncation)
            return msb.evaluate_lut(luts.LUTs["reciprocal_haar"])
        elif method == "bior":
            reciprocal_truncation = cfg.functions.reciprocal_lut_max_bits + cfg.encoder.precision_bits - cfg.functions.reciprocal_bior_size_bits
            msb, lsb = self.divmod(2**reciprocal_truncation)
            return msb.evaluate_bior_lut(luts.LUTs["reciprocal_bior"], lsb, reciprocal_truncation)
    elif method == "NR":
        nr_iters = cfg.functions.reciprocal_nr_iters
        if initial is None:
            # Initialization to a decent estimate (found by qualitative inspection):
            #                1/x = 3exp(1 - 2x) + 0.003
            result = 3 * (1 - 2 * self).exp() + 0.003
        else:
            result = initial
        for _ in range(nr_iters):
            if hasattr(result, "square"):
                result += result - result.square().mul_(self)
            else:
                result = 2 * result - result * result * self
        return result
    elif method == "log":
        log_iters = cfg.functions.reciprocal_log_iters
        with cfg.temp_override({"functions.log_iters": log_iters}):
            return exp(-log(self))
    else:
        raise ValueError(f"Invalid method {method} given for reciprocal function")


def inv_sqrt(self):
    r"""
    Computes the inverse square root of the input using the Newton-Raphson method.

    Configuration params:
        sqrt_nr_iters (int):  determines the number of Newton-Raphson iterations to run.
        sqrt_nr_initial (tensor): sets the initial value for the Newton-Raphson iterations.
                    By default, this will be set to allow the method to converge over a
                    fairly large domain.

    .. _Newton-Raphson:
        https://en.wikipedia.org/wiki/Fast_inverse_square_root#Newton's_method
    """
    initial = cfg.functions.sqrt_nr_initial
    iters = cfg.functions.sqrt_nr_iters
    method = cfg.functions.inv_sqrt_method

    if method in ("haar", "bior"):
        luts = LookupTables()
        if method == "haar":
            truncation = cfg.functions.inv_sqrt_lut_max_bits + cfg.encoder.precision_bits - cfg.functions.inv_sqrt_haar_size_bits
            msb = self.div(2**truncation)
            return msb.evaluate_lut(luts.LUTs["inv_sqrt_haar"])
        elif method == "bior":
            truncation = cfg.functions.inv_sqrt_lut_max_bits + cfg.encoder.precision_bits - cfg.functions.inv_sqrt_bior_size_bits
            msb, lsb = self.divmod(2**truncation)
            return msb.evaluate_bior_lut(luts.LUTs["inv_sqrt_bior"], lsb, truncation)
    elif method == "NR":
        # Initialize using decent approximation
        if initial is None:
            y = exp(self.div(2).add(0.2).neg()).mul(2.2).add(0.2)
            y -= self.div(1024)
        else:
            y = initial

        # Newton Raphson iterations for inverse square root
        for _ in range(iters):
            y = y.mul_(3 - self * y.square()).div_(2)
        return y
    else:
        raise ValueError(f"Invalid method {method} given for inv_sqrt function")

def sqrt(self):
    r"""
    Computes the square root of the input by computing its inverse square root using
    the Newton-Raphson method and multiplying by the input.

    Configuration params:
        sqrt_nr_iters (int):  determines the number of Newton-Raphson iterations to run
        sqrt_initial (tensor): sets the initial value for the inverse square root
            Newton-Raphson iterations. By default, this will be set to allow convergence
            over a fairly large domain.

    .. _Newton-Raphson:
        https://en.wikipedia.org/wiki/Fast_inverse_square_root#Newton's_method
    """
    method = cfg.functions.sqrt_method

    if method in ("haar", "bior"):
        luts = LookupTables()
        if method == "haar":
            truncation = cfg.functions.sqrt_lut_max_bits + cfg.encoder.precision_bits - cfg.functions.sqrt_haar_size_bits
            msb = self.div(2**truncation)
            return msb.evaluate_lut(luts.LUTs["sqrt_haar"])
        elif method == "bior":
            truncation = cfg.functions.sqrt_lut_max_bits + cfg.encoder.precision_bits - cfg.functions.sqrt_bior_size_bits
            msb, lsb = self.divmod(2**truncation)
            return msb.evaluate_bior_lut(luts.LUTs["sqrt_bior"], lsb, truncation)
    elif method == "NR":
        return inv_sqrt(self).mul_(self)
    else:
        raise ValueError(f"Invalid method {method} given for sqrt function")


def _eix(self):
    r"""Computes e^(i * self) where i is the imaginary unit.
    Returns (Re{e^(i * self)}, Im{e^(i * self)} = cos(self), sin(self)
    """
    iterations = cfg.functions.trig_iterations

    re = 1
    im = self.div(2**iterations)

    # First iteration uses knowledge that `re` is public and = 1
    re -= im.square()
    im *= 2

    # Compute (a + bi)^2 -> (a^2 - b^2) + (2ab)i `iterations` times
    for _ in range(iterations - 1):
        a2 = re.square()
        b2 = im.square()
        im = im.mul_(re)
        im._tensor *= 2
        re = a2 - b2

    return re, im


def cossin(self):
    r"""Computes cosine and sine of input via exp(i * x).

    Args:
        iterations (int): for approximating exp(i * x)
    """
    method = cfg.functions.trigonometry_method
    if method in ("haar", "bior"):
        luts = LookupTables()
        sgn = self.sign()
        pos = sgn * self
        tau = int(np.floor(2 * np.pi))
        self = self.div(tau)
        mod = pos.mod(2**cfg.encoder.precision_bits)
        if method == "haar":
            trig_truncation = 3 + cfg.encoder.precision_bits - cfg.functions.trigonometry_haar_size_bits
            msb = mod.div(2**trig_truncation)
            cos = msb.evaluate_lut(luts.LUTs["cos_haar"])
            sin = msb.evaluate_lut(luts.LUTs["sin_haar"])
        elif method == "bior":
            trig_truncation = 3 + cfg.encoder.precision_bits - cfg.functions.trigonometry_bior_size_bits
            msb, lsb = mod.divmod(2**trig_truncation)
            cos = msb.evaluate_bior_lut(luts.LUTs["cos_bior"], lsb, trig_truncation)
            sin = msb.evaluate_bior_lut(luts.LUTs["sin_bior"], lsb, trig_truncation)
        sin = sgn * sin
        return cos, sin
    elif method == "NR":
        return self._eix()
    else:
        raise ValueError(f"Invalid method {method} given for cossin function")


def cos(self):
    r"""Computes the cosine of the input using cos(x) = Re{exp(i * x)}

    Args:
        iterations (int): for approximating exp(i * x)
    """
    return cossin(self)[0]


def sin(self):
    r"""Computes the sine of the input using sin(x) = Im{exp(i * x)}

    Args:
        iterations (int): for approximating exp(i * x)
    """
    return cossin(self)[1]


# Logistic Functions
def sigmoid(self):
    r"""Computes the sigmoid function using the following definition

    .. math::
        \sigma(x) = (1 + e^{-x})^{-1}

    If a valid method is given, this function will compute sigmoid
        using that method:

    "chebyshev" - computes tanh via Chebyshev approximation with
        truncation and uses the identity:

    .. math::
        \sigma(x) = \frac{1}{2}tanh(\frac{x}{2}) + \frac{1}{2}

    "reciprocal" - computes sigmoid using :math:`1 + e^{-x}` and computing
        the reciprocal

    """  # noqa: W605
    method = cfg.functions.sigmoid_tanh_method

    if method in ("haar", "bior"):
        luts = LookupTables()
        ltz = self._ltz()
        sgn = 1 - 2 * ltz
        abs = sgn * self
        if method == "haar":
            st_truncation = cfg.functions.sigmoid_lut_max_bits + cfg.encoder.precision_bits - cfg.functions.sigmoid_tanh_haar_size_bits
            msb = abs.div(2**st_truncation)
            lut = msb.evaluate_lut(luts.LUTs["sigmoid_haar"])
        elif method == "bior":
            st_truncation = cfg.functions.sigmoid_lut_max_bits + cfg.encoder.precision_bits - cfg.functions.sigmoid_tanh_bior_size_bits
            msb, lsb = abs.divmod(2**st_truncation)
            lut = msb.evaluate_bior_lut(luts.LUTs["sigmoid_bior"], lsb, st_truncation)
        eval = ltz + sgn * lut
        limit = 1 - ltz
        check = abs < 2**cfg.functions.sigmoid_lut_max_bits
        return limit + check * (eval - limit)
    elif method == "chebyshev":
        tanh_approx = tanh(self.div(2))
        return tanh_approx.div(2) + 0.5
    elif method == "reciprocal":
        ltz = self._ltz()
        sgn = 1 - 2 * ltz

        pos_input = self.mul(sgn)
        denominator = pos_input.neg().exp().add(1)

        # TODO: Set these with configurable parameters
        with cfg.temp_override(
            {
                "functions.exp_iterations": 9,
                "functions.reciprocal_nr_iters": 3,
                "functions.reciprocal_all_pos": True,
                "functions.reciprocal_initial": 0.75,
            }
        ):
            pos_output = denominator.reciprocal()

        result = pos_output.where(1 - ltz, 1 - pos_output)
        # TODO: Support addition with different encoder scales
        # result = pos_output + ltz - 2 * pos_output * ltz
        return result
    else:
        raise ValueError(f"Unrecognized method {method} for sigmoid")


def tanh(self):
    r"""Computes the hyperbolic tangent function using the identity

    .. math::
        tanh(x) = 2\sigma(2x) - 1

    If a valid method is given, this function will compute tanh using that method:

    "chebyshev" - computes tanh via Chebyshev approximation with truncation.

    .. math::
        tanh(x) = \sum_{j=1}^terms c_{2j - 1} P_{2j - 1} (x / maxval)

    where c_i is the ith Chebyshev series coefficient and P_i is ith polynomial.
    The approximation is truncated to +/-1 outside [-1, 1].

    Args:
        terms (int): highest degree of Chebyshev polynomials.
                        Must be even and at least 6.
    """
    method = cfg.functions.sigmoid_tanh_method

    if method in ("haar", "bior"):
        luts = LookupTables()
        sgn = self.sign()
        abs = sgn * self
        if method == "haar":
            st_truncation = cfg.functions.tanh_lut_max_bits + cfg.encoder.precision_bits - cfg.functions.sigmoid_tanh_haar_size_bits
            msb = abs.div(2**st_truncation)
            lut = msb.evaluate_lut(luts.LUTs["tanh_haar"])
        elif method == "bior":
            st_truncation = cfg.functions.tanh_lut_max_bits + cfg.encoder.precision_bits - cfg.functions.sigmoid_tanh_bior_size_bits
            msb, lsb = abs.divmod(2**st_truncation)
            lut = msb.evaluate_bior_lut(luts.LUTs["tanh_bior"], lsb, st_truncation)
        check = abs < 2**cfg.functions.tanh_lut_max_bits
        return sgn * (1-check + lut * check)
    elif method == "reciprocal":
        return self.mul(2).sigmoid().mul(2).sub(1)
    elif method == "chebyshev":
        terms = cfg.functions.sigmoid_tanh_terms
        coeffs = crypten.common.util.chebyshev_series(torch.tanh, 1, terms)[1::2]
        tanh_polys = _chebyshev_polynomials(self, terms)
        tanh_polys_flipped = (
            tanh_polys.unsqueeze(dim=-1).transpose(0, -1).squeeze(dim=0)
        )
        out = tanh_polys_flipped.matmul(coeffs)

        # truncate outside [-maxval, maxval]
        return out.hardtanh()
    else:
        raise ValueError(f"Unrecognized method {method} for tanh")


def _chebyshev_polynomials(self, terms):
    r"""Evaluates odd degree Chebyshev polynomials at x

    Chebyshev Polynomials of the first kind are defined as

    .. math::
        P_0(x) = 1, P_1(x) = x, P_n(x) = 2 P_{n - 1}(x) - P_{n-2}(x)

    Args:
        self (MPCTensor): input at which polynomials are evaluated
        terms (int): highest degree of Chebyshev polynomials.
                        Must be even and at least 6.
    Returns:
        MPCTensor of polynomials evaluated at self of shape `(terms, *self)`
    """
    if terms % 2 != 0 or terms < 6:
        raise ValueError("Chebyshev terms must be even and >= 6")

    polynomials = [self.clone()]
    y = 4 * self.square() - 2
    z = y - 1
    polynomials.append(z.mul(self))

    for k in range(2, terms // 2):
        next_polynomial = y * polynomials[k - 1] - polynomials[k - 2]
        polynomials.append(next_polynomial)

    return crypten.stack(polynomials)


def erf(self):
    r"""
    Approximates the error function of the input tensor using a Taylor approximation.
    """
    method = cfg.functions.erf_method

    if method in ("haar", "bior"):
        luts = LookupTables()
        sgn = self.sign()
        abs = sgn * self
        if method == "haar":
            erf_truncation = cfg.functions.erf_lut_max_bits + cfg.encoder.precision_bits - cfg.functions.erf_haar_size_bits
            msb = abs.div(2**erf_truncation)
            lut = msb.evaluate_lut(luts.LUTs["erf_haar"])
        elif method == "bior":
            erf_truncation = cfg.functions.erf_lut_max_bits + cfg.encoder.precision_bits - cfg.functions.erf_bior_size_bits
            msb, lsb = abs.divmod(2**erf_truncation)
            lut = msb.evaluate_bior_lut(luts.LUTs["erf_bior"], lsb, erf_truncation)
        check = abs < 2**cfg.functions.erf_lut_max_bits
        return sgn * (1-check + lut * check)
    elif method == "Taylor":
        iters = cfg.functions.erf_iterations

        output = self.clone()
        for n in range(1, iters + 1):
            multiplier = ((-1) ** n) / (math.factorial(n) * (2 * n + 1))
            output = output.add(self.pos_pow(2 * n + 1).mul(multiplier))
        return output.mul(2.0 / math.sqrt(math.pi))
        # NOTE: This approximation is not unstable for large tensor values.
    else:
        raise ValueError(f"Unrecognized method {method} for erf")

def gelu(self):
    r"""
    Approximates the gelu function of the input tensor.
    """
    method = cfg.functions.gelu_method

    if method in ("haar", "bior"):
        luts = LookupTables()
        sgn = self.sign()
        abs = sgn * self
        drelu = 1 - self._ltz()
        relu = self * drelu
        if method == "haar":
            truncation = cfg.functions.gelu_lut_max_bits + cfg.encoder.precision_bits - cfg.functions.gelu_haar_size_bits
            msb = abs.div(2**truncation)
            lut = msb.evaluate_lut(luts.LUTs["gelu_haar"])
        elif method == "bior":
            truncation = cfg.functions.gelu_lut_max_bits + cfg.encoder.precision_bits - cfg.functions.gelu_bior_size_bits
            msb, lsb = abs.divmod(2**truncation)
            lut = msb.evaluate_bior_lut(luts.LUTs["gelu_bior"], lsb, truncation)
        check = abs < 2**cfg.functions.gelu_lut_max_bits
        return relu - lut * check
    elif method == "erf":
        gelu = self * (1 + (self / math.sqrt(2)).erf()) / 2
        return gelu
    else:
        raise ValueError(f"Unrecognized method {method} for gelu")

def silu(self):
    r"""
    Approximates the silu function of the input tensor.
    """
    method = cfg.functions.silu_method

    if method in ("haar", "bior"):
        luts = LookupTables()
        sgn = self.sign()
        abs = sgn * self
        drelu = 1 - self._ltz()
        relu = self * drelu
        if method == "haar":
            truncation = cfg.functions.silu_lut_max_bits + cfg.encoder.precision_bits - cfg.functions.silu_haar_size_bits
            msb = abs.div(2**truncation)
            lut = msb.evaluate_lut(luts.LUTs["silu_haar"])
        elif method == "bior":
            truncation = cfg.functions.silu_lut_max_bits + cfg.encoder.precision_bits - cfg.functions.silu_bior_size_bits
            msb, lsb = abs.divmod(2**truncation)
            lut = msb.evaluate_bior_lut(luts.LUTs["silu_bior"], lsb, truncation)
        check = abs < 2**cfg.functions.silu_lut_max_bits
        return relu - lut * check
    elif method == "sigmoid":
        silu = self * self.sigmoid()
        return silu
    else:
        raise ValueError(f"Unrecognized method {method} for gelu")

def softmax(self, dim, **kwargs):
    r"""Compute the softmax of a tensor's elements along a given dimension"""
    # 0-d case
    if self.dim() == 0:
        assert dim == 0, "Improper dim argument"
        return self.new(torch.ones_like((self.data)))

    if self.size(dim) == 1:
        return self.new(torch.ones_like(self.data))

    maximum_value = self.max(dim, keepdim=True)[0]
    logits = self - maximum_value
    with cfg.temp_override({"functions.exp_all_neg": True}):
        numerator = logits.exp()
    with cfg.temp_override({"functions.reciprocal_all_pos": True}):
        inv_denominator = numerator.sum(dim, keepdim=True).reciprocal()
    return numerator * inv_denominator


def log_softmax(self, dim, **kwargs):
    r"""Applies a softmax followed by a logarithm.
    While mathematically equivalent to log(softmax(x)), doing these two
    operations separately is slower, and numerically unstable. This function
    uses an alternative formulation to compute the output and gradient correctly.
    """
    # 0-d case
    if self.dim() == 0:
        assert dim == 0, "Improper dim argument"
        return self.new(torch.zeros((), device=self.device))

    if self.size(dim) == 1:
        return self.new(torch.zeros_like(self.data))

    maximum_value = self.max(dim, keepdim=True)[0]
    logits = self - maximum_value
    normalize_term = exp(logits).sum(dim, keepdim=True)
    result = logits - normalize_term.log()
    return result
