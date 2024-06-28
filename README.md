<p align="center"><img width="70%" src="https://github.com/jimouris/curl/blob/main/Curl.png" alt="Curl logo" /></p>

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/jimouris/curl/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/jimouris/curl/blob/main/CONTRIBUTING.md)
--------------------------------------------------------------------------------

Curl is a framework for Privacy Preserving Machine Learning (PPML) that builds on top of [CrypTen](https://github.com/facebookresearch/CrypTen) and [PyTorch](https://github.com/pytorch/pytorch).
CrypTen relies on expensive polynomial approximations for evaluating non linear functions such as logarithm, square root, etc.
In contrast, Curl uses lookup tables (LUTs) encoded with Discrete Wavelet Transforms (DWT) to approximate non-linearities that result in faster evaluation while achieving better approximations.

This way, in Curl we are able to evaluate Large Language Models (LLMs) such as GPT-2, GPT Neo, BERT (tiny, base, large).
Curl's goal and model is similar to CrypTen:

> Its goal is to make secure computing techniques accessible to Machine Learning practitioners.
> It currently implements [Secure Multiparty Computation](https://en.wikipedia.org/wiki/Secure_multi-party_computation)
> as its secure computing backend and offers three main benefits to ML researchers:
>
> 1. It is machine learning first. The framework presents the protocols via a `CrypTensor`
>    object that looks and feels exactly like a PyTorch `Tensor`. This allows the user to use
>    automatic differentiation and neural network modules akin to those in PyTorch.
>
> 2. CrypTen is library-based. It implements a tensor library just as PyTorch does.
>    This makes it easier for practitioners to debug, experiment on, and explore ML models.
>
> 3. The framework is built with real-world challenges in mind. CrypTen does not scale back or
>    oversimplify the implementation of the secure protocols.


## How to cite this work
The preprint can be accessed [here](https://eprint.iacr.org/2024/XXX); you can cite this work as follows:
```bibtex
@Misc{EPRINT:GUMVT24,
  author =       "Manuel B. Santos and
                  Dimitris Mouris and
                  Mehmet Ugurbil and
                  Stanislaw Jarecki and
                  José Reis and
                  Shubho Sengupta and
                  Miguel de Vega",
  title =        "{Curl: A Framework for Secure Large Language Models}",
  year =         2024,
  howpublished = "Cryptology ePrint Archive, Report 2024/XXX",
  note =         "\url{https://eprint.iacr.org/2024/XXX}",
}
```

The original CrypTen paper can be accessed [here](https://arxiv.org/pdf/2109.00984.pdf) (documented [here](https://crypten.readthedocs.io/en/latest/)); you can cite this work as follows:
```bibtex
@inproceedings{crypten2020,
  author={B. Knott and S. Venkataraman and A.Y. Hannun and S. Sengupta and M. Ibrahim and L.J.P. van der Maaten},
  title={CrypTen: Secure Multi-Party Computation Meets Machine Learning},
  booktitle={arXiv 2109.00984},
  year={2021},
}
```


## Installing CrypTen (Curl)

CrypTen currently runs on Linux and Mac with Python 3.7.
We also support computation on GPUs.
Windows **is not** supported.
To install Curl, follow the instructions in the [CONTRIBUTING.md](https://github.com/jimouris/curl/blob/main/CONTRIBUTING.md) file.


## Examples

CrypTen has a series of tutorial built on Jupyter notebooks in the [tutorials directory](./tutorials/) as well as examples in the [examples directory](./examples/).

We extend these with our LLM applications in the [LLMs directory](./examples/llms/), which you can run as:
```shell
❯❯ python examples/llms/launcher.py --world_size 2 --tensor_size 1,10 --multiprocess --model GPT2
```

To see the full list of arguments and LLMs available run the script with the `--help` flag:
```shell
❯❯ python examples/llms/launcher.py --help
```

## Disclaimer
This is software for a research prototype and not production-ready code.
This repository builds upon [CrypTen](https://github.com/facebookresearch/CrypTen) and [PyTorch](https://github.com/pytorch/pytorch).
