# DLmake

A library to pretrain LLMs (and maybe other kinds of deep learning networks)

## Features

- Modular classes to ease instantiation of different models
- Wide assortment of attention and feed forward methods
- Support custom dataloaders
- Integrated `Trainer` class to simplify training
- Benchmark support to evaluate models during and after training
- Hyper parameter search classes to find the best learning parameters

## Installation

Currently, only downloading from source and cloning the repo are supported. Then install the requirements:

```bash
git clone https://github.com/Shillianne/dlmake.git
cd dlmake
python -m pip install -r requirements.txt # or other requirement management programs you may use
```

You may then want to use the library in other projects. For that, you can use pip's supported local project installs like so:

```bash
python -m pip install /path/to/dlmake
```

In order to use the server functionality during training, you will need to install the library with its server optional dependencies:

```bash
python -m pip install /path/to/dlmake[server]
```


## Usage

DLmake is a library for pretraining LLMs (and maybe other deep learning networks), so it doesn't come with any predefined models. You will have to instantiate and train any models you create.
The process of creating a model is exactly the same as it would be with PyTorch, as most classes in this library are subclasses of `nn.Module`. In most scenarios, you shouldn't need to do anything else other than use the declared classes of the library and call a trainer, but some architectures may require further work from your part or a specifc trainer implementation.

In most cases, you will want to make use of the `Config` class, as that should make it easier to declare differnt models based on their parameters, both while testing manually and while launching automated training jobs. Please refer to the [configuration](./dlmake/config/CONFIG.md) documentation for more details on all the parameters.

For any other doubts about how to use the library, look at the [examples](./dlmake/examples/README.md) page.
