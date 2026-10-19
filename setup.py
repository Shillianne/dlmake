from setuptools import setup, find_packages

setup(
    name="dlmake",
    version="0.2.0",
    author="Shillianne",
    description="A package that groups all necesities for training and evaluating transformers (and maybe other networks)",
    packages=find_packages(),
    install_requires=[
        "matplotlib>=3.10.8", "tiktoken>=0.12.0", "torchmetrics==1.9.0", "torchvision==0.29.1", "bitsandbytes==0.49.1"
    ],
    python_requires=">=3.12.0",
    keywords=["Python", "Machine Learning"],
    calssifier=[
        "Programming Language :: Python",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)
