from setuptools import setup, find_packages

setup(
    name="dlmake",
    version="0.1.1",
    author="Shillianne",
    description="A package that groups all necesities for training and evaluating Neural Networks",
    packages=find_packages(),
    install_requires=[
        "matplotlib>=3.10.8", "tiktoken>=0.12.0", "torchmetrics==1.8.2", "torchvision==0.25.0", "bitsandbytes==0.49.1"
    ],
    python_requires=">=3.11.0",
    keywords=["Python", "Machine Learning"],
    calssifier=[
        "Programming Language :: Python",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)
