from setuptools import setup, find_packages

setup(
    name="dlmake",
    version="0.1.0",
    author="Shillianne",
    description="A package that groups all necesities for training and evaluating Neural Networks",
    packages=find_packages(),
    install_requires=[
        "matplotlib==3.10.1", "tiktoken==0.8.0", "torchaudio==2.5.1", "torchmetrics==1.6.1", "torchvision==0.20.1"
    ],
    python_requires=">=3.12.0",
    keywords=["Python", "Machine Learning"],
    calssifier=[
        "Programming Language :: Python",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)
