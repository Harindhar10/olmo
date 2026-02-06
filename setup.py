from setuptools import setup, find_packages

setup(
    name="olmochem",
    version="0.1.0",
    description="Minimal library for molecular property prediction with OLMo",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "transformers>=4.35.0",
        "peft>=0.6.0",
        "bitsandbytes>=0.41.0",
        "datasets>=2.14.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "torchmetrics>=1.0.0",
        "mlflow>=2.8.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
        ],
    },
    entry_points={
        "console_scripts": [
            "olmochem-train-cls=scripts.train_classification:main",
            "olmochem-train-reg=scripts.train_regression:main",
            "olmochem-pretrain=scripts.pretrain:main",
            "olmochem-instruction=scripts.train_instruction:main",
        ],
    },
)
