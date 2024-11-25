from setuptools import setup, find_packages

setup(
    name="cross-species-vae",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "torch-geometric>=2.3.0",
        "numpy>=1.24.0",
        "scanpy>=1.9.3",
        "anndata>=0.9.1",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "mlflow>=2.3.0",
    ],
    author="Alexander Tarashansky",
    description="A cross-species variational autoencoder for single-cell data analysis",
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
