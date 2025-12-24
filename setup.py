"""Setup script for PySpark Parameter Server & Distributed Trainer."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pyspark-ps",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Distributed Parameter Server and Training Framework for PySpark + TensorFlow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pyspark-ps",
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "pyspark>=3.2.0",
        "tensorflow>=2.8.0",
        "pyarrow>=8.0.0",
        "msgpack>=1.0.0",
        "cloudpickle>=2.0.0",
        "boto3>=1.20.0",
    ],
    extras_require={
        "s3": [
            "s3fs>=2022.1.0",
        ],
        "compression": [
            "lz4>=4.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "mypy>=0.990",
        ],
        "all": [
            "s3fs>=2022.1.0",
            "lz4>=4.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "mypy>=0.990",
        ],
    },
    entry_points={
        "console_scripts": [
            "pyspark-ps-example=pyspark_ps.examples.simple_training:main",
        ],
    },
)
