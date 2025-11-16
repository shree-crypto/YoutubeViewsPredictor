"""
Setup configuration for YouTube Views Predictor package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="youtube-views-predictor",
    version="1.0.0",
    author="shree-crypto",
    description="Machine learning system to predict YouTube video views",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shree-crypto/YoutubeViewsPredictor",
    packages=find_packages(exclude=["tests", "notebooks"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.1.3,<2.3.0",
        "numpy>=1.26.2,<2.0.0",
        "scikit-learn>=1.3.2,<1.6.0",
        "xgboost>=2.0.2,<3.0.0",
        "lightgbm>=4.1.0,<5.0.0",
        "streamlit>=1.28.2,<2.0.0",
        "matplotlib>=3.8.2,<4.0.0",
        "seaborn>=0.13.0,<0.14.0",
        "plotly>=5.18.0,<6.0.0",
        "nltk>=3.8.1,<4.0.0",
        "textblob>=0.17.1,<0.19.0",
        "joblib>=1.3.2,<2.0.0",
        "pydantic>=2.5.0,<3.0.0",
        "pyyaml>=6.0.1,<7.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0,<9.0.0",
            "pytest-cov>=4.1.0,<6.0.0",
            "pylint>=3.0.0,<4.0.0",
            "flake8>=6.1.0,<8.0.0",
            "black>=23.11.0,<25.0.0",
            "isort>=5.12.0,<6.0.0",
            "mypy>=1.7.0,<2.0.0",
            "pre-commit>=3.5.0,<4.0.0",
        ],
        "docs": [
            "sphinx>=7.2.0,<9.0.0",
            "sphinx-rtd-theme>=2.0.0,<3.0.0",
        ],
        "api": [
            "transformers>=4.35.2,<4.50.0",
            "torch>=2.2.0",
            "google-api-python-client>=2.108.0,<3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "youtube-predictor-train=train_model:train_model",
            "youtube-predictor-app=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json"],
    },
)
