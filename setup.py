from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="youtube-views-predictor",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="ML model to predict YouTube views based on video metadata",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shree-crypto/YoutubeViewsPredictor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "lightgbm>=3.3.0",
        "google-api-python-client>=2.40.0",
        "fastapi>=0.85.0",
        "uvicorn>=0.18.0",
        "pydantic>=1.9.0",
        "python-dotenv>=0.19.0",
        "pyyaml>=6.0",
        "requests>=2.27.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "pylint>=2.12.0",
        ],
    },
)
