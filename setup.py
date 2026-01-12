from setuptools import setup, find_packages

setup(
    name="prism-miner",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "huggingface_hub>=0.20.0",
        "openai>=1.0.0",
        "pandas>=2.0.0",
        "tqdm>=4.66.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "structlog>=24.1.0",
    ],
    python_requires=">=3.9",
    author="Prism Commerce",
    description="Amazon Review & Metadata Mining Service",
)
