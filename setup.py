from setuptools import setup, find_packages

setup(
    name="epicdiffusion",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "diffusers>=0.20.0",
        "transformers>=4.30.0",
        "opencv-python>=4.7.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "Pillow>=9.5.0",
        "safetensors>=0.3.0",
        "openai>=0.27.0",
        "tqdm>=4.65.0"
    ],
    python_requires=">=3.9",
)
