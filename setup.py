import setuptools

NAME = "torchcomp"
VERSION = "0.1.1"
MAINTAINER = "Chin-Yun Yu"
EMAIL = "chin-yun.yu@qmul.ac.uk"


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=NAME,
    version=VERSION,
    author=MAINTAINER,
    author_email=EMAIL,
    description="Fast, efficient, and differentiable dynamic range compression/expansion/limiting for PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yoyololicon/torchcomp",
    packages=["torchcomp"],
    install_requires=["torch", "torchaudio", "torchlpc", "numpy", "numba"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
