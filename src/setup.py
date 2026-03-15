"""
setup.py
--------
Installazione di HybridStyleNet come pacchetto Python.

    pip install -e .          # sviluppo locale (editable)
    pip install .             # installazione standard
"""

from setuptools import setup, find_packages
from pathlib import Path

# Legge il requirements.txt come lista di dipendenze
_HERE = Path(__file__).parent
_REQS = [
    line.strip()
    for line in (_HERE / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name             = "hybridstylenet",
    version          = "0.1.0",
    description      = "RAG-ColorNet: Retrieval-Augmented Grading Network for photographer-specific color grading",
    author           = "Lorenzo Arcioni",
    python_requires  = ">=3.9",
    package_dir      = {"": "src"},
    packages         = find_packages(where="src"),
    install_requires = _REQS,
    extras_require   = {
        "raw": ["rawpy>=0.18.0"],          # solo per file RAW (ARW, DNG, ...)
        "log": ["wandb>=0.15.0"],           # logging wandb opzionale
        "dev": [
            "pytest>=7.0",
            "black",
            "isort",
        ],
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
