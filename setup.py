from setuptools import setup, find_packages

setup(
    name="quadricsreg",
    version="1.0.0",
    description="QuadricsReg: Quadric Surface Based Point Cloud Registration",
    packages=find_packages(),
    python_requires=">=3.8,<3.11",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "open3d>=0.18.0",
        "gtsam>=4.2",
        "scikit-learn>=1.3",
        "pyyaml",
    ],
)
