"""Module setuptools script."""

from setuptools import setup, find_packages

description = "Integration of GraphCast into the ILab Foundation Model Framework"

setup(
    name="fmgraphcast",
    version="0.1",
    description=description,
    long_description=description,
    author="NASA Innovation Lab",
    license="Apache License, Version 2.0",
    keywords="GraphCast Foundation Model",
    url="https://github.com/nasa-nccs-cds/FMGraphcast.git",
    packages=find_packages(),
    install_requires=[  "numpy", "xarray", "dask", "matplotlib", "scipy", "netCDF4", "hydra-core", "cartopy", "chex",
                        "colabtools", "dm-haiku", "jax", "jraph", "pandas", "rtree", "tree", "trimesh", "typing_extensions" ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
