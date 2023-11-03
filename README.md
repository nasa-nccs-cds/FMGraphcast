---
title: ILAB TEMPLATE - Data Science
purpose: Template for python projects tailored to scientific applications (e.g., machine model)
---

# FMGraphCast

Integration of GraphCast into the ILab Foundation Model Framework

## Conda Environment Setup

#### Create Base Environment
    > conda create -n fmgc -c conda-forge 
    > conda activate fmgc
    > conda install -c conda-forge ipywidgets numpy xarray dask matplotlib scipy netCDF4 cartopy chex dm-haiku jax jraph pandas rtree tree trimesh typing_extensions 
    > pip install hydra-core --upgrade
    > pip install dm-tree

#### Install FMBase
    > git clone https://github.com/nasa-nccs-cds/FoundationModelBase.git
    > cd FoundationModelBase
    > pip install .

#### Install GraphCast
    > git clone https://github.com/google-deepmind/graphcast.git
    > cd graphcast
    > pip install .