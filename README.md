---
title: ILAB TEMPLATE - Data Science
purpose: Template for python projects tailored to scientific applications (e.g., machine model)
---

# FMGraphCast

Integration of GraphCast into the ILab Foundation Model Framework

## Conda Environment Setup

#### Create Base Environment
    > conda create -n jax -c conda-forge 
    > conda activate jax
    > conda install -c nvidia cuda-python=12.2
    > pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    > conda install -c conda-forge hvplot geoviews rasterio jupyterlab ipykernel ipython ipywidgets numpy xarray dask scipy netCDF4 chex pandas dm-haiku jraph rtree tree trimesh typing_extensions 
    > pip install hydra-core --upgrade
    > pip install dm-tree
    > python -m ipykernel install --user --name=jax


#### Install FMBase
    > git clone https://github.com/nasa-nccs-cds/FoundationModelBase.git
    > cd FoundationModelBase
    > pip install .

#### Install GraphCast
    > git clone https://github.com/google-deepmind/graphcast.git
    > cd graphcast
    > pip install .

#### Workaround for shapely error:
    > pip uninstall -y shapely
    > pip install shapely --no-binary shapely