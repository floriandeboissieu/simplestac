The example notebooks `local_stac.py` and `remote_stac.py` are showing
how to build and use ItemCollection objects, with local data and remote data (e.g. Planetary Computer and Element84).

In order to execute the notebooks, several dependencies not coming with the package are necessary: ipykernels, planetary-computer, xpystac.
Use `mamba` or `pip` depending on your type of environment, example:
```shell
# conda env
mamba install -n simplestac ipykernels planetary-computer xpystac
# virtualenv
pip install ipykernels planetary-computer xpystac
```

Both notebooks are based using example data that can be downloaded
and extracted with script `download_data.py` (thus to run first).

