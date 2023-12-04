# SimpleSTAC

SimpleSTAC eases the use of pystac.

# Install

It is recommended to install the package in a virtual environment. See
[miniforge](https://github.com/conda-forge/miniforge) for conda/mamba, or 
[venv](https://docs.python.org/3/library/venv.html) for virtualenv.

Within a mamba env:
```shell
mamba env create -f https://gitlab.irstea.fr/umr-tetis/stac/simplestac/-/raw/main/environment.yml

```

Within a virtualenv:
```shell
pip install git+https://gitlab.irstea.fr/umr-tetis/stac/simplestac
```

# Examples

## Create a local STAC ItemCollection

```python
from simplestac.utils import collection_format, build_item_collection

tmpdir = Path(tempfile.TemporaryDirectory(prefix="fordead_").name)
tmpdir = Path('/tmp/fordead_xpxnrg_e')
data_dir = tmpdir/'fordead_data-main'
col_file = tmpdir / "collection.json"


if not data_dir.exists():
    data_url = Path("https://gitlab.com/fordead/fordead_data/-/archive/main/fordead_data-main.zip")

    with tempfile.TemporaryDirectory() as tmpdir2:
        dl_dir = Path(tmpdir2)
        zip_path, _ = urlretrieve(data_url, dl_dir / data_url.name)
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(tmpdir.mkdir_p())

    # Let's start from the example collection built as in static_stac.py
    # directory containing the remote sensing scenes
    image_dir = data_dir / "sentinel_data/dieback_detection_tutorial/study_area"
    fmt = collection_format("S2_L2A_THEIA")
    col = build_item_collection(image_dir, fmt)
    col.save_object(col_file)

tmpdir.rmtree()
```

