import os
import time
from io import BytesIO

import aria2p
import httpx
import polars as pl
from pytest import fixture


@fixture(scope="session")
def aria2_api(port=6800):
    aria2_host = "http://localhost"
    aria2_secret = "fmpowkmi2390mp3dg8ijd92hunybqgdiuj3oinchn9ufhn9823hn"
    aria2_client = aria2p.Client(host=aria2_host, port=port, secret=aria2_secret)
    aria2_options = {"min_split_size": 1000}
    aria2_client.change_global_option(aria2_options)
    aria2_api = aria2p.API(aria2_client)
    return aria2_api


def test_pokemon_fsspec_url(tmp_path):
    df = pl.DataFrame._read_parquet(
        "hf://datasets/diffusers/pokemon-gpt4-captions/data/**/*.parquet"
    )
    print(df)


def test_pokemon_httpx_download(tmp_path):
    base = "https://huggingface.co/api/datasets"
    url = f"{base}/diffusers/pokemon-gpt4-captions/parquet/default/train/0.parquet"
    data = httpx.get(url, follow_redirects=True).content
    df = pl.read_parquet(BytesIO(data))
    print(df)


def test_pokemon_aria2_download(aria2_api, tmp_path):
    base = "https://huggingface.co/api/datasets"
    url = f"{base}/diffusers/pokemon-gpt4-captions/parquet/default/train/0.parquet"
    download_path = tmp_path / "hf_dataset.parquet"
    aria2_options = {"dir": str(tmp_path), "out": download_path.name}
    download = aria2_api.add_uris([url], options=aria2_options)
    # Wait for the download to complete
    while not download.is_complete:
        time.sleep(0.2)  # Sleep to prevent busy waiting
        download.update()
    # Check if the download was successful
    assert download.is_complete, "Download did not complete successfully"
    df = pl.read_parquet(download_path)
    print(df)
