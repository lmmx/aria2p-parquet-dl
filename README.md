# plpq

Test download speeds on 50MB parquet dataset
([Pokemon GPT-4 Captions](https://huggingface.co/datasets/diffusers/pokemon-gpt4-captions)) using:
- regular `httpx.get` with regular parquet URL (`https://`)
- [polars](https://github.com/pola-rs/polars) `read_parquet`  with `hf://` fsspec URL (via HF Hub)
- [aria2p](https://pawamoy.github.io/aria2p/) with parallel split file stream downloading

## Motivation

You can download a file in HuggingFace using the `hf://` protocol provided by the fsspec registry:

```py
import polars as pl

pl.read_parquet("hf://datasets/tatsu-lab/alpaca")
```

But this is basically just as slow as a `httpx.get` call: the above example provides 180K rows in a
minute for me, whereas I can download 3M rows from BigQuery in ~13 seconds, thanks to an 8x speedup
of the `bqstorage` client's parallel streaming tricks. Specific numbers aside: we can do way better.
