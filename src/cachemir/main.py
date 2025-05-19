"""
Basic implementation of file-system based memoization scheme.

TODO: 
* each dataset as separate file
* each dataset of the same size in a subfolder grouping them together?
* perhaps context managers should return some classes with methods like len and some fields like types?


"""
from __future__ import annotations

import functools
import json
import lmdb
import msgpack
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Iterable
from typing import Iterator

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import time

from mmapped_df import DatasetWriter
from mmapped_df import open_dataset


@dataclass
class Index:
    """LMDB based index for the memory mapped files.

    Supposed to store indices and some non-grouped statistics.
    """

    path: str
    map_size: int = 2**30
    max_dbs: int = 1

    def __post_init__(self):
        self.path = str(self.path)

    @contextmanager
    def open(self, mode="r"):
        env = lmdb.open(self.path, map_size=self.map_size, max_dbs=self.max_dbs)
        write = mode in ("r+", "w")
        txn = env.begin(write=write)
        try:
            yield txn
            if write:
                txn.commit()
        except Exception:
            if write:
                txn.abort()
            raise
        finally:
            env.close()


@dataclass
class MemoizedOutput:
    input: tuple
    stats: tuple
    data: pd.DataFrame


def input_to_bytes(
    inputs: tuple[int | float | str, ...],
    types: tuple[type, ...],
) -> bytes:
    assert len(inputs) == len(types)
    inputs_in_user_types = tuple(_type(_input) for _type, _input in zip(types, inputs))
    inputs_bytes = msgpack.packb(inputs_in_user_types)
    return inputs_bytes


# TODO: perhaps add this to the Prosit model instance.
def get_index_and_stats(
    path: Path | str,
    inputs_df: pd.DataFrame,
    results_iter: Callable[[tuple[Iterable, ...]], Iterator[MemoizedOutput]],
    input_types: dict[str, type],
    stats_types: dict[str, type],
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    in2bytes = functools.partial(
        input_to_bytes,
        types=tuple(input_types.values()),
    )
    out2bytes = functools.partial(
        input_to_bytes,
        types=(int, int, *tuple(stats_types.values())),
    )

    with (
        Index(path / "index.lmbd").open("w") as txn,
        DatasetWriter(path / "data", append_ok=True) as mmapped_df,
    ):
        inputs = inputs_df.itertuples(index=False, name=None)
        if verbose:
            inputs = tqdm(
                inputs,
                total=len(inputs_df),
                desc="Establishing what is cached",
            )
        outputs = []
        for _in in inputs:
            value = txn.get(in2bytes(_in), None)
            if value is not None:
                value = tuple(msgpack.unpackb(value))
            outputs.append(value)

        missing_idxs = [i for i, o in enumerate(outputs) if o is None]
        missing_df = inputs_df.iloc[missing_idxs]

        if len(missing_idxs) == 0:
            if verbose:
                print("All calls were in cache.")
        else:
            if verbose:
                print(f"{len(missing_idxs)} calls were not in cache.")
            idx = txn.stat()["entries"]  # current maximal idx
            missing_results = results_iter(
                **{col: missing_df[col].to_numpy() for col in missing_df},
            )
            if verbose:
                missing_results = tqdm(
                    missing_results,
                    total=len(missing_idxs),
                    desc="Saving missing results",
                )

            for missing_idx, missing_result in zip(missing_idxs, missing_results):
                out_stats = (idx, len(missing_result.data), *missing_result.stats)
                txn.put(
                    in2bytes(missing_result.input),
                    out2bytes(out_stats),
                )
                mmapped_df.append_df(missing_result.data)
                idx += len(missing_result.data)
                assert outputs[missing_idx] is None
                outputs[missing_idx] = out_stats

    index_and_stats = pd.DataFrame(
        outputs, copy=False, columns=["idx", "cnt", *list(stats_types)]
    )
    raw_data = open_dataset(path / "data")

    return index_and_stats, raw_data
