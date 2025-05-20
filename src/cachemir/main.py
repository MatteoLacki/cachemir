from __future__ import annotations

import json

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import pandas as pd

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any
from typing import Callable
from typing import ContextManager
from typing import Iterable
from typing import Iterator

from tqdm import tqdm

from cachemir.pandas_ops import df2dct
from cachemir.serialization import derive_input_types
from cachemir.serialization import enforce_types
from cachemir.serialization import type_to_name


@dataclass
class EncodingTransation:
    """A wrappar around lmbd.Transation that simplfies its usage.

    Arguments:
        lmdb_transaction (lmbd.Transation): A transaction.
        encoder (Callable): A function taking input and serializing it to bytes.
        decoder (Callable): The inverse of `encoder`.
    """

    lmdb_transaction: lmbd.Transation
    encoder: Callable[[Any], bytes]
    decoder: Callable[[bytes], Any]

    def __setitem__(self, key, value):
        self.lmdb_transaction.put(self.encoder(key), self.encoder(value))

    def __getitem__(self, key):
        return self.decoder(self.lmdb_transaction.get(self.encoder(key)))

    def get(self, key, default):
        value = self.lmdb_transaction.get(self.encoder(key))
        if value is None:
            return default
        return self.decoder(value)

    def __contains__(self, key) -> bool:
        value = self.lmdb_transaction.get(self.encoder(key))
        return not value is None

    def __len__(self):
        return self.lmdb_transaction.stat()["entries"]


def ITERTUPLES(df):
    """How pandas should have set defaults in .itertuples"""
    yield from df.itertuples(index=False, name=None)


@dataclass
class SimpleLMDB:
    """A simple wrapper around LMDB that works.

    The only inoptimal things is that with each open we use lmdb.open.
    But we have long transcations so that's not a problem and without it it would not work.

    Arguments:
        encoder (Callable): A function taking input and serializing it to bytes.
        decoder (Callable): The inverse of `encoder`.
        path (str): Where to keep the data (a folder).
        map_size (int): Size of the DB.
    """

    path: str
    encoder: Callable[[Any], bytes] = partial(
        msgpack.packb, default=msgpack_numpy.encode
    )
    decoder: Callable[[bytes], Any] = partial(
        msgpack.unpackb, object_hook=msgpack_numpy.decode
    )
    map_size: int = 2**30

    def __post_init__(self):
        self.path = str(self.path)

    @contextmanager
    def open(self, mode="r") -> ContextManager[EncodingTransation]:
        """Open a transacation."""
        env = lmdb.open(self.path, map_size=self.map_size, max_dbs=1)
        write = mode in ("r+", "w")
        txn = EncodingTransation(
            lmdb_transaction=env.begin(write=write),
            encoder=self.encoder,
            decoder=self.decoder,
        )
        try:
            yield txn
            if write:
                txn.lmdb_transaction.commit()
        except Exception:
            if write:
                txn.lmdb_transaction.abort()
            raise
        finally:
            env.close()

    def iter_IO(
        self,
        iter_eval: Callable[[pd.DataFrame], Iterable[tuple[tuple, pd.DataFrame]]],
        inputs_df: pd.DataFrame,
        input_types: dict[str, type] | None = None,
        meta: dict[str, str | float | int] = {},
    ) -> Iterator[tuple[tuple, dict[str, npt.NDArray]]]:
        """Iter results from cache if they are there, else get them there.

        Arguments:
            iter_eval (Callable): A function that returns an iterable sequence of pairs (input, output), where input is a tuple and output a data frame.
            inputs_df (pd.DataFrame): Input to query the DB. A subset of those might be submitted to `iter_eval` so make sure this can go smoothly.
            input_types (dict|None): Optional types for the input. If not provided will be derived. Output types will be automatically saved.
            meta (dict): Optional information about what is saved.

        Yields:
            tuple of inputs tuple and outputs dictionary mapping column names to numpy arrays.
        """
        if input_types is None:
            input_types = derive_input_types(inputs_df)
        serializable_input_types = {c: type_to_name[t] for c, t in input_types.items()}
        sanitize_types = partial(enforce_types, types=tuple(input_types.values()))

        with self.open("w") as txn:
            if not "__input_types__" in txn:
                txn["__input_types__"] = serializable_input_types
            assert txn["__input_types__"] == serializable_input_types

            if len(meta) and not "__meta__" in txn:
                txn["__meta__"] = meta

            missing_idxs = [
                i
                for i, inputs in enumerate(map(sanitize_types, ITERTUPLES(inputs_df)))
                if not inputs in txn
            ]
            if len(missing_idxs):
                missing_df = inputs_df.iloc[missing_idxs].drop_duplicates()
                for inputs, results_df in iter_eval(missing_df):
                    txn[sanitize_types(inputs)] = df2dct(results_df)

        with self.open("r") as txn:  # assuming appenders only
            for inputs in map(sanitize_types, ITERTUPLES(inputs_df)):
                yield inputs, txn[inputs]


#### BELOW: COMPATIBILITY, NO IMMEDIATE USE-CAS


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


def get_index_and_stats(
    path: Path | str,
    inputs_df: pd.DataFrame,
    results_iter: Callable[[tuple[Iterable, ...]], Iterator[MemoizedOutput]],
    input_types: dict[str, type],
    stats_types: dict[str, type],
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ """
    from mmapped_df import DatasetWriter
    from mmapped_df import open_dataset

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    in2bytes = partial(
        input_to_bytes,
        types=tuple(input_types.values()),
    )
    out2bytes = partial(
        input_to_bytes,
        types=(int, int, *tuple(stats_types.values())),
    )
    with (
        Index(path / "index.lmbd").open("w") as txn,
        DatasetWriter(path / "data", append_ok=True) as mmapped_df,
    ):
        missing_idxs = [
            i
            for i, row in enumerate(ITERTUPLES(inputs_df))
            if txn.get(in2bytes(row), None) is None
        ]
        missing_df = inputs_df[list(input_types)].iloc[missing_idxs]

        if len(missing_df) > 0:
            missing_df = missing_df.drop_duplicates()
            idx = txn.stat()["entries"]  # current maximal idx
            missing_results = results_iter(
                **{col: missing_df[col].to_numpy() for col in missing_df},
            )
            for missing_idx, missing_result in zip(missing_idxs, missing_results):
                out_stats = (idx, len(missing_result.data), *missing_result.stats)
                txn.put(
                    in2bytes(missing_result.input),
                    out2bytes(out_stats),
                )
                mmapped_df.append_df(missing_result.data)
                idx += len(missing_result.data)

    with Index(path / "index.lmbd").open("r") as txn:
        outputs = []
        for _in in ITERTUPLES(inputs_df):
            value = txn.get(in2bytes(_in), None)
            assert value is not None
            outputs.append(tuple(msgpack.unpackb(value)))

    index_and_stats = pd.DataFrame(
        outputs, copy=False, columns=["idx", "cnt", *list(stats_types)]
    )
    raw_data = open_dataset(path / "data")

    return index_and_stats, raw_data
