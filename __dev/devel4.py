"""
%load_ext autoreload
%autoreload 2
"""
from cachemir.main import Index
from mmapped_df import DatasetWriter
from mmapped_df import open_dataset
from mmapped_df import open_dataset_dct
from pathlib import Path
from prosit_timsTOF_2023_wrapper.main import MemoizedOutput
from prosit_timsTOF_2023_wrapper.main import Prosit2023TimsTofWrapper
from tqdm import tqdm

import functools
import lmdb
import msgpack
import numba
import numpy as np
import pandas as pd
import typing

pd.set_option("display.max_rows", 4)
pd.set_option("display.max_columns", None)
ions = pd.read_parquet("/home/matteo/tmp/ions.parquet")

prosit = Prosit2023TimsTofWrapper()

sequences = np.array(["PEPTIDE", "PEPTIDECPEPTIDE"])
amino_acid_cnts = np.array(list(map(len, sequences)))
collision_energies = np.array([30.0, 31.1])
charges = np.array([1, 2])

inputs_df = pd.concat(
    [
        ions[["unmoded_sequence", "charge", "average_collision_energy"]],
        pd.DataFrame(
            dict(
                unmoded_sequence=sequences,
                charge=charges,
                average_collision_energy=collision_energies,
            )
        ),
    ],
    axis=0,
)
inputs_df.columns = ["sequences", "charges", "collision_energies"]


def input_to_bytes(inputs, types):
    assert len(inputs) == len(types)
    inputs_in_user_types = tuple(_type(_input) for _type, _input in zip(types, inputs))
    inputs_bytes = msgpack.packb(inputs_in_user_types)
    return inputs_bytes


def get_index_and_stats(
    path,
    inputs_df: pd.DataFrame,
    results_iter: typing.Iterator[MemoizedOutput],
    input_types: dict[str, type],
    stats_types: dict[str, type],
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                print("All inputs are cached.")
        else:
            if verbose:
                print("Some input were not cached.")
            idx = txn.stat()["entries"]  # current biggest result.
            missing_results = results_iter(
                **{col: missing_df[col].to_numpy() for col in missing_df}
            )
            if verbose:
                missing_results = tqdm(
                    missing_results,
                    total=len(missing_idxs),
                    desc="Evaluating missing results",
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


# rm -rf /home/matteo/tmp/test2
# path = Path("/home/matteo/tmp/test")
path = Path("/home/matteo/tmp/test2")
index_and_stats, raw_data = get_index_and_stats(
    path=path,
    inputs_df=inputs_df,
    results_iter=prosit.iter_predict_intensities,
    input_types=dict(sequences=str, charges=int, collision_energies=float),
    stats_types=dict(max_ordinal=int, max_fragment_charge=int),
    verbose=True,
)
K = 1000
inputs_df.iloc[K]
idx, cnt, _, _ = index_and_stats.iloc[K]
raw_data[idx : idx + cnt]

path = Path("/home/matteo/tmp/test7")
index_and_stats, raw_data = get_index_and_stats(
    path=path,
    inputs_df=inputs_df,
    results_iter=prosit.iter_predict_intensities,
    input_types=dict(sequences=str, charges=int, collision_energies=float),
    stats_types=dict(max_ordinal=int, max_fragment_charge=int),
    verbose=True,
)

raw_data[4182 : 4182 + 54]
