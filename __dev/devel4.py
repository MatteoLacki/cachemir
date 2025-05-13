"""
%load_ext autoreload
%autoreload 2
"""
from cachemir.main import Index
from mmapped_df import DatasetWriter
from mmapped_df import open_dataset
from mmapped_df import open_dataset_dct
from pathlib import Path
from prosit_timsTOF_2023_wrapper.main import Prosit2023TimsTofWrapper
from tqdm import tqdm

import functools
import lmdb
import msgpack
import numba
import numpy as np
import pandas as pd


pd.set_option("display.max_rows", 4)
pd.set_option("display.max_columns", None)
ions = pd.read_parquet("/home/matteo/tmp/ions.parquet")

prosit = Prosit2023TimsTofWrapper()

sequences = np.array(["PEPTIDE", "PEPTIDECPEPTIDE"])
amino_acid_cnts = np.array(list(map(len, sequences)))
collision_energies = np.array([30.0, 31.1])
charges = np.array([1, 2])

# with tqdm(total=len(ions)) as pbar:
#     results_small = list(
#         prosit.iter_predict_intensities(
#             sequences=sequences,
#             charges=charges,
#             collision_energies=collision_energies,
#             pbar=pbar,
#         )
#     )

# with tqdm(total=len(ions)) as pbar:
#     results = list(
#         prosit.iter_predict_intensities(
#             sequences=ions.unmoded_sequence.to_numpy(),
#             charges=ions.charge.to_numpy(),
#             collision_energies=ions.average_collision_energy.to_numpy(),
#             pbar=pbar,
#         )
#     )


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

# rm -rf /home/matteo/tmp/test/data
# result = results[0]
with index.open("r") as txn:
    print(txn.stat())

results_iter = prosit.iter_predict_intensities

input_types = (str, int, float)
stats_types = dict(max_ordinal=int, max_fragment_charge=int)


def input_to_bytes(inputs, types):
    assert len(inputs) == len(types)
    inputs_in_user_types = tuple(_type(_input) for _type, _input in zip(types, inputs))
    inputs_bytes = msgpack.packb(inputs_in_user_types)
    return inputs_bytes


path = Path("/home/matteo/tmp/test")


# perhaps add this to prosit directly?
def get_index_and_stats(
    path,
    inputs_df: pd.DataFrame,
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
        types=(int, *tuple(stats_types.values())),
    )

    with (
        Index(path_to_data / "index.lmbd").open("w") as txn,
        DatasetWriter(path_to_data / "data", append_ok=True) as mmapped_df,
    ):
        outputs = [None] * len(inputs_df)
        inputs = inputs_df.itertuples(index=False, name=None)
        if verbose:
            inputs = tqdm(
                inputs,
                total=len(inputs_df),
                desc="Establishing what is cached",
            )
        for idx, _in in enumerate(inputs):
            value = txn.get(in2bytes(_in), None)
            if value is not None:
                outputs[idx] = tuple(msgpack.unpackb(value))

        missing_idxs = [i for i, o in enumerate(outputs) if o is None]
        missing_df = inputs_df.iloc[missing_idxs]

        if len(missing_df) == 0:
            if verbose:
                print("All inputs are cached.")
        else:
            if verbose:
                print("Some input were not cached.")
            missing_dct = {col: missing_df[col].to_numpy() for col in missing_df}

            idx = txn.stat()["entries"]
            missing_results = results_iter(pbar=pbar, **missing_dct)
            if verbose:
                missing_results = tqdm(
                    missing_results,
                    total=len(missing_df),
                    desc="Evaluating missing results",
                )

            out_types = (int, *stats_types)
            for missing_idx, missing_result in zip(missing_idxs, missing_results):
                txn.put(
                    in2bytes(missing_result.input),
                    out2bytes((idx, *missing_result.stats)),
                )
                mmapped_df.append(intensity=missing_result.intensities)
                idx += len(missing_result.intensities)
                assert outputs[missing_idx] is None
                outputs[missing_idx] = missing_result.stats

    index_and_stats = pd.DataFrame(
        outputs, copy=False, columns=["idx", *list(stats_types)]
    )
    raw_data = open_dataset(path_to_data / "data")

    return index_and_stats, raw_data


path = Path("/home/matteo/tmp/test")
index_and_stats, raw_data = get_index_and_stats(
    path,
    inputs_df,
    input_types=dict(sequences=str, charges=int, collision_energies=float),
    stats_types=dict(max_ordinal=int, max_fragment_charge=int),
    verbose=True,
)
