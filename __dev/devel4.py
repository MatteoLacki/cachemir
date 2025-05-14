"""
%load_ext autoreload
%autoreload 2
"""
from pathlib import Path

import numpy as np
import pandas as pd

from prosit_timsTOF_2023_wrapper.main import Prosit2023TimsTofWrapper

from cachemir.main import get_index_and_stats

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

K = 121
inputs_df.iloc[K]
(
    idx,
    cnt,
    _,
    _,
) = index_and_stats.iloc[K]
raw_data.iloc[idx : idx + cnt]

type(msgpack.packb((1, 23)))
