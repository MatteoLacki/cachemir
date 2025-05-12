"""
%load_ext autoreload
%autoreload 2
"""


import json
import os

from contextlib import ExitStack
from contextlib import contextmanager
from time import sleep

from massimo.memmapped_arrays import open_memmapped_data

import lmdb
import msgpack
import numba
import numpy as np
import pandas as pd
import pkg_resources

from dataclasses import dataclass
from functools import wraps
from numba_progress import ProgressBar
from pathlib import Path
from tqdm import tqdm

from prosit_timsTOF_2023_wrapper.main import Prosit2023TimsTofWrapper

from massimo.memmapped_arrays import MemmappedArrays
from massimo.mergers import get_index


pd.set_option("display.max_rows", 4)
pd.set_option("display.max_columns", None)


ions = pd.read_parquet("/home/matteo/tmp/ions.parquet")
fragment_library_path = Path("/home/matteo/tmp/fragment_library")


prosit = Prosit2023TimsTofWrapper()


sequences = np.array(["PEPTIDE", "PEPTIDECPEPTIDE"])
amino_acid_cnts = np.array(list(map(len, sequences)))
collision_energies = np.array([30.0, 31.1])
charges = np.array([1, 2])

with tqdm(total=len(ions)) as pbar:
    results_small = list(
        prosit.iter_predict_intensities(
            sequences=sequences,
            charges=charges,
            collision_energies=collision_energies,
            pbar=pbar,
        )
    )

with tqdm(total=len(ions)) as pbar:
    results = list(
        prosit.iter_predict_intensities(
            sequences=ions.unmoded_sequence.to_numpy(),
            charges=ions.charge.to_numpy(),
            collision_energies=ions.average_collision_energy.to_numpy(),
            pbar=pbar,
        )
    )

ions["fragment_intensities_cnt"] = [len(o.intensities) for o in results]


input_types = (str, int, float)
output_types = (int, int, int)

fragment_intensities_index = get_index(ions.fragment_intensities_cnt)
fragment_library_path.mkdir()


# this likely would not hurt to have in the db.
pipfreeze = [
    f"{dist.project_name}=={dist.version}" for dist in pkg_resources.working_set
]

# the name of the lmbd file should reflect one function call. should it?
lmbd = lmdb.open("/home/matteo/tmp/mydata.lmdb", map_size=2 * 1024 * 1024 * 1024)

len(lmbd)

def input_to_bytes(inputs, types):
    assert len(inputs) == len(types)
    inputs_in_user_types = tuple(_type(_input) for _type, _input in zip(types, inputs))
    inputs_bytes = msgpack.packb(inputs_in_user_types)
    return inputs_bytes

cachemir = Cachemir(
    folder=fragment_library_path,
    scheme=dict(
        index=np.uint32,
        intensity=np.float32,
    )
)

lmbd = lmdb.open("/home/matteo/tmp/mydata.lmdb", map_size=2 * 1024 * 1024 * 1024)
with lmbd.begin() as txn:
    stats = txn.stat()
    print("Number of keys:", stats["entries"])


with cachemir.read() as reader: # reader
    idx = reader.in2idx[("SEQUENCE", charge, collision_energy)]
    reader.idx

    
import lmdb

from __future__ import annotations
from dataclasses import dataclass


folder = '/home/matteo/tmp/dupa'
scheme = dict(intensity=np.float32)

@contextmanager
def open_index(path, mode="r", map_size_bytes=2**30):
    env = lmdb.open(path, map_size=map_size_bytes, max_dbs=1)
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



from functools import partial


# we could also make it that Cachemir will contain index and dataset handler openers.





class Cachemir:
    @classmethod
    def new(cls, folder: str|Path, scheme: dict["str",type], *args, **kwargs) -> Cachemir:
        folder = Path(folder)
        folder.mkdir()
        (folder / "index").mkdir()
        scheme = [(column_name, dtype) for column_name, dtype in scheme.items()]
        with open(folder / "scheme.json", "w") as f:
            json.dump(np.dtype(scheme).descr, f)
        return cls(folder, *args, **kwargs)

    def __init__(self, folder: str | Path, index_kwargs=dict(map_size=2**30, max_dbs=1)):
        self.folder = Path(folder)
        self.index = Index(str(self.folder / "index"), **index_kwargs)
        with open(self.folder / "scheme.json", "r") as f:
            self.scheme = {col: np.dtype(dtype_str) for col, dtype_str in json.load(f)}

    def index_stats(self):
        with self.index.open("r") as txn:
            return txn.stat()
        
    def close():
        self.env.close()

    def __len__(self) -> int:
        return self.index_stats()["entries"]

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def connect(self, mode="r", resize:int=0):
        pass

cachemir = Cachemir("/home/matteo/tmp/dupa3")
cachemir.index_stats()

# now, we need to give possibility to open the datasets and write to them.


# perhaps the index should be a class in its own and Cachemir should simply open it in a context manager? Not stupid!
# and each variable the same as its own class
# and cachemir simply opens them all.






%%timeit
with open_index("/home/matteo/tmp/dupa3") as txn:
    x = txn.stat()




import fcntl
import numpy as np

from contextlib import ExitStack
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import Union


@contextmanager
def locked_open_multiple(
    paths: Iterable[Union[str, Path]],
    mode: str = "r",
    _writer_lock_type: int = fcntl.LOCK_EX,
    _reader_lock_type: int = fcntl.LOCK_SH,
    _reader_modes: tuple[str, ...] = ("r", "rb"),
) -> Iterator[dict[Path, Any]]:
    """
    Context manager to open and lock multiple files.

    For `.npy` files, returns numpy memmaps.
    For `.lmdb` files, returns the Path (no external lock).
    For all others, returns standard file handles with fcntl locking.

    If mode is not a reader mode, memmaps will be flushed on exit.

    Yields:
        Dict[Path, file-like or memmap or Path]
    """
    with ExitStack() as stack:
        opened = {}
        memmaps = []  # To flush if writing

        for path in sorted(map(Path, paths)):
            suffix = path.suffix
            if suffix == ".lmdb":
                opened[path] = path
            elif suffix == ".npy":
                file_mode = mode if mode in ("r", "r+", "w+", "c") else "r+"
                f = open(path, file_mode)
                fcntl.flock(
                    f,
                    _reader_lock_type if mode in _reader_modes else _writer_lock_type,
                )
                stack.callback(lambda f=f: (fcntl.flock(f, fcntl.LOCK_UN), f.close()))
                mmap = np.load(f, mmap_mode=mode)
                opened[path] = mmap
                if mode not in _reader_modes:
                    memmaps.append(mmap)
            else:
                f = open(path, mode)
                fcntl.flock(
                    f,
                    _reader_lock_type if mode in _reader_modes else _writer_lock_type,
                )
                stack.enter_context(f)
                stack.callback(lambda f=f: fcntl.flock(f, fcntl.LOCK_UN))
                opened[path] = f

        try:
            yield opened
        finally:
            for mmap in memmaps:
                mmap.flush()


cachemir = Cachemir.new("/home/matteo/tmp/dupa3", scheme=scheme)



len(cachemir)
cachemir2 = Cachemir("/home/matteo/tmp/dupa3")
len(cachemir2)
def enlarge_file(file_path: str | Path, resize: int, datatype: str):
    assert resize >= 0
    with locked_open(file_path, "ab") as f:
        f.truncate(os.path.getsize(filename) + resize * np.dtype(datatype).itemsize)


cachemir.env.close()
cachemir2.env.close()



cachemir = Cachemir("/home/matteo/tmp/dupa")
len(cachemir)




with cachemir(mode="w+", index, resize=) as lib:
    lib.index


# writing
with (
    MemmappedArrays(
        folder=fragment_library_pcath,
        column_to_type_and_shape=dict(
            index=(np.uint32, len(fragment_intensities_index)),
            intensity=(np.float32, fragment_intensities_index[-1]),
        ),
        mode="w+",
    ) as fragments_library,
    lmbd.begin(write=True) as input2idx,
):
    fragments_library["index"][:] = fragment_intensities_index

    i = 0
    for fragments in tqdm(results):
        n = len(fragments.intensities)
        fragments_library["intensity"][i : i + n] = fragments.intensities
        input2idx.put(
            input_to_bytes(fragments.input, input_types),
            input_to_bytes(
                (
                    i,
                    fragments.max_ordinal,
                    fragments.max_fragment_charge,
                ),
                output_types,
            ),
        )
        i += n


# retrieval
with env.begin() as txn:
    with txn.cursor() as cursor:
        res = {
            tuple(msgpack.unpackb(key)): msgpack.unpackb(value) for key, value in cursor
        }

# but this is not what I wanted.

# here we need to collect things to run and get things already there.
# perhaps simplest to get pd.DataFrame in and one with extra columns out?

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
outputs = np.array([None] * len(inputs_df))

# get back old results
pbar = None
missing = []
with lmdb.open("/home/matteo/tmp/mydata.lmdb", readonly=True) as env:
    with env.begin() as txn:
        for idx, inputs in enumerate(inputs_df.itertuples(index=False, name=None)):
            value = txn.get(input_to_bytes(inputs, input_types), None)
            outputs[idx] = tuple(msgpack.unpackb(value)) if value else None
            if not value:
                missing.append(idx)
            if pbar is not None:
                pbar.update(1)

missing_df = inputs_df.iloc[missing]
with tqdm(total=len(ions)) as pbar:
    missing_results = list(
        prosit.iter_predict_intensities(
            sequences=missing_df.sequences.to_numpy(),
            charges=missing_df.charges.to_numpy(),
            collision_energies=missing_df.collision_energies.to_numpy(),
            pbar=pbar,
        )
    )


# all that needs to be in the blocking context.
data_update_size = sum(len(o.intensities) for o in missing_results)
index_update_size = len(missing_results)
library_current_size = len(inputs_df) - index_update_size


folder = Path("/home/matteo/tmp/fragment_library")


column2resize = dict(index=index_update_size, intensity=data_update_size)


# where should this be done? also in the Context manager of the structure.
with locked_open_multiple(file_paths, "a") as files:
    for f in files:
        f.write("Appended safely\n")


# need mode.
def enlarge_file(file_path: str | Path, resize: int, datatype: str):
    assert resize >= 0
    with locked_open(file_path, "ab") as f:
        f.truncate(os.path.getsize(filename) + resize * np.dtype(datatype).itemsize)


def enlarge_data(folder: str | Path, **column2resize: int):
    folder = Path(folder)
    with open(folder / "scheme.json", "r") as f:
        scheme = dict(json.load(f))
    for col in column2resize:
        assert col in scheme
    for col, resize in column2resize.items():
        enlarge_file(file_path=folder / col, resize=resize, datatype=scheme[col])


enlarge_data(folder)


# OK, now need to simply put them in. That does not seem very process friendly to me.
# but it needs to be this way to make data usable from numba without chunking.
# but perhaps chunking is not that bad...


# OK, how to deal now?
# the outcomes will not have a fixed size... but they are likely small.
# so simply get all of them.

# inputs_df


# # problem: need to store types somehow.

# str_to_type = {
#     "int": int,
#     "float": float,
#     "bool": bool,
#     "str": str,
#     "bytes": bytes,
# }
# type_to_str = {t: s for s, t in str_to_type.items()}

# python_type_map["int"]
# # powinniśmy zachować stan w formacie lmdb.
