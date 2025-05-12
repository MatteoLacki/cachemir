"""
Basic implementation of file-system based memoization scheme.

TODO: 
* each dataset as separate file
* each dataset of the same size in a subfolder grouping them together?
* perhaps context managers should return some classes with methods like len and some fields like types?


"""
from __future__ import annotations

import fcntl
import json
import lmdb

from typing import Iterator

from contextlib import ExitStack
from contextlib import contextmanager
from dataclasses import dataclass
from massimo.data_structures import DotDict
from pathlib import Path
from time import time


@dataclass
class Index:
    """LMDB based index for the memory mapped files.

    Supposed to store indices and some non-grouped statistics.
    """

    path: Path
    map_size: int = 2**30
    max_dbs: int = 1

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


SHAPE = int | tuple[int, ...]


@dataclass
class DataSet:
    path: Path
    dtype: dtype
    name: str = ""

    @classmethod
    def new(cls, path, dtype, shape, name, memmap_kwargs={}, **kwargs) -> DataSet:
        np.memmap(
            path,
            mode="w",
            dtype=self.dtype,
            **memmap_kwargs,
        )
        return DataSet(path, dtype, shape, name, **kwargs)

    @contextmanager
    def open(
        self,
        mode: str = "r",
        verbose: bool = False,
    ):
        file = open(path, file_mode)
        try:
            fcntl.flock(file, fcntl.LOCK_SH if mode in ("r", "rb") else fcntl.LOCK_EX)
            array = np.load(file, mmap_mode=mode)
            yield array
        finally:
            if not mode in self._reader_modes:
                flush_start = time()
                array.flush()
                flush_runtime = time() - flush_start
                if verbose:
                    print(f"Flushed `{self.name}` in `{flush_runtime} sec.`")
            del array
            fcntl.flock(file, fcntl.LOCK_UN)


class DataTransaction:
    def __init__(self, index_txn, datasets: DotDict):
        self.index_txn = index_txn
        self.datasets = datasets

    def __len__(self) -> int:
        return self.index_stats()["entries"]


# TODO: merge __init__ and new cause it is essentially the same thing?
class Cachemir:
    def __init__(self, folder: str | Path) -> None:
        """Open EXISTING dataset folder."""
        self.folder = Path(folder)

        with open(self.folder / "scheme.json", "r") as f:
            scheme = json.load(f)
        for col, dtype in scheme.items():
            assert (self.folder / f"data/{col}.npy").exists()
        self.index = Index(path=folder / "index", **index_kwargs)
        self.datasets = DotDict(
            (
                (col, DataSet(folder / "data/{col}.npy", dtype, col))
                for col, dtype in scheme.items()
            )
        )

    @classmethod
    def new(
        cls,
        folder: str | Path,
        scheme: dict["str", type],
        shape: SHAPE = 0,
        index_kwargs: dict[str, str | int] = {},
    ) -> Cachemir:
        """Create new dataset folder.

        Arguments:
            folder (str|Path): Path to the DB.

        Returns:
            Cachemir: A new instance.
        """
        folder = Path(folder)
        for subfolder in ("index", "data"):
            (folder / subfolder).mkdir(parents=True)
        with open(folder / "scheme.json", "w") as f:
            json.dump(np.dtype(list(scheme.items())).descr, f)
        index = Index(path=folder / "index")  # **kwargs likely unnecessary: make sure
        datasets = DotDict(
            (
                (col, DataSet.new(folder / f"data/{col}.npy", dtype, shape, col))
                for col, dtype in scheme.items()
            )
        )
        return cls(folder)

    @contextmanager
    def open(self, mode: str = "r") -> Iterator[DataTransaction]:
        with ExitStack() as stack:
            yield DataTransaction(
                index=self.index.open(mode),
                datasets=DotDict(
                    ((col, dataset.open()) for col, dataset in self.datasets)
                ),
            )
