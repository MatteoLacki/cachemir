"""
Basic implementation of file-system based memoization scheme.

TODO: 
* each dataset as separate file
* each dataset of the same size in a subfolder grouping them together?
* perhaps context managers should return some classes with methods like len and some fields like types?


"""
from future import __annotations__

import fcntl
import json

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
    map_size: int=2**30
    max_dbs: int=1

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


SHAPE = int|tuple[int,...]


@dataclass
class DataSet:
    path: Path
    dtype: dtype
    shape: SHAPE
    name: str=""
    verbose: bool=False,
    _writer_lock_type: int = fcntl.LOCK_EX,
    _reader_lock_type: int = fcntl.LOCK_SH,
    _reader_modes: tuple[str, ...] = ("r", "rb"),

    @classmethod
    def new(cls, path, dtype, shape, name, memmap_kwargs={}, **kwargs) -> DataSet:
        np.memmap(
            path,
            mode="w",
            dtype=self.dtype,
            **memmap_kwargs,
        )
        return DataSet(path, dtype, shape, name, **kwargs)

    @contextmanager# or simply __enter__?
    def open(self, mode="r"):
        file_mode = mode if mode in ("r", "r+", "w+", "c") else "r+"
        file_handler = open(path, file_mode)
        try:
            fcntl.flock(
                file_handler,
                _reader_lock_type if mode in _reader_modes else _writer_lock_type,
            )
            array = np.load(file_handler, mmap_mode=mode)
            yield array 
        finally:
            if not mode in self._reader_modes:
                flush_start = time()
                array.flush()
                if verbose:
                    print(f"Flushed `{self.name}` in `{time()-flush_start} sec.`")
            del array
            fcntl.flock(file_handler, fcntl.LOCK_UN)
        

class Transaction:
    def __init__(self, index, datasets: DotDict):
        self.index = index
        self.datasets = datasets



class Cachemir:
    @classmethod
    def new(
        cls,
        folder: str|Path,
        scheme: dict["str",type],
        shape: SHAPE=0,
        index_kwargs: dict[str,str|int]={},
    ) -> Cachemir:
        """Create new dataset folder.

        Arguments:
            folder (str|Path): 
        Returns:
            Cachemir: A new instance."""
        folder = Path(folder)
        for subfolder in ("index", "data"):
            (folder / subfolder).mkdir(parents=True)

        with open(folder / "scheme.json", "w") as f:
            json.dump(np.dtype(list(scheme.items())).descr, f)

        index = Index(path=folder / "index", **index_kwargs)# save **index_kwargs in index. not scheme not: then not human readable.
        datasets = DotDict(
            col: DataSet.new(folder / f"data/{col}.npy", dtype, shape, col)
            for col, dtype in scheme.items()
        )
        return cls(folder, index, datasets)

    @classmethod
    def open(cls, folder: str|Path) -> Cachemir:
        """Open EXISTING dataset folder."""
        folder = Path(folder)

        with open(folder / "scheme.json", "r") as f:
            scheme = json.load(f)

        for col, dtype in scheme.items():
            assert (self.folder / f"data/{col}.npy").exists()

    def __init__(self, index: Index, datasets: DotDict[str, DataSet]):
        self.index = index
        self.datasets = datasets

    def __enter__(self) -> Transaction:
        return Transaction(index=index, datasets=datasets)

    def __exit__(self, exc_type, exc_value, traceback):
        pass


