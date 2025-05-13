"""
%load_ext autoreload
%autoreload 2
"""
import numpy as np

from cachemir.main import Cachemir
from cachemir.main import DataSet2
from pathlib import Path


folder = Path("/home/matteo/tmp/testlib")

# rm -rf /home/matteo/tmp/testlib
cachemir = Cachemir.new(folder, scheme=dict(intensity=np.float32), shape=100)
folder.exists()
cachemir = Cachemir(folder)
with cachemir.open(mode="r") as txn:
    print(len(txn))
    # print(len(txn.datasets))


# where to enlarge the dataset?
# how to do it?


def enlarge_file(file_path: str | Path, resize: int, datatype: str):
    assert resize >= 0
    with locked_open(file_path, "ab") as f:
        f.truncate(os.path.getsize(filename) + resize * np.dtype(datatype).itemsize)


from cachemir.main import DataSet

from pathlib import Path

dataset = DataSet(
    Path("/tmp/intensity.npy"),
    dtype=np.float32,
    name="intensity",
)

with dataset.open(mode="w") as arr:
    print(arr)

np.load(file, mmap_mode=mode)
# OK, shit. None of this will work and python is shit.

import fcntl
import numpy as np

from cachemir.main import DataSet2
from contextlib import contextmanager
from pathlib import Path
from time import time
# rm -rf /tmp/intensity.npy
path = Path("/tmp/intensity.npy")


# likely creation does not require this.
x = np.memmap(path, mode="w+", dtype=np.float32, shape=100)
x[:len(x)//2] = 1.0
x.flush()
del x



file = path.open(mode="rb")
fcntl.flock(
    file,
    fcntl.LOCK_EX,
)
y = np.memmap(path, dtype=np.float32)
y


r,rb -> r
w, wb -> w+
r+,rb+ -> r+

# resize should be another operation
from time import sleep

from mmapped_df import DatasetWriter



mode = "r"


@contextmanager
def opendataset(
    path,
    dtype,
    name,
    mode: str = "r",
    verbose: bool = False,
    **kwargs
):
    file = open(path, mode)
    match mode:
        case "r" | "rb":
            lock = fcntl.LOCK_SH
            np_mode = "r"
        case "w" | "wb":
            lock = fcntl.LOCK_EX
            np_mode = "w+"
        case "r+"| "rb+":
            lock = fcntl.LOCK_EX
            np_mode = "r+"
        case other:
            raise ValueError(f"invalid mode: `{mode}`")
    try:
        fcntl.flock(file,lock)
        array = np.memmap(file, dtype, np_mode, **kwargs)
        yield array
    except Exception as e:
        print(e)
    finally:
        if np_mode != "r":
            flush_start = time()
            array.flush()
            flush_runtime = time() - flush_start
            if verbose:
                print(f"Flushed `{name}` in `{flush_runtime} sec.`")
        del array
        fcntl.flock(file, fcntl.LOCK_UN)

with opendataset(path, np.float32, "intensity", mode='w') as arr:
    print(arr)
    # sleep(100)


with opendataset(path, np.float32, "intensity") as arr:
    print(arr)
