"""
%load_ext autoreload
%autoreload 2
"""
import numba
import numpy as np
import tqdm

from prosit_timsTOF_2023_wrapper.main import Prosit2023TimsTofWrapper
from prosit_timsTOF_2023_wrapper.normalizations import normalize_to_sum


sequences = np.array(["PEPTIDE", "PEPTIDECPEPTIDE"])
amino_acid_cnts = np.array(list(map(len, sequences)))
collision_energies = np.array([30.0, 31.1])
charges = np.array([1, 2])

prosit = Prosit2023TimsTofWrapper()



import lmdb
import struct

fmt = "<iBf"  # : little endian
byte_seq = struct.pack(fmt, 42, 255, 3.14)
# Unpack data
a, b, c = struct.unpack(fmt, byte_seq)


# use with lmdb
key = struct.pack(fmt, 42, 255, 3.14)
value = np.uint32(124234)

# where to store data on disk
env = lmdb.open("/home/matteo/tmp/mydata.lmdb", map_size=2 * 1024 * 1024 * 1024)

with env.begin(write=True) as txn:
    txn.put(key, value.tobytes())

with env.begin() as txn:
    retrieved_value = txn.get(key)
    if retrieved_value:
        retrieved_value_decoded = np.frombuffer(retrieved_value, dtype=np.uint32)[0]
        print(retrieved_value_decoded)


zero = np.uint32(0).tobytes()
res = np.uint32(102323).tobytes() + zero + "PEPTIDECPEPTIDE".encode()
len(res)

var2type = dict(sequences=str, charges=int, collision_energies=float)



sequence = str(sequences[0])
charge = int(charges[0])
ce = float(collision_energies[0])

import msgpack

# could do it differnetly, this is simple.





types = (str,int,float)

def test(sequences, charges, collision_energies):
    results = []
    with tqdm.tqdm(total=len(sequences)) as pbar:
        for (
            fragment_intensities,
            max_ordinal,
            max_charge,
        ) in prosit.iter_predict_intensities(
            sequences=sequences,
            # amino_acid_cnts=amino_acid_cnts,
            charges=charges,
            collision_energies=collision_energies,
            pbar=pbar,
        ):
            results.append(fragment_intensities)


compress = True

from dataclasses import dataclass
from functools import wraps

@dataclass
class Cachemir:
    types: tuple
    lmdb_path    
    
    def wrap(self, foo):

        def wrapper(*arg,**kwargs)

        return wrapped


    def write_to

        

with env.begin(write=True) as txn:
    for inputs in zip(sequences, charges, collision_energies):
        assert len(inputs) == len(types)
        inputs_in_user_types = tuple(_type(_input) for _type, _input in zip(types, inputs))
        inputs_bytes = msgpack.packb(inputs_in_user_types)
        txn.put(inputs_bytes, value.tobytes())



# Data to compress
data = b"common words are common and repeat often"
sequence = "PEPTIDECPEPTIDE"
sequence_bits = sequence.encode()




# --- Compression with dictionary ---
compressor = zlib.compressobj(level=9, zdict=dictionary)
compressed_data = compressor.compress(data) + compressor.flush()

# --- Decompression with same dictionary ---
decompressor = zlib.decompressobj(zdict=dictionary)
decompressed_data = decompressor.decompress(compressed_data)

# Verify it works
assert data == decompressed_data
print("Decompressed:", decompressed_data)


# 
numpy_to_struct_format = {
    np.bool_:     '?',
    np.int8:      'b',
    np.uint8:     'B',
    np.int16:     'h',
    np.uint16:    'H',
    np.int32:     'i',
    np.uint32:    'I',
    np.int64:     'q',
    np.uint64:    'Q',
    np.float32:   'f',
    np.float64:   'd',
    np.dtype('S1'): 'c',   # Single character
    # Note: For np.dtype('S10'), format is '10s' — use a function to handle variable-length strings
}
def create_format_string(types: dict[type,sr]):
    unparsed = {}
    for _name, _type in types.items():
        try:
            numpy_to_struct_format[_type]
    
# let's say numpy 

