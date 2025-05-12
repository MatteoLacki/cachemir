from cachemir.main import Cachemir

cachemir = Cachemir.open(folder)

with cachemir.open(mode="r") as txn:
    print(len(txn))
    print(len(txn.datasets))


def T1():
    print("T1")


def T2():
    print("T2")
