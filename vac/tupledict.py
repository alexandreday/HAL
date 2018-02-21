import collections

class TupleDict(collections.MutableMapping):
    """Keys are ordered tuples"""

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        return self.store[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.store[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store._)
    
    def __repr__(self):
        return str(self.store)

    def __keytransform__(self, key):
        if key[0] > key[1]:
            return (key[1], key[0])
        else:
            return key
