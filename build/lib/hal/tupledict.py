import collections

class TupleDict(collections.MutableMapping):
    """Keys are ordered tuples"""

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.nn = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys
    
    def get_nn(self, first_value):
        if first_value not in self.nn.keys():
            return None
        else:
            return self.nn[first_value]

    def __getitem__(self, key):
        return self.store[self.__keytransform__(key)]

    def __setitem__(self, key, value):

        new_key = self.__keytransform__(key)
        if new_key[0] not in self.nn.keys():
            self.nn[new_key[0]] = set([])
        if new_key[1] not in self.nn.keys():
            self.nn[new_key[1]] = set([])

        self.nn[new_key[0]].add(new_key[1])
        self.nn[new_key[1]].add(new_key[0])

        self.store[new_key] = value

    def __delitem__(self, key):
        k1, k2 = self.__keytransform__(key)
        self.nn[k1].discard(k2)
        self.nn[k2].discard(k1)
        del self.store[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)
    
    def __repr__(self):
        return str(self.store)

    def __keytransform__(self, key):
        if key[0] > key[1]:
            return (key[1], key[0])
        else:
            return key
