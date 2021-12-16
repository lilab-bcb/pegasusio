from collections.abc import MutableMapping



class DataDict(MutableMapping):
    def __init__(self, initial_data: dict = None):
        self.mapping = initial_data if initial_data is not None else dict()

        self.init_keys = set(self.mapping) # record keys when data is loaded
        self.modified = set() # record keys that are modified
        self.deleted = None # will record keys need to be removed from file, only calculated in is_dirty
        self._tmp = None # temporary value to cache new init_keys

        self._overwrite = False # if overwrite happened

    def __getitem__(self, key):
        return self.mapping[key]

    def __setitem__(self, key, value):
        self.mapping[key] = value
        self.modified.add(key)

    def __delitem__(self, key):
        del self.mapping[key]
        self.modified.discard(key)

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)

    def __repr__(self):
        return f"DataDict object with keys: {str(list(self.mapping))[1:-1]}"

    def is_dirty(self):
        """ DataDict should not be updated after calling is_dirty and before calling clear_dirty """
        if self.deleted is None:
            self._tmp = set(self.mapping)
            self.deleted = self.init_keys - self._tmp
        return self._overwrite or len(self.deleted) > 0 or len(self.modified) > 0

    def clear_dirty(self):
        self.init_keys = self._tmp
        self.modified.clear()
        self.deleted = self._tmp = None
        self._overwrite = False

    def overwrite(self, mapping: dict):
        """ Completely overwrite the DataDict """
        self._overwrite = True
        self.mapping = mapping
        self.init_keys.clear()
        self.modified.clear()
        self.deleted = self._tmp = None



class MultiDataDict(MutableMapping):
    def __init__(self):
        self.mapping = dict()

        self.init_keys = set() # record keys when data is loaded
        self.accessed = set() # record keywords for accessed UnimodalData
        self.modified = set() # record keys that are modified
        self.deleted = self._tmp = None

    def __getitem__(self, key):
        value = self.mapping[key]
        self.accessed.add(key)
        return value

    def __setitem__(self, key, value):
        self.mapping[key] = value
        self.accessed.add(key)
        self.modified.add(key)

    def __delitem__(self, key):
        del self.mapping[key]
        self.accessed.discard(key)
        self.modified.discard(key)

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)

    def __repr__(self):
        return f"MultiDataDict object with keys: {str(list(self.mapping))[1:-1]}"

    def kick_start(self, data_key):
        """ Since UnimodalData are added sequentially when loading the object, kick_start marks the finish of loading: we begin to track changes now """
        self.init_keys = set(self.mapping)
        self.accessed.clear()
        self.accessed.add(data_key)
        self.modified.clear()

    def is_dirty(self):
        self._tmp = set(self.mapping)
        self.deleted = self.init_keys - self._tmp

        if len(self.deleted) > 0 or len(self.modified) > 0:
            return True

        for key in self.accessed:
            if self.mapping[key]._is_dirty():
                return True

        return False

    def clear_dirty(self, data_key):
        for key in self.accessed:
            self.mapping[key]._clear_dirty()
        self.init_keys = self._tmp
        self.accessed.clear()
        self.accessed.add(data_key)
        self.modified.clear()
        self.deleted = self._tmp = None
