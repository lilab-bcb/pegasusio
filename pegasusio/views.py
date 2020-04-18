import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections.abc import MutableMapping
from typing import List, Dict, Union, Tuple
from pandas.api.types import is_list_like


INDEX1D = Union[pd.Index, List[str], List[bool], List[int], slice]
INDEX = Union[INDEX1D, Tuple[INDEX1D, INDEX1D]]


class MultiArrayView(MutableMapping):
    def __init__(self, multiarrays: Dict[str, np.ndarray], index: List[int]):
        self.parent = multiarrays
        self.index = index
        self.multiarrays = {}

    def __getitem__(self, key: Union[str, "Ellipsis"]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if key is Ellipsis:
            for key in self.parent:
                if key not in self.multiarrays:
                    self.multiarrays[key] = self.parent[key][self.index]
            return self.multiarrays

        if key not in self.multiarrays:
            if key in self.parent:
                self.multiarrays[key] = self.parent[key][self.index]
            else:
                raise ValueError(f"Key '{key}' does not exist!")
        return self.multiarrays[key]    

    def __setitem__(self, key: str, value: object):
        raise ValueError("Cannot set key for MultiArrayView object!")

    def __delitem__(self, key: str):
        raise ValueError("Cannot delete key for MultiArrayView object!")

    def __iter__(self):
        return iter(self.parent)

    def __len__(self):
        return len(self.parent)

    def __repr__(self) -> str:
        return f"View of multiarrays with keys: {str(list(self.parent))[1:-1]}."



def _parse_index(parent: Union["UnimodalData", "UnimodalDataView"], index: INDEX) -> Tuple[List[int], List[int]]:

    def _extract_indices_from_parent(parent: Union["UnimodalData", "UnimodalDataView"]) -> Tuple[pd.Index, pd.Index, List[int], List[int]]:
        if hasattr(parent, "barcode_index"):
            return parent.parent.obs_names, parent.parent.var_names, parent.barcode_index, parent.feature_index
        else:
            return parent.obs_names, parent.var_names, None, None

    def _check_index_type(index_1d, index_name: str) -> INDEX1D:
        if isinstance(index_1d, slice):
            if index_1d == slice(None):
                return None
            else:
                if not (isinstance(index_1d.start, int) or isinstance(index_1d.start, np.integer) or isinstance(index_1d.start, str) or (index_1d.start is None)):
                    raise ValueError(f"Invalid slice.start '{index_1d.start}'. slice.start must be either None, integer or string!")
                if not (isinstance(index_1d.stop, int) or isinstance(index_1d.stop, np.integer) or isinstance(index_1d.stop, str) or (index_1d.stop is None)):
                    raise ValueError(f"Invalid slice.stop '{index_1d.stop}'. slice.stop must be either None, integer or string!")
                if not (isinstance(index_1d.step, int) or isinstance(index_1d.step, np.integer) or (index_1d.step is None)):
                    raise ValueError(f"Invalid slice.step '{index_1d.step}'. slice.step must be either None, integer or string!")
                return index_1d
        elif isinstance(index_1d, pd.Index):
            return index_1d
        else:
            if not is_list_like(index_1d):
                index_1d = np.array([index_1d])
            index_1d = np.array(index_1d, copy = False)
            if index_1d.ndim != 1:
                raise ValueError(f"{index_name} index must be 1 dimension!")
            if index_1d.dtype.kind not in {'b', 'i', 'u', 'O', 'U'}:
                raise ValueError(f"Unknown {index_name} index dtype: {index_1d.dtype}!")
            return index_1d

    def _process_pd_index(base_idx: pd.Index, index_1d: pd.Index) -> List[int]:
        if index_1d.has_duplicates:
            index_1d = index_1d.drop_duplicates()
        indexer = base_idx.get_indexer(index_1d)
        indexer = indexer[indexer >= 0]
        return indexer

    def _parse_one_index(base_idx: pd.Index, index_1d: Union[INDEX1D, None], index_name: str, view_index: List[int] = None) -> List[int]:
        """ index_name: 'row' or 'column' """
        if view_index is not None:
            base_idx = base_idx[view_index]

        indexer = None

        if index_1d is None:
            indexer = range(base_idx.size)
        elif isinstance(index_1d, slice):
            step = 1 if index_1d.step is None else index_1d.step
            
            start = index_1d.start
            if isinstance(start, str):
                if start not in base_idx:
                    raise ValueError(f"Cannot locate slice.start '{start}' in {index_name} index!")
                start = base_idx.get_loc(start)
            elif start is None:
                start = 0
            else:
                if start < 0 or start >= base_idx.size:
                    raise ValueError(f"slice.start '{start}' is out of the boundary [0, {base_idx.size}) for {index_name} index!")

            stop = index_1d.stop
            if isinstance(stop, str):
                if stop not in base_idx:
                    raise ValueError(f"Cannot locate slice.stop '{stop}' in {index_name} index!")
                stop = base_idx.get_loc(stop) + np.sign(step) # if str , use [] instead of [)
            elif stop is None:
                stop = base_idx.size
            else:
                if stop < 0 or stop > base_idx.size:
                    raise ValueError(f"slice.stop '{stop}' is out of the boundary [0, {base_idx.size}] for {index_name} index!")

            indexer = range(start, stop, step)
        elif isinstance(index_1d, np.ndarray) and (index_1d.dtype.kind in {'b', 'i', 'u'}):
            assert index_1d.ndim == 1

            if index_1d.dtype.kind == 'b':
                if index_1d.size != base_idx.size:
                    raise ValueError(f"{index_name} index size does not match: actual size {base_idx.size}, input size {index_1d.size}!")
                indexer = np.where(index_1d)[0]
            elif index_1d.dtype.kind == 'i' or index_1d.dtype.kind == 'u':
                if np.any(index_1d < 0):
                    raise ValueError(f"Detect negative values in {index_name} index!")
                if np.any(index_1d >= base_idx.size):
                    raise ValueError(f"Detect values exceeding the largest valid position {base_idx.size - 1} in {index_name} index!")
                if np.unique(index_1d).size < index_1d.size:
                    raise ValueError(f"{index_name} index values are not unique!")
        else:
            if not isinstance(index_1d, pd.Index):
                assert isinstance(index_1d, np.ndarray) and index_1d.ndim == 1
                index_1d = pd.Index(index_1d)
            indexer = _process_pd_index(base_idx, index_1d)

        indexer = np.array(indexer, copy = False) if view_index is None else view_index[indexer]
        return indexer


    bidx = fidx = None
    base_bidx, base_fidx, view_bidx, view_fidx = _extract_indices_from_parent(parent)

    if isinstance(index, tuple):
        if len(index) > 2:
            raise ValueError(f"Index dimension {len(index)} exceeds 2!")
        elif len(index) == 1:
            bidx = _check_index_type(index[0], "row")
        else:
            bidx = _check_index_type(index[0], "row")
            fidx = _check_index_type(index[1], "column")
    else:
        bidx = _check_index_type(index, "row")

    return _parse_one_index(base_bidx, bidx, "row", view_index = view_bidx), _parse_one_index(base_fidx, fidx, "column", view_index = view_fidx)



class UnimodalDataView:
    def __init__(self, unidata: "UnimodalData", barcode_index: List[int], feature_index: List[int], cur_matrix: str, obj_name: str = "UnimodalData"):
        self.parent = unidata
        self.barcode_index = barcode_index
        self.feature_index = feature_index

        self.barcode_metadata = None
        self.feature_metadata = None
        self.matrices = {}
        self.barcode_multiarrays = MultiArrayView(unidata.barcode_multiarrays, barcode_index)
        self.feature_multiarrays = MultiArrayView(unidata.feature_multiarrays, feature_index)

        self.metadata = {}
        for key, value in unidata.metadata.items():
            # For views, only copy string objects
            if isinstance(value, str):
                self.metadata[key] = value

        self._cur_matrix = cur_matrix
        self._shape = (self.barcode_index.size, self.feature_index.size)
        self._obj_name = obj_name

    def __repr__(self, repr_dict: Dict[str, str] = None) -> str:
        repr_str = f"View of {self._obj_name} object with n_obs x n_vars = {self._shape[0]} x {self._shape[1]}"
        repr_str += f"\n    Genome: {self.parent.get_genome()}; Modality: {self.parent.get_modality()}"
        repr_str += f"\n    It contains {len(self.parent.matrices)} matrices: {str(list(self.parent.matrices))[1:-1]}"
        repr_str += f"\n    It currently binds to matrix '{self._cur_matrix}' as X\n" if len(self.parent.matrices) > 0 else "\n    It currently binds to no matrix\n"
        for key in ["obs", "var", "obsm", "varm"]:
            str_out = repr_dict[key] if (repr_dict is not None) and (key in repr_dict) else str(list(getattr(self.parent, key).keys()))[1:-1]
            repr_str += f"\n    {key}: {str_out}"
        repr_str += f"\n    uns: {str(list(self.metadata))[1:-1]}"
        return repr_str

    @property
    def obs(self) -> pd.DataFrame:
        if self.barcode_metadata is None:
            self.barcode_metadata = self.parent.barcode_metadata.iloc[self.barcode_index].copy(deep = False)
        return self.barcode_metadata

    @obs.setter
    def obs(self, obs: pd.DataFrame):
        raise ValueError("Cannot set obs for UnimodalDataView object!")

    @property
    def obs_names(self) -> pd.Index:
        return self.obs.index

    @obs_names.setter
    def obs_names(self, obs_names: pd.Index):
        raise ValueError("Cannot set obs_names for UnimodalDataView object!")

    @property
    def var(self) -> pd.DataFrame:
        if self.feature_metadata is None:
            self.feature_metadata = self.parent.feature_metadata.iloc[self.feature_index].copy(deep = False)
        return self.feature_metadata

    @var.setter
    def var(self, var: pd.DataFrame):
        raise ValueError("Cannot set var for UnimodalDataView object!")

    @property
    def var_names(self) -> pd.Index:
        return self.var.index

    @var_names.setter
    def var_names(self, var_names: pd.Index):
        raise ValueError("Cannot set var_names for UnimodalDataView object!")

    @property
    def X(self) -> Union[csr_matrix, None]:
        if self._cur_matrix not in self.matrices:
            X = self.parent.matrices.get(self._cur_matrix, None)
            if X is not None:
                self.matrices[self._cur_matrix] = X[self.barcode_index][:, self.feature_index] if self.barcode_index.size <= self.feature_index.size else X[:, self.feature_index][self.barcode_index]
        return self.matrices.get(self._cur_matrix, None)

    @X.setter
    def X(self, X: csr_matrix):
        raise ValueError("Cannot set X for UnimodalDataView object!")

    @property
    def obsm(self) -> Dict[str, np.ndarray]:
        return self.barcode_multiarrays

    @obsm.setter
    def obsm(self, obsm: Dict[str, np.ndarray]):
        raise ValueError("Cannot set obsm for UnimodalDataView object!")

    @property
    def varm(self) -> Dict[str, np.ndarray]:
        return self.feature_multiarrays

    @varm.setter
    def varm(self, varm: Dict[str, np.ndarray]):
        raise ValueError("Cannot set varm for UnimodalDataView object!")

    @property
    def uns(self) -> dict:
        return self.metadata

    @uns.setter
    def uns(self, uns: dict):
        raise ValueError("Cannot set uns for UnimodalDataView object!")

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape
    
    @shape.setter
    def shape(self, _shape: Tuple[int, int]):
        raise ValueError("Cannot set shape for UnimodalDataView object!")

    def select_matrix(self, key: str) -> None:
        if key not in self.parent.matrices:
            raise ValueError(f"Matrix key '{key}' does not exist!")
        self._cur_matrix = key

    def __getitem__(self, index: INDEX) -> "UnimodalDataView":
        barcode_index, feature_index = _parse_index(self, index)
        return UnimodalDataView(self.parent, barcode_index, feature_index, self._cur_matrix, obj_name = self._obj_name)

    def _copy_matrices(self) -> Dict[str, csr_matrix]:
        for key in self.parent.matrices:
            if key not in self.matrices:
                X = self.parent.matrices[key]
                self.matrices[key] = X[self.barcode_index][:, self.feature_index] if self.barcode_index.size <= self.feature_index.size else X[:, self.feature_index][self.barcode_index]
        return self.matrices

    def copy(self, deep: bool = True) -> "UnimodalData":
        """ If not deep, copy shallowly, which means that if contents of obsm, varm and matrices of the copied object is modified, the view object is also modified """
        return self.parent._copy_view(self, deep)
