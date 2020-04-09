import gc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, vstack
from typing import List, Dict, Union, Set, Tuple

import logging
logger = logging.getLogger(__name__)

import anndata

from pegasusio import UnimodalData
from .views import INDEX, UnimodalDataView



class MultimodalData:
    def __init__(self, data_dict: Union[Dict[str, UnimodalData], anndata.AnnData] = None, genome: str = None, modality: str = None):
        if isinstance(data_dict, anndata.AnnData):
            self.from_anndata(data_dict, genome = genome, modality = modality)
            return None

        self.data = data_dict if data_dict is not None else dict()
        self._selected = self._unidata = None


    def __repr__(self) -> str:
        repr_str = "MultimodalData object with {} UnimodalData: {}".format(len(self.data), str(list(self.data))[1:-1])
        if self._selected is not None:
            repr_str += "\n    It currently binds to UnimodalData object {}\n\n".format(self._selected)
            repr_str += self._unidata.__repr__()
        else:
            repr_str += "\n    It currently binds to no UnimodalData object"

        return repr_str


    def update(self, data: "MultimodalData") -> None:
        for key in data.data:
            self.data[key] = data.data[key]


    @property
    def obs(self) -> Union[pd.DataFrame, None]:
        return self._unidata.obs if self._unidata is not None else None

    @obs.setter
    def obs(self, obs: pd.DataFrame):
        assert self._unidata is not None
        self._unidata.obs = obs

    @property
    def obs_names(self) -> Union[pd.Index, None]:
        return self._unidata.obs_names if self._unidata is not None else None

    @obs_names.setter
    def obs_names(self, obs_names: pd.Index):
        assert self._unidata is not None
        self._unidata.obs_names = obs_names

    @property
    def var(self) -> Union[pd.DataFrame, None]:
        return self._unidata.var if self._unidata is not None else None

    @var.setter
    def var(self, var: pd.DataFrame):
        assert self._unidata is not None
        self._unidata.var = var

    @property
    def var_names(self) -> Union[pd.Index, None]:
        return self._unidata.var_names if self._unidata is not None else None

    @var_names.setter
    def var_names(self, var_names: pd.Index):
        assert self._unidata is not None
        self._unidata.var_names = var_names

    @property
    def X(self) -> Union[csr_matrix, None]:
        return self._unidata.X if self._unidata is not None else None

    @X.setter
    def X(self, X: csr_matrix):
        assert self._unidata is not None
        self._unidata.X = X

    @property
    def obsm(self) -> Union[Dict[str, np.ndarray], None]:
        return self._unidata.obsm if self._unidata is not None else None

    @obsm.setter
    def obsm(self, obsm: Dict[str, np.ndarray]):
        assert self._unidata is not None
        self._unidata.obsm = obsm

    @property
    def varm(self) -> Union[Dict[str, np.ndarray], None]:
        return self._unidata.varm if self._unidata is not None else None

    @varm.setter
    def varm(self, varm: Dict[str, np.ndarray]):
        assert self._unidata is not None
        self._unidata.varm = varm

    @property
    def uns(self) -> Union[dict, None]:
        return self._unidata.uns if self._unidata is not None else None

    @uns.setter
    def uns(self, uns: dict):
        assert self._unidata is not None
        self._unidata.uns = uns

    @property
    def shape(self) -> Tuple[int, int]:
        return self._unidata.shape if self._unidata is not None else None
    
    @shape.setter
    def shape(self, _shape: Tuple[int, int]):
        assert self._unidata is not None
        self._unidata.shape = _shape

    def as_float(self, matkey: str = None) -> None:
        """ Surrogate function to convert matrix to float """
        assert self._unidata is not None
        self._unidata.as_float(matkey)


    def list_keys(self, key_type: str = "matrix") -> List[str]:
        """ Surrogate function for UnimodalData, return available keys in metadata, key_type = barcode, feature, matrix, other
        """
        assert self._unidata is not None
        return self._unidata.list_keys(key_type)

    def select_matrix(self, key: str) -> None:
        """ Surrogate function for UnimodalData, select a matrix
        """
        assert self._unidata is not None
        self._unidata.select_matrix(key)

    def get_matrix(self, key: str) -> csr_matrix:
        """ Surrogate function for UnimodalData, return a matrix indexed by key
        """
        assert self._unidata is not None
        return self._unidata.get_matrix(key)

    def get_modality(self) -> str:
        """ Surrogate function for UnimodalData, return modality, can be either 'rna', 'citeseq', 'hashing', 'tcr', 'bcr', 'crispr' or 'atac'.
        """
        assert self._unidata is not None
        return self._unidata.get_modality()

    def _inplace_subset_obs(self, index: List[bool]) -> None:
        """ Surrogate function for UnimodalData, subset barcode_metadata inplace """
        assert self._unidata is not None
        self._unidata._inplace_subset_obs(index)

    def _inplace_subset_var(self, index: List[bool]) -> None:
        """ Surrogate function for UnimodalData, subset feature_metadata inplace """
        assert self._unidata is not None
        self._unidata._inplace_subset_var(index)

    def __getitem__(self, index: INDEX) -> UnimodalDataView:
        """ Surrogate function for UnimodalData, [] operation """
        assert self._unidata is not None
        return self._unidata[index]


    def list_data(self) -> List[str]:
        return list(self.data)


    def add_data(self, key: str, uni_data: UnimodalData) -> None:
        """ Add data, if _selected is not set, set as the first added dataset
        """
        if key in self.data:
            raise ValueError("Key {} already exists!".format(key))
        self.data[key] = uni_data
        if self._selected is None:
            self._selected = key
            self._unidata = uni_data


    def select_data(self, key: str) -> None:
        if key not in self.data:
            raise ValueError("Key {} does not exist!".format(key))
        self._selected = key
        self._unidata = self.data[self._selected]


    def current_data(self) -> str:
        return self._selected


    def get_data(self, key: str = None, modality: str = None) -> Union[UnimodalData, List[UnimodalData]]:
        if key is not None:
            if key not in self.data:
                raise ValueError("Key {} does not exist!".format(key))
            return self.data[key]

        if modality is None:
            raise ValueError("Either key or modality needs to be set!")

        data_arr = []
        for key in self.data:
            if self.data[key].uns["modality"] == modality:
                data_arr.append(self.data[key])
        if len(data_arr) == 0:
            raise ValueError("No UnimodalData with modality {}!".format(modality))

        return data_arr[0] if len(data_arr) == 1 else data_arr


    def drop_data(self, key: str) -> UnimodalData:
        if key not in self.data:
            raise ValueError("Key {} does not exist!".format(key))
        return self.data.pop(key)


    def concat_data(self, modality: str = "rna"):
        """ Used for raw data, Ignore multiarrays and only consider one matrix per unidata """
        genomes = []
        unidata_arr = []

        for key in list(self.data):
            genomes.append(key)
            unidata_arr.append(self.data.pop(key))

        if len(genomes) == 1:
            unikey = genomes[0]
            self.data[unikey] = unidata_arr[0]
        else:
            unikey = ",".join(genomes)
            feature_metadata = pd.concat([unidata.feature_metadata for unidata in unidata_arr], axis = 0)
            feature_metadata.reset_index(inplace = True)
            feature_metadata.fillna(value = "N/A", inplace = True)
            X = hstack([unidata.matrices["X"] for unidata in unidata_arr], format = "csr")
            self.data[unikey] = UnimodalData(unidata_arr[0].barcode_metadata, feature_metadata, {"X": X}, metadata = {"genome": unikey, "modality": "rna"})
            del unidata_arr
            gc.collect()

        self._selected = unikey
        self._unidata = self.data[unikey]


    def subset_data(self, data_subset: Set[str] = None, modality_subset: Set[str] = None) -> None:
        """ Only keep data that are in data_subset and modality_subset
        """
        if data_subset is not None:
            for key in self.list_data():
                if key not in data_subset:
                    self.data.pop(key)

        if modality_subset is not None:
            for key in self.list_data():
                if self.data[key].uns["modality"] not in modality_subset:
                    self.data.pop(key)


    def scan_black_list(self, black_list: Set[str] = None):
        """ Remove unwanted keys in the black list
            Note: black_list might be changed.
        """
        if black_list is None:
            return None

        def _check_reserved_keyword(black_list: Set[str], keyword: str):
            if keyword in black_list:
                logger.warning("Removed reserved keyword '{}' from black list.".format(keyword))
                black_list.remove(keyword)

        _check_reserved_keyword(black_list, "genome")
        _check_reserved_keyword(black_list, "modality")

        for key in self.data:
            self.data[key].scan_black_list(black_list)


    def from_anndata(self, data: anndata.AnnData, genome: str = None, modality: str = None) -> None:
        """ Initialize from an anndata object
        """
        unidata = UnimodalData(data)
        key = unidata.uns["genome"]
        self.data = {key: unidata}
        self._selected = key
        self._unidata = unidata


    def to_anndata(self) -> anndata.AnnData:
        """ Convert current data to an anndata object
        """
        if self._unidata is None:
            raise ValueError("Please first select a unimodal data to convert!")
        return self._unidata.to_anndata()


    def copy(self) -> "MultimodalData":
        from copy import deepcopy
        new_data = MultimodalData(deepcopy(self.data))
        new_data._selected = self._selected
        if new_data._selected is not None:
            new_data._unidata = new_data.data[new_data._selected]
        return new_data


    def __deepcopy__(self, memo):
        return self.copy()


