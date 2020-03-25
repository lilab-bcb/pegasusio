import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, vstack

from typing import List, Dict, Union, Set
import anndata

from pegasusio import UnimodalData



class MultimodalData:
    def __init__(self, data_dict: Union[Dict[str, UnimodalData], anndata.AnnData] = None, genome: str = None, exptype: str = None):
        if isinstance(data_dict, anndata.AnnData):
            self.from_anndata(data_dict, genome = genome, exptype = exptype)
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

    def get_exptype(self) -> str:
        """ Surrogate function for UnimodalData, return experiment tpye, can be either 'rna', 'citeseq', 'hashing', 'tcr', 'bcr', 'crispr' or 'atac'.
        """
        assert self._unidata is not None
        return self._unidata.get_exptype()



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


    def get_data(self, key: str) -> UnimodalData:
        if key not in self.data:
            raise ValueError("Key {} does not exist!".format(key))
        return self.data[key]


    def drop_data(self, key: str) -> UnimodalData:
        if key not in self.data:
            raise ValueError("Key {} does not exist!".format(key))
        return self.data.pop(key)


    def subset_data(self, data_subset: Set[str] = None, exptype_subset: Set[str] = None) -> None:
        """ Only keep data that are in data_subset and exptype_subset
        """
        if data_subset is not None:
            for key in self.list_data():
                if key not in data_subset:
                    self.data.pop(key)

        if exptype_subset is not None:
            for key in self.list_data():
                if self.data[key].uns["experiment_type"] not in exptype_subset:
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
        _check_reserved_keyword(black_list, "experiment_type")

        for key in self.data:
            self.data[key].scan_black_list(black_list)


    def from_anndata(self, data: anndata.AnnData, genome: str = None, exptype: str = None) -> None:
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
