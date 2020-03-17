import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, vstack

from typing import List, Dict, Union
import anndata

from pegasusio import UnimodalData



class MultimodalData:
    def __init__(self, data_dict: Dict[str, UnimodalData] = None):
        self.data = data_dict if data_dict is not None else dict()
        self._selected = self._unidata = None



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


    def restrain_keywords(self, keywords: str) -> None:
        """May load more data, this will restrain keys to the ones listed in keywords, which is a comma-separated list
        """
        if keywords is None:
            return None

        keywords = set(keywords.split(","))
        available = set(self.data)

        invalid_set = keywords - available
        if len(invalid_set) > 0:
            raise ValueError(
                "Keywords {} do not exist.".format(",".join(list(invalid_set)))
            )

        remove_set = available - keywords
        for keyword in remove_set:
            self.data.pop(keyword)
