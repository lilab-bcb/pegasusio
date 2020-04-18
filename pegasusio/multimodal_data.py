import gc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, vstack
from typing import List, Dict, Union, Set, Tuple

import logging
logger = logging.getLogger(__name__)

import anndata

from pegasusio import UnimodalData, VDJData, CITESeqData, CytoData
from .views import INDEX, UnimodalDataView
from .vdj_data import VDJDataView


class MultimodalData:
    def __init__(self, data: Union[UnimodalData, List[UnimodalData], anndata.AnnData] = None, genome: str = None, modality: str = None):
        if isinstance(data, anndata.AnnData):
            self.from_anndata(data, genome = genome, modality = modality)
            return None

        self.data = dict()
        self._selected = self._unidata = None

        if data is not None:
            if isinstance(data, UnimodalData):
                self._selected = data.get_uid()
                assert self._selected is not None
                self._unidata = self.data[self._selected] = data
                return None

            for unidata in data:
                key = unidata.get_uid()
                assert key is not None
                self.data[key] = unidata

            self._selected = list(self.data)[0]
            self._unidata = self.data[self._selected]


    def __repr__(self) -> str:
        repr_str = f"MultimodalData object with {len(self.data)} UnimodalData: {str(list(self.data))[1:-1]}"
        if self._selected is not None:
            repr_str += f"\n    It currently binds to {self._unidata.__class__.__name__} object {self._selected}\n\n"
            repr_str += self._unidata.__repr__()
        else:
            repr_str += "\n    It currently binds to no UnimodalData object"
        return repr_str


    def update(self, data: "MultimodalData") -> None:
        for key in data.data:
            if key in self.data:
                raise ValueError(f"Detected duplicated key '{key}'")
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
    def X(self) -> Union[csr_matrix, np.ndarray, None]:
        return self._unidata.X if self._unidata is not None else None

    @X.setter
    def X(self, X: Union[csr_matrix, np.ndarray]):
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

    def get_matrix(self, key: str) -> Union[csr_matrix, np.ndarray]:
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

    def __getitem__(self, index: INDEX) -> Union[UnimodalDataView, VDJDataView]:
        """ Surrogate function for UnimodalData, [] operation """
        assert self._unidata is not None
        return self._unidata[index]


    def get_chain(self, chain: str) -> pd.DataFrame:
        """ Surrogate function for VDJData """
        assert self._unidata is not None and isinstance(self._unidata, VDJData)
        return self._unidata.get_chain(chain)

    def set_aside(self, params: List[str] = None) -> None:
        """ Surrogate function for CITESeqData and CytoData """
        assert self._unidata is not None and (isinstance(self._unidata, CITESeqData) or isinstance(self._unidata, CytoData))
        if params is None:
            self._unidata.set_aside()
        else:
            self._unidata.set_aside(params)
        
    def load_control_list(self, control_csv: str) -> None:
        """ Surrogate function for CITESeqData and CytoData """
        assert self._unidata is not None and (isinstance(self._unidata, CITESeqData) or isinstance(self._unidata, CytoData))
        self._unidata.load_control_list(control_csv)

    def log_transform(self) -> None:
        """ Surrogate function for CITESeqData """
        assert self._unidata is not None and isinstance(self._unidata, CITESeqData)
        self._unidata.log_transform()

    def arcsinh_transform(self, cofactor: float = 5.0, jitter = False, random_state = 0, select: bool = True) -> None:
        """ Surrogate function for CITESeqData and CytoData"""
        assert self._unidata is not None and (isinstance(self._unidata, CITESeqData) or isinstance(self._unidata, CytoData))
        self._unidata.arcsinh_transform(cofactor = cofactor, jitter = jitter, random_state = random_state, select = select)



    def list_data(self) -> List[str]:
        return list(self.data)


    def add_data(self, unidata: UnimodalData) -> None:
        """ Add data, if _selected is not set, set as the first added dataset
        """
        key = unidata.get_uid()
        assert key is not None
        if key in self.data:
            raise ValueError(f"Key '{key}' already exists!")
        self.data[key] = unidata
        if self._selected is None:
            self._selected = key
            self._unidata = unidata


    def select_data(self, key: str) -> None:
        if key not in self.data:
            raise ValueError(f"Key '{key}' does not exist!")
        self._selected = key
        self._unidata = self.data[self._selected]


    def current_data(self) -> str:
        return self._selected


    def get_data(self, key: str = None, genome: str = None, modality: str = None) -> Union[UnimodalData, List[UnimodalData]]:
        """ get UnimodalData or list of UnimodalData based on MultimodalData key or genome or modality
        """

        if key is not None:
            if key not in self.data:
                raise ValueError(f"Key '{key}' does not exist!")
            return self.data[key]

        data_arr = []

        if genome is not None:
            for key in self.data:
                unidata = self.data[key]
                if unidata.get_genome() == genome:
                    data_arr.append(unidata)
            
            if len(data_arr) == 0:
                raise ValueError(f"No UnimodalData with genome '{genome}'!")
        else:
            if modality is None:
                raise ValueError("Either key or genome or modality needs to be set!")

            for key in self.data:
                unidata = self.data[key]
                if unidata.get_modality() == modality:
                    data_arr.append(unidata)

                if len(data_arr) == 0:
                    raise ValueError(f"No UnimodalData with modality '{modality}'!")

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
            unidata = self.data.pop(key)
            assert unidata.get_modality() == modality
            genomes.append(unidata.get_genome())
            unidata_arr.append(unidata)

        unikey = None
        if len(genomes) == 1:
            unikey = unidata_arr[0].get_uid()
            self.data[unikey] = unidata_arr[0]
        else:
            genome = ",".join(genomes)
            feature_metadata = pd.concat([unidata.feature_metadata for unidata in unidata_arr], axis = 0)
            feature_metadata.reset_index(inplace = True)
            feature_metadata.fillna(value = "N/A", inplace = True)
            X = hstack([unidata.matrices["X"] for unidata in unidata_arr], format = "csr")
            unidata = UnimodalData(unidata_arr[0].barcode_metadata, feature_metadata, {"X": X}, {"genome": genome, "modality": "rna"})
            unikey = unidata.get_uid()
            self.data[unikey] = unidata
            del unidata_arr
            gc.collect()

        self._selected = unikey
        self._unidata = self.data[unikey]


    def subset_data(self, data_subset: Set[str] = None, genome_subset: Set[str] = None, modality_subset: Set[str] = None) -> None:
        """ Only keep data that are in data_subset or genome_subset or modality_subset
        """
        if data_subset is not None:
            for key in self.list_data():
                if key not in data_subset:
                    del self.data[key]
        elif genome_subset is not None:
            for key in self.list_data():
                if self.data[key].get_genome() not in genome_subset:
                    del self.data[key]
        elif modality_subset is not None:
            for key in self.list_data():
                if self.data[key].get_modality() not in modality_subset:
                    del self.data[key]


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
        key = unidata.get_uid()
        assert key is not None
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
