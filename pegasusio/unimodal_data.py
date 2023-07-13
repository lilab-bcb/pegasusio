import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections.abc import MutableMapping
from copy import deepcopy
from natsort import natsorted
from typing import List, Dict, Union, Set, Tuple, Optional

import logging
logger = logging.getLogger(__name__)

import anndata

from pegasusio import run_gc
from pegasusio import modalities
from .views import INDEX, _parse_index, UnimodalDataView
from .datadict import DataDict



def _set_genome(metadata: dict, genome: str):
    if genome is not None:
        metadata["genome"] = genome
    elif "genome" not in metadata:
        metadata["genome"] = "unknown"
    elif isinstance(metadata["genome"], np.ndarray):
        assert metadata["genome"].ndim == 1
        metadata["genome"] = metadata["genome"][0]

def _set_modality(metadata: dict, modality: str):
    if modality is not None:
        metadata["modality"] = modality
    elif "modality" not in metadata:
        metadata["modality"] = "rna"

def _set_uid(metadata: dict, uid: str):
    if uid is not None:
        metadata["uid"] = uid
    else:
        metadata["uid"] = f"{metadata['genome']}-{metadata['modality']}"


class UnimodalData:
    def __init__(
        self,
        barcode_metadata: Optional[Union[dict, pd.DataFrame, anndata.AnnData]] = None,
        feature_metadata: Optional[Union[dict, pd.DataFrame]] = None,
        matrices: Optional[Dict[str, csr_matrix]] = None,
        metadata: Optional[dict] = None,
        barcode_multiarrays: Optional[Dict[str, np.ndarray]] = None,
        feature_multiarrays: Optional[Dict[str, np.ndarray]] = None,
        barcode_multigraphs: Optional[Dict[str, csr_matrix]] = None,
        feature_multigraphs: Optional[Dict[str, csr_matrix]] = None,
        cur_matrix: str = None,
        genome: Optional[str] = None,
        modality: Optional[str] = None,
        uid: Optional[str] = None,
    ) -> None:
        """ Note that metadata, barcode_mutiarrays, feature_multiarrays, barcode_multigraphs, feature_multigraphs can be modified.
        """
        if isinstance(barcode_metadata, anndata.AnnData):
            self.from_anndata(barcode_metadata, genome = genome, modality = modality, uid = uid)
            return None

        if barcode_metadata is None:
            barcode_metadata = pd.DataFrame()
        if feature_metadata is None:
            feature_metadata = pd.DataFrame()
        if metadata is None:
            metadata = dict()
        if barcode_multiarrays is None:
            barcode_multiarrays = dict()
        if feature_multiarrays is None:
            feature_multiarrays = dict()
        if barcode_multigraphs is None:
            barcode_multigraphs = dict()
        if feature_multigraphs is None:
            feature_multigraphs = dict()

        _set_genome(metadata, genome)
        _set_modality(metadata, modality)
        _set_uid(metadata, uid)

        self.barcode_metadata = barcode_metadata # barcode metadata
        self.feature_metadata = feature_metadata # feature metadata

        if len(self.barcode_metadata) > 0:
            if isinstance(self.barcode_metadata, MutableMapping):
                if "barcodekey" not in self.barcode_metadata:
                    raise ValueError("Cannot locate barcode index barcodekey!")
                barcodekey = self.barcode_metadata.pop("barcodekey")
                self.barcode_metadata = pd.DataFrame(data = self.barcode_metadata, index = pd.Index(barcodekey, name = "barcodekey"))
            else:
                if not isinstance(self.barcode_metadata, pd.DataFrame):
                    raise ValueError(f"Unknown barcode_metadata type: {type(self.barcode_metadata)}!")
                if "barcodekey" in self.barcode_metadata:
                    self.barcode_metadata.set_index("barcodekey", inplace=True)
                elif self.barcode_metadata.index.name != "barcodekey":
                    raise ValueError("Cannot locate barcode index barcodekey!")

        if len(self.feature_metadata) > 0:
            if isinstance(self.feature_metadata, MutableMapping):
                self.feature_metadata = pd.DataFrame(self.feature_metadata)
            if not isinstance(self.feature_metadata, pd.DataFrame):
                raise ValueError(f"Unknown feature_metadata type: {type(self.feature_metadata)}!")
            if "featurekey" in self.feature_metadata:
                self._polish_featurekey(self.feature_metadata, metadata["genome"])
                self.feature_metadata.set_index("featurekey", inplace=True)
            elif self.feature_metadata.index.name != "featurekey":
                raise ValueError("Cannot locate feature index featurekey!")

        self._update_shape()


        self.matrices = DataDict(matrices) # a dictionary of scipy csr matrix

        for key, mat in self.matrices.items():
            if mat.shape[0] != self._shape[0]:
                raise ValueError(
                    f"Wrong number of barcodes : matrix '{key}' has {mat.shape[0]} barcodes, barcodes file has {self._shape[0]} barcodes."
                )
            if mat.shape[1] != self._shape[1]:
                raise ValueError(
                    f"Wrong number of features : matrix '{key}' has {mat.shape[1]} features, features file has {self._shape[1]} features."
                )

        if len(self.matrices) == 0:
            self._cur_matrix = ""
        else:
            if cur_matrix == None or len(list(self.matrices.keys())) == 1:
                cur_matrix = list(self.matrices.keys())[0]
            elif cur_matrix not in self.matrices.keys():
                raise ValueError("Cannot find the default count matrix to bind to. Please set 'cur_matrix' argument in UnimodalData constructor!")

            self._cur_matrix = cur_matrix # cur_matrix

        # For backword compatibility, check metadata and move arrays and graphs to multiarrays/multigraphs
        for key in list(metadata.keys()):
            value = metadata[key]
            if isinstance(value, np.ndarray) and value.ndim == 2:
                if value.shape[0] == self._shape[0]: # put in barcode_multiarrays
                    barcode_multiarrays[key] = metadata.pop(key)
                elif value.shape[0] == self._shape[1]: # put in feature_multiarrays
                    feature_multiarrays[key] = metadata.pop(key)
            elif isinstance(value, csr_matrix) and value.shape[0] == value.shape[1]: # is a graph
                if value.shape[0] == self._shape[0]: # put in barcode_multigraphs
                    barcode_multigraphs[key] = metadata.pop(key)
                elif value.shape[0] == self._shape[1]: # put in feature_multigraphs
                    feature_multigraphs[key] = metadata.pop(key)

        self.metadata = DataDict(metadata) # other metadata, a dictionary

        self.barcode_multiarrays = DataDict(barcode_multiarrays)
        self.feature_multiarrays = DataDict(feature_multiarrays)
        self.barcode_multigraphs = DataDict(barcode_multigraphs)
        self.feature_multigraphs = DataDict(feature_multigraphs)


    def _gen_repr_str_for_attrs(self, key: str) -> str:
        if key == "obs" or key == "obsm":
            attr_str = ""
            for attr in getattr(self, key):
                attr_type = self.get_attr_type(attr)
                attr_str += f"'{attr}', " if attr_type is None else f"'{attr}'({attr_type}), "
            return attr_str[:-2]
        else:
            return str(list(getattr(self, key).keys()))[1:-1]

    def __repr__(self) -> str:
        repr_str = f"{self.__class__.__name__} object with n_obs x n_vars = {self.barcode_metadata.shape[0]} x {self.feature_metadata.shape[0]}"
        repr_str += f"\n    UID: {self.get_uid()}; Genome: {self.get_genome()}; Modality: {self.get_modality()}"
        mat_word = 'matrices' if len(self.matrices) > 1 else 'matrix'
        repr_str += f"\n    It contains {len(self.matrices)} {mat_word}: {str(list(self.matrices))[1:-1]}"
        repr_str += f"\n    It currently binds to matrix '{self._cur_matrix}' as X\n" if len(self.matrices) > 0 else "\n    It currently binds to no matrix\n"
        for key in ["obs", "var", "obsm", "varm", "obsp", "varp", "uns"]:
            fstr = self._gen_repr_str_for_attrs(key)
            if fstr != '':
                repr_str += f"\n    {key}: {fstr}"
        return repr_str


    def _update_shape(self) -> None:
        self._shape = (self.barcode_metadata.shape[0], self.feature_metadata.shape[0]) # shape


    def _is_dirty(self) -> bool:
        """ Check if any field is modified.
        """
        return self.matrices.is_dirty() or self.metadata.is_dirty() or self.barcode_multiarrays.is_dirty() or self.feature_multiarrays.is_dirty() or self.barcode_multigraphs.is_dirty() or self.feature_multigraphs.is_dirty()

    def _clear_dirty(self) -> None:
        """ Clear all dirty sets
        """
        self.matrices.clear_dirty()
        self.barcode_multiarrays.clear_dirty()
        self.feature_multiarrays.clear_dirty()
        self.barcode_multigraphs.clear_dirty()
        self.feature_multigraphs.clear_dirty()
        self.metadata.clear_dirty()

    @property
    def obs(self) -> pd.DataFrame:
        return self.barcode_metadata

    @obs.setter
    def obs(self, obs: pd.DataFrame):
        assert obs.shape[0] == 0 or obs.index.name == "barcodekey"
        self.barcode_metadata = obs
        self._update_shape()

    @property
    def obs_names(self) -> pd.Index:
        return self.barcode_metadata.index

    @obs_names.setter
    def obs_names(self, obs_names: pd.Index):
        self.barcode_metadata.index = obs_names
        self.barcode_metadata.index.name = "barcodekey"

    @property
    def var(self) -> pd.DataFrame:
        return self.feature_metadata

    @var.setter
    def var(self, var: pd.DataFrame):
        assert var.shape[0] == 0 or var.index.name == "featurekey"
        self.feature_metadata = var
        self._update_shape()

    @property
    def var_names(self) -> pd.Index:
        return self.feature_metadata.index

    @var_names.setter
    def var_names(self, var_names: pd.Index):
        self.feature_metadata.index = var_names
        self.feature_metadata.index.name = "featurekey"

    @property
    def X(self) -> Union[csr_matrix, None]:
        return self.matrices.get(self._cur_matrix, None)

    @X.setter
    def X(self, X: csr_matrix):
        self.matrices[self._cur_matrix] = X

    @property
    def obsm(self) -> Dict[str, np.ndarray]:
        return self.barcode_multiarrays

    @obsm.setter
    def obsm(self, obsm: Dict[str, np.ndarray]):
        self.barcode_multiarrays.overwrite(obsm)

    @property
    def varm(self) -> Dict[str, np.ndarray]:
        return self.feature_multiarrays

    @varm.setter
    def varm(self, varm: Dict[str, np.ndarray]):
        self.feature_multiarrays.overwrite(varm)

    @property
    def obsp(self) -> Dict[str, csr_matrix]:
        return self.barcode_multigraphs

    @obsp.setter
    def obsp(self, obsp: Dict[str, csr_matrix]):
        self.barcode_multigraphs.overwrite(obsp)

    @property
    def varp(self) -> Dict[str, csr_matrix]:
        return self.feature_multigraphs

    @varp.setter
    def varp(self, varp: Dict[str, csr_matrix]):
        self.feature_multigraphs.overwrite(varp)

    @property
    def uns(self) -> DataDict:
        return self.metadata

    @uns.setter
    def uns(self, uns: DataDict):
        self.metadata.overwrite(uns)

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @shape.setter
    def shape(self, _shape: Tuple[int, int]):
        raise ValueError("Cannot set shape attribute!")


    def get_attr_type(self, attr:str) -> str:
        """ Return registered type for an attribute. If not registered, return None
        """
        if ("_attr2type" not in self.metadata) or (attr not in self.metadata["_attr2type"]):
            return None
        return self.metadata["_attr2type"][attr]

    def register_attr(self, attr: str, attr_type: str = None) -> None:
        """ Register an attribute (either in obs or obsm) with an attr_type (e.g. signature, cluster, basis, knn). Default is None (no attr_type or pop up)
        """
        if attr_type is None:
            if "_attr2type" in self.metadata:
                self.metadata["_attr2type"].pop(attr, None)
        else:
            if "_attr2type" not in self.metadata:
                self.metadata["_attr2type"] = dict()
            self.metadata["_attr2type"][attr] = attr_type


    def _polish_featurekey(self, feature_metadata: pd.DataFrame, genome: str) -> None:
        """    Remove prefixing genome strings and deduplicate feature keys
        """
        if genome != "unknown":
            # remove genome strings for 10x chemistry if > 1 genomes exist
            import re
            prefix = re.compile(f"^{genome}_+")
            if prefix.match(feature_metadata["featurekey"][0]):
                feature_metadata["featurekey"] = np.array([prefix.sub("", x) for x in feature_metadata["featurekey"].values], dtype = object)
                if "featureid" in feature_metadata:
                    feature_metadata["featureid"] = np.array([prefix.sub("", x) for x in feature_metadata["featureid"].values], dtype = object)

        if feature_metadata["featurekey"].duplicated().sum() > 0:
            # deduplicate feature keys
            keys = feature_metadata["featurekey"].values
            ids = feature_metadata["featureid"].values if "featureid" in feature_metadata else None

            from collections import Counter
            dup_ids = Counter()

            for i in range(keys.size):
                idn = dup_ids[keys[i]]
                if idn > 0:
                    # key only is duplicated
                    if ids is not None:
                        keys[i] = keys[i] + '_' + ids[i]
                        idn = dup_ids[keys[i]]
                idn += 1
                dup_ids[keys[i]] = idn
                if idn > 1:
                    keys[i] = keys[i] + f".#~{idn}" # duplicate ID starts from 2. .#~ makes it unique.


    def get_genome(self) -> str:
        """ return genome
        """
        return self.metadata.get("genome", None)


    def get_modality(self) -> str:
        """ return modality, can be either 'rna', 'atac', 'tcr', 'bcr', 'crispr', 'hashing', 'citeseq' or 'cyto' (flow cytometry / mass cytometry).
        """
        return self.metadata.get("modality", None)


    def get_uid(self) -> str:
        """ return uid used for indexing this object in a MultimodalData object.
        """
        return self.metadata.get("uid", None)


    def list_keys(self, key_type: str = "matrix") -> List[str]:
        """ Return available keys in metadata, key_type = barcode, feature, matrix, other
        """
        if key_type == "matrix":
            return list(self.matrices)
        elif key_type == "barcode":
            return [
                self.barcode_metadata.index.name
            ] + self.barcode_metadata.columns.tolist()
        elif key_type == "feature":
            return [
                self.feature_metadata.index.name
            ] + self.feature_metadata.columns.tolist()
        elif key_type == "other":
            return list(self.metadata)
        else:
            raise ValueError(f"Unknown type {key_type}!")


    def current_matrix(self) -> str:
        """ Return current matrix as self.X
        """
        return self._cur_matrix


    def add_matrix(self, key: str, mat: csr_matrix) -> None:
        """ Add a new matrix, can be raw count or others
        """
        if key in self.matrices:
            raise ValueError(f"Matrix key '{key}' already exists!")
        if mat.shape[0] != self._shape[0]:
            raise ValueError(f"Wrong number of barcodes: matrix '{key}' has {mat.shape[0]} barcodes, which does not match with the barcode file ({self._shape[0]})!")
        if mat.shape[1] != self._shape[1]:
            raise ValueError(f"Wrong number of features: matrix '{key}' has {mat.shape[1]} features, which does not match with the feature file ({self._shape[1]})!")

        self.matrices[key] = mat


    def select_matrix(self, key: str) -> None:
        """ Select a matrix for self.X
        """
        if key not in self.matrices:
            raise ValueError(f"Matrix key '{key}' does not exist!")
        self._cur_matrix = key


    def get_matrix(self, key: str) -> csr_matrix:
        """ Return a matrix indexed by key
        """
        if key not in self.matrices:
            raise ValueError(f"Matrix key '{key}' does not exist!")
        return self.matrices[key]


    def as_float(self, key: str = None) -> None:
        """ Convert self.matrices[key] as float
        """
        key = self._cur_matrix if key is None else key
        X = self.matrices[key]
        if X.dtype == np.int32:
            X.dtype = np.float32
            orig_data = X.data.view(np.int32)
            X.data[...] = orig_data


    @run_gc
    def _inplace_subset_obs(self, index: List[bool]) -> None:
        """ Subset barcode_metadata inplace """
        if isinstance(index, pd.Series):
            index = index.values
        if index.sum() == self._shape[0]:
            return None
        self.barcode_metadata = self.barcode_metadata.loc[index].copy(deep = False)
        for key in list(self.matrices):
            self.matrices[key] = self.matrices[key][index, :]
        for key in list(self.barcode_multiarrays):
            self.barcode_multiarrays[key] = self.barcode_multiarrays[key][index]
        for key in list(self.barcode_multigraphs):
            self.barcode_multigraphs[key] = self.barcode_multigraphs[key][index][:,index]
        if "_obs_keys" in self.metadata:
            for key in self.metadata["_obs_keys"]:
                self.metadata[key] = self.metadata[key][index]
        self._update_shape()


    @run_gc
    def _inplace_subset_var(self, index: List[bool]) -> None:
        """ Subset feature_metadata inplace """
        if isinstance(index, pd.Series):
            index = index.values
        if index.sum() == self._shape[1]:
            return None
        self.feature_metadata = self.feature_metadata.loc[index].copy(deep = False)
        for key in list(self.matrices):
            self.matrices[key] = self.matrices[key][:, index]
        for key in list(self.feature_multiarrays):
            self.feature_multiarrays[key] = self.feature_multiarrays[key][index]
        for key in list(self.feature_multigraphs):
            self.feature_multigraphs[key] = self.feature_multigraphs[key][index][:,index]
        if "_var_keys" in self.metadata:
            for key in self.metadata["_var_keys"]:
                self.metadata[key] = self.metadata[key][index]
        self._update_shape()


    def separate_channels(self) -> None:
        """ Separate channel information from barcodekeys, used for 10x v2, v3 h5 and mtx as well as Optimus loom.
        """
        if self.barcode_metadata.shape[0] == 0 or self.barcode_metadata.index[0].find("-") < 0:
            return None # no data or no dash to remove

        try:
            from pegasusio.cylib.funcs import split_barcode_channel
        except ModuleNotFoundError:
            print("No module named 'pegasusio.cylib.funcs'")

        barcodes, channels = split_barcode_channel(self.barcode_metadata.index.values)

        if np.unique(channels).size > 1:
            # we have multiple channels
            self.barcode_metadata["Channel"] = channels
            barcodes = np.array([x + "-" + y for x, y in zip(channels, barcodes)], dtype = object)

        self.barcode_metadata.index = pd.Index(barcodes, name="barcodekey")



    def scan_black_list(self, black_list: Set[str]):
        """ Remove (key, value) pairs where key is in black_list
        """
        def _scan_dataframe(df: pd.DataFrame, black_list: Set[str]):
            cols = []
            for key in df.columns:
                if key in black_list:
                    cols.append(key)
            if len(cols) > 0:
                df.drop(columns = cols, inplace = True)

        def _scan_dict(mapping: dict, black_list: Set[str]):
            for key in list(mapping):
                if key in black_list:
                    del mapping[key]

        _scan_dataframe(self.barcode_metadata, black_list)
        _scan_dataframe(self.feature_metadata, black_list)

        _scan_dict(self.matrices, black_list)
        _scan_dict(self.barcode_multiarrays, black_list)
        _scan_dict(self.feature_multiarrays, black_list)
        _scan_dict(self.barcode_multigraphs, black_list)
        _scan_dict(self.feature_multigraphs, black_list)
        _scan_dict(self.metadata, black_list)


    def from_anndata(self, data: anndata.AnnData, genome: str = None, modality: str = None, uid: str = None) -> None:
        """ Initialize from an anndata object, try best not to copy
            If genome/modality is not None, set 'genome'/'modality' as genome/modality
        """
        if data.is_view:
            data = data.copy()

        self.barcode_metadata = data.obs
        self.barcode_metadata.index.name = "barcodekey"

        self.feature_metadata = data.var
        self.feature_metadata.index.name = "featurekey"
        if "gene_ids" in self.feature_metadata:
            self.feature_metadata.rename(columns = {"gene_ids": "featureid"}, inplace = True)

        self.matrices = DataDict({"X": csr_matrix(data.X)}) # csr_matrix will not copy X.data
        if data.raw is not None:
            self.matrices["raw.X"] = csr_matrix(data.raw.X)
        for key, value in data.layers.items():
            self.matrices[key] = csr_matrix(value)

        def _create_data_dict(old_dict: dict) -> DataDict:
            from pandas.api.types import is_dict_like
            new_dict = dict()
            for key, value in old_dict.items():
                if str(type(value)).find("anndata") >= 0:
                    # This is anndata defined type
                    if is_dict_like(value):
                        new_dict[key] = dict(value)
                    else:
                        logger.warning(f"{key} is in anndata-defined data type {type(value)} and thus skipped!")
                else:
                    new_dict[key] = value
            return DataDict(new_dict)

        self.barcode_multiarrays = _create_data_dict(data.obsm)
        self.feature_multiarrays = _create_data_dict(data.varm)
        self.barcode_multigraphs = _create_data_dict(data.obsp) # Note that we do not check if data.obsp contains sparse matrix
        self.feature_multigraphs = _create_data_dict(data.varp) # Note that we do not check if data.obsp contains sparse matrix
        self.metadata = _create_data_dict(data.uns)
        _set_genome(self.metadata, genome)
        _set_modality(self.metadata, modality)
        _set_uid(self.metadata, uid)


        self._cur_matrix = "X"
        self._shape = data.shape

    def to_anndata(self) -> anndata.AnnData:
        """ Convert to anndata
        """

        X_key = raw_key = None
        if "X" in self.matrices:
            X_key = "X"
            if "raw.X" in self.matrices:
                raw_key = "raw.X"
        else:
            X_key = self._cur_matrix
            components = X_key.split(".")
            if len(components) > 1:
                raw_key = '.'.join(components[:-1])
                if raw_key not in self.matrices:
                    raw_key = None

        raw = None
        if raw_key != None:
            var_cols = []
            if "featureid" in self.feature_metadata:
                var_cols.append("featureid")
            raw = anndata.AnnData(X = self.matrices[raw_key], dtype = self.matrices[raw_key].dtype, var = self.feature_metadata[var_cols])

        layers = {}
        for key, value in self.matrices.items():
            if key != X_key and key != raw_key:
                layers[key] = value

        return anndata.AnnData(X = self.matrices[X_key],
            dtype = self.matrices[X_key].dtype,
            obs = self.barcode_metadata,
            var = self.feature_metadata,
            uns = self.metadata,
            obsm = self.barcode_multiarrays,
            varm = self.feature_multiarrays,
            obsp = self.barcode_multigraphs,
            varp = self.feature_multigraphs,
            layers = layers,
            raw = raw)


    def copy(self) -> "UnimodalData":
        return UnimodalData(self.barcode_metadata.copy(),
                            self.feature_metadata.copy(),
                            deepcopy(self.matrices),
                            deepcopy(self.metadata),
                            deepcopy(self.barcode_multiarrays),
                            deepcopy(self.feature_multiarrays),
                            deepcopy(self.barcode_multigraphs),
                            deepcopy(self.feature_multigraphs),
                            self._cur_matrix)


    def __deepcopy__(self, memo: Dict):
        return self.copy()


    def _copy_view(self, viewobj: UnimodalDataView) -> "UnimodalData":
        """ In uns: Do not copy any sparse matrix or ndarrady with ndim > 1
        """
        return UnimodalData(viewobj.obs.copy(),
                            viewobj.var.copy(),
                            viewobj._copy_matrices(),
                            viewobj.uns[...],
                            viewobj.obsm[...],
                            viewobj.varm[...],
                            viewobj.obsp[...],
                            viewobj.varp[...],
                            viewobj._cur_matrix)


    def __getitem__(self, index: INDEX) -> UnimodalDataView:
        barcode_index, feature_index = _parse_index(self, index)
        return UnimodalDataView(self, barcode_index, feature_index, self._cur_matrix)


    def _update_barcode_metadata_info(
        self, row: pd.Series, attributes: Set[str], append_sample_name: bool
    ) -> None:
        """ Update barcodekey, update channel and add attributes
        """
        nsample = self.barcode_metadata.shape[0]
        sample_name = row["Sample"]

        if append_sample_name:
            barcodes = [sample_name + "-" + x for x in self.barcode_metadata.index]
            self.barcode_metadata.index = pd.Index(barcodes, name="barcodekey")
            if "Channel" in self.barcode_metadata:
                self.barcode_metadata["Channel"] = [
                    sample_name + "-" + x for x in self.barcode_metadata["Channel"]
                ]

        if "Channel" not in self.barcode_metadata:
            self.barcode_metadata["Channel"] = np.repeat(sample_name, nsample)

        for attr in attributes:
            self.barcode_metadata[attr] = np.repeat(row[attr], nsample)

        self.metadata["_sample"] = sample_name # record sample_name for merging


    def _convert_attributes_to_categorical(self, attributes: Set[str]) -> None:
        for attr in attributes:
            values = self.barcode_metadata[attr].values
            self.barcode_metadata[attr] = pd.Categorical(values, categories=natsorted(np.unique(values)))


    def _clean_tmp(self) -> dict:
        _tmp_dict = {}
        for key in ["metadata", "barcode_multiarrays", "feature_multiarrays", "barcode_multigraphs", "feature_multigraphs"]:
            obj = getattr(self, key)
            a_dict = {}
            for attr in list(obj):
                if attr.startswith("_tmp"):
                    a_dict[attr] = obj.pop(attr)
            if len(a_dict) > 0:
                _tmp_dict[key] = a_dict
        return _tmp_dict if len(_tmp_dict) > 0 else None

    def _addback_tmp(self, _tmp_dict: dict) -> None:
        for key in ["metadata", "barcode_multiarrays", "feature_multiarrays", "barcode_multigraphs", "feature_multigraphs"]:
            if key in _tmp_dict:
                getattr(self, key).update(_tmp_dict[key])
