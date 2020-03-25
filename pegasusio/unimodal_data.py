import gc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections.abc import MutableMapping
from typing import List, Dict, Union, Set

import logging
logger = logging.getLogger(__name__)

import anndata
from pegasusio.cylib.funcs import split_barcode_channel



class UnimodalData:
    def __init__(
        self,
        barcode_metadata: Union[dict, pd.DataFrame, anndata.AnnData] = None,
        feature_metadata: Union[dict, pd.DataFrame] = None,
        matrices: Dict[str, csr_matrix] = None,
        barcode_multiarrays: Dict[str, np.ndarray] = None,
        feature_multiarrays: Dict[str, np.ndarray] = None,
        metadata: dict = None,
    ) -> None:
        if isinstance(barcode_metadata, anndata.AnnData):
            self.from_anndata(barcode_metadata)
            return None

        def replace_none_df(value):
            return value if value is not None else pd.DataFrame()

        def replace_none(value):
            """ Set empty dictionary as default is dangerous since others might fill in values to the dict
            """
            return value if value is not None else dict()

        self.barcode_metadata = replace_none_df(barcode_metadata) # barcode metadata
        self.feature_metadata = replace_none_df(feature_metadata) # feature metadata
        self.matrices = replace_none(matrices) # a dictionary of scipy csr matrix
        self.barcode_multiarrays = replace_none(barcode_multiarrays)
        self.feature_multiarrays = replace_none(feature_multiarrays)
        self.metadata = replace_none(metadata)  # other metadata, a dictionary
        self.cur_matrix = "X" # default matrix is X

        if len(self.barcode_metadata) > 0:
            if isinstance(self.barcode_metadata, MutableMapping):
                if "barcodekey" not in self.barcode_metadata:
                    raise ValueError("Cannot locate barcode index barcodekey!")
                barcodekey = self.barcode_metadata.pop("barcodekey")
                self.barcode_metadata = pd.DataFrame(data = self.barcode_metadata, index = pd.Index(barcodekey, name = "barcodekey"))
            else:
                if not isinstance(self.barcode_metadata, pd.DataFrame):
                    raise ValueError("Unknown barcode_metadata type: {}!".format(type(self.barcode_metadata)))
                if "barcodekey" in self.barcode_metadata:
                    self.barcode_metadata.set_index("barcodekey", inplace=True)
                elif self.barcode_metadata.index.name != "barcodekey":
                    raise ValueError("Cannot locate barcode index barcodekey!")

        if len(self.feature_metadata) > 0:
            if isinstance(self.feature_metadata, MutableMapping):
                self.feature_metadata = pd.DataFrame(self.feature_metadata)
            if not isinstance(self.feature_metadata, pd.DataFrame):
                raise ValueError("Unknown feature_metadata type: {}!".format(type(self.feature_metadata)))
            if "featurekey" in self.feature_metadata:
                self.polish_featurekey(self.feature_metadata, self.metadata.get('genome', None))
                self.feature_metadata.set_index("featurekey", inplace=True)
            elif self.feature_metadata.index.name != "featurekey":
                raise ValueError("Cannot locate feature index featurekey!")

        if (self.X is not None) and (self.barcode_metadata.shape[0] != self.X.shape[0]):
            raise ValueError(
                "Wrong number of cells : matrix has {} cells, barcodes file has {}".format(
                    self.X.shape[0], self.barcode_metadata.shape[0]
                )
            )
        if (self.X is not None) and (self.feature_metadata.shape[0] != self.X.shape[1]):
            raise ValueError(
                "Wrong number of features : matrix has {} features, features file has {}".format(
                    self.X.shape[1], self.feature_metadata.shape[0]
                )
            )


    def __repr__(self) -> str:
        repr_str = "UnimodalData object with n_obs x n_vars = {} x {}".format(self.barcode_metadata.shape[0], self.feature_metadata.shape[0])
        repr_str += "\n    It contains {} matrices: {}".format(len(self.matrices), str(list(self.matrices))[1:-1])
        repr_str += "\n    It currently binds to matrix '{}' as X\n".format(self.cur_matrix) if len(self.matrices) > 0 else "\n    It currently binds to no matrix\n"
        for key in ["obs", "var", "obsm", "varm", "uns"]:
            repr_str += "\n    {}: {}".format(key, str(list(getattr(self, key).keys()))[1:-1])

        return repr_str


    @property
    def obs(self) -> pd.DataFrame:
        return self.barcode_metadata

    @obs.setter
    def obs(self, obs: pd.DataFrame):
        assert obs.shape[0] == 0 or obs.index.name == "barcodekey"
        self.barcode_metadata = obs

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

    @property
    def var_names(self) -> pd.Index:
        return self.feature_metadata.index

    @var_names.setter
    def var_names(self, var_names: pd.Index):
        self.feature_metadata.index = var_names
        self.feature_metadata.index.name = "featurekey"

    @property
    def X(self) -> Union[csr_matrix, None]:
        return self.matrices.get(self.cur_matrix, None)

    @X.setter
    def X(self, X: csr_matrix):
        self.matrices[self.cur_matrix] = X

    @property
    def obsm(self) -> Dict[str, np.ndarray]:
        return self.barcode_multiarrays

    @obsm.setter
    def obsm(self, obsm: Dict[str, np.ndarray]):
        self.barcode_multiarrays = obsm

    @property
    def varm(self) -> Dict[str, np.ndarray]:
        return self.feature_multiarrays

    @varm.setter
    def varm(self, varm: Dict[str, np.ndarray]):
        self.feature_multiarrays = varm

    @property
    def uns(self) -> dict:
        return self.metadata

    @uns.setter
    def uns(self, uns: dict):
        self.metadata = uns


    def polish_featurekey(self, feature_metadata: pd.DataFrame, genome: str) -> None:
        """    Remove prefixing genome strings and deduplicate feature keys
        """
        if genome is not None:
            # remove genome strings for 10x chemistry if > 1 genomes exist
            import re
            prefix = re.compile("^{}_+".format(genome))
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
                    keys[i] = keys[i] + ".#~{}".format(idn) # duplicate ID starts from 2. .#~ makes it unique.


    def get_exptype(self) -> str:
        """ return experiment tpye, can be either 'rna', 'citeseq', 'hashing', 'tcr', 'bcr', 'crispr' or 'atac'.
        """
        return self.metadata.get("experiment_type", None)


    def get_genome(self) -> str:
        """ return genome key
        """
        return self.metadata.get("genome", None)


    def list_keys(self, key_type: str = "matrix") -> List[str]:
        """ Return available keys in metadata, key_type = barcode, feature, matrix, other
        """
        if key_type == "barcode":
            return [
                self.barcode_metadata.index.name
            ] + self.barcode_metadata.columns.tolist()
        elif key_type == "feature":
            return [
                self.feature_metadata.index.name
            ] + self.feature_metadata.columns.tolist()
        elif key_type == "matrix":
            return list(self.matrices)
        elif key_type == "other":
            return list(self.metadata)
        else:
            raise ValueError("Unknown type {}!".format(type))


    def current_matrix(self) -> str:
        """ Return current matrix as self.X
        """
        return self.cur_matrix


    def add_matrix(self, key: str, mat: csr_matrix) -> None:
        """ Add a new matrix, can be raw count or others
        """
        if key in self.matrices:
            raise ValueError("Matrix key {} already exists!".format(key))
        if self.barcode_metadata.shape[0] != mat.shape[0]:
            raise ValueError("Wrong number of cells: matrix has {} cells, which is not match with barcode file ({})!".format(mat.shape[0], self.barcode_metadata.shape[0]))
        if self.feature_metadata.shape[0] != mat.shape[1]:
            raise ValueError("Wrong number of features: matrix has {} features, which is not match with feature file ({})!".format(mat.shape[1], self.feature_metadata.shape[0]))

        self.matrices[key] = mat


    def select_matrix(self, key: str) -> None:
        """ Select a matrix for self.X
        """
        if key not in self.matrices:
            raise ValueError("Matrix key {} does not exist!".format(key))
        self.cur_matrix = key


    def get_matrix(self, key: str) -> csr_matrix:
        """ Return a matrix indexed by key
        """
        if key not in self.matrices:
            raise ValueError("Matrix key {} does not exist!".format(key))
        return self.matrices[key]


    def trim(self, selected: List[bool]) -> None:
        """ Only keep barcodes in selected
        """
        self.barcode_metadata = self.barcode_metadata[selected]
        for key, mat in self.matrices.items():
            self.matrices[key] = mat[selected, :]
        gc.collect()


    def filter(self, ngene: int = None, select_singlets: bool = False) -> None:
        """ Filter out low quality barcodes, only keep barcodes satisfying ngene >= ngene and selecting singlets if select_singlets is True
        """
        if (len(self.matrices) == 0) or ((ngene is None) and (not select_singlets)):
            return None

        self.select_matrix("X")
        selected = np.ones(self.X.shape[0], dtype=bool)
        if ngene is not None:
            selected = selected & (self.X.getnnz(axis=1) >= ngene)
        if select_singlets and ("demux_type" in self.barcode_metadata):
            selected = (
                selected & (self.barcode_metadata["demux_type"] == "singlet").values
            )
            self.barcode_metadata.drop(columns="demux_type", inplace=True)

        self.trim(selected)


    def separate_channels(self) -> None:
        """ Separate channel information from barcodekeys, only used for 10x v2, v3 h5 and mtx.
        """
        if self.barcode_metadata.shape[0] == 0:
            return None # no data

        barcodes, channels = split_barcode_channel(self.barcode_metadata.index.values)

        if channels[0] is None:
            return None # no need to separate channel information and the file should not be generated by cellranger

        if (channels != "1").sum() > 0:
            # we have multiple channels
            self.barcode_metadata["Channel"] = channels
            barcodes = np.array([x + "-" + y for x, y in zip(channels, barcodes)], dtype = object)
            
        self.barcode_metadata.index = pd.Index(barcodes, name="barcodekey")


    def update_barcode_metadata_info(
        self, sample_name: str, row: "pd.Series", attributes: List[str]
    ) -> None:
        """ Update barcodekey, update channel and add attributes
        """
        nsample = self.barcode_metadata.shape[0]
        barcodes = [sample_name + "-" + x for x in self.barcode_metadata.index]
        self.barcode_metadata.index = pd.Index(barcodes, name="barcodekey")
        if "Channel" in self.barcode_metadata:
            self.barcode_metadata["Channel"] = [
                sample_name + "-" + x for x in self.barcode_metadata["Channel"]
            ]
        else:
            self.barcode_metadata["Channel"] = np.repeat(sample_name, nsample)
        if attributes is not None:
            for attr in attributes:
                self.barcode_metadata[attr] = np.repeat(row[attr], nsample)


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
            for key in mapping:
                if key in black_list:
                    mapping.pop(key)

        _scan_dataframe(self.barcode_metadata, black_list)
        _scan_dataframe(self.feature_metadata, black_list)

        _scan_dict(self.matrices, black_list)
        _scan_dict(self.barcode_multiarrays, black_list)
        _scan_dict(self.feature_multiarrays, black_list)
        _scan_dict(self.metadata, black_list)


    def from_anndata(self, data: anndata.AnnData, genome: str = None, exptype: str = None) -> None:
        """ Initialize from an anndata object
            If genome/exptype is not None, set 'genome'/'experiment_type' as genome/exptype
        """
        self.barcode_metadata = data.obs
        self.barcode_metadata.index.name = "barcodekey"
        
        self.feature_metadata = data.var
        self.feature_metadata.index.name = "featurekey"
        if "gene_ids" in self.feature_metadata:
            self.feature_metadata.rename(columns = {"gene_ids": "featureid"}, inplace = True)

        def _to_csr(X):
            return X if isinstance(X, csr_matrix) else csr_matrix(X)

        self.matrices = {"X": _to_csr(data.X)}
        if data.raw is not None:
            self.matrices["raw.X"] = _to_csr(data.raw.X)
        for key, value in data.layers.items():
            self.matrices[key] = _to_csr(value)

        self.barcode_multiarrays = dict(data.obsm)

        self.feature_multiarrays = dict(data.varm)

        self.metadata = dict(data.uns)

        if genome is not None:
            self.metadata["genome"] = genome
        elif "genome" not in self.metadata:
            self.metadata["genome"] = "unknown"
        elif isinstance(self.metadata["genome"], np.ndarray):
            assert self.metadata["genome"].ndim == 1
            self.metadata["genome"] = self.metadata["genome"][0]

        if exptype is not None:
            self.metadata["experiment_type"] = exptype
        elif "experiment_type" not in self.metadata:
            self.metadata["experiment_type"] = "rna"

        self.cur_matrix = "X"

    
    def to_anndata(self) -> anndata.AnnData:
        """ Convert to anndata
        """
        raw = None
        if "raw.X" in self.matrices:
            raw = anndata.AnnData(X = self.matrices["raw.X"])

        layers = {}
        for key, value in self.matrices.items():
            if key != "X" and key != "raw.X":
                layers[key] = value
        
        return anndata.AnnData(X = self.matrices.get("X", None),
            obs = self.barcode_metadata,
            var = self.feature_metadata,
            uns = self.metadata,
            obsm = self.barcode_multiarrays,
            varm = self.feature_multiarrays,
            layers = layers,
            raw = raw)
