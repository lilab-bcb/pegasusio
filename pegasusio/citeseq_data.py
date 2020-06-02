import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from typing import List, Dict, Union

import logging
logger = logging.getLogger(__name__)

import anndata
from pegasusio import UnimodalData
from .views import INDEX, _parse_index, UnimodalDataView


class CITESeqData(UnimodalData):
    _matrix_keywords = ["arcsinh.transformed", "log.transformed", "raw.count", "arcsinh.jitter"] # 'arcsinh.jitter' is in dense format, np.ndarray
    _uns_keywords = ["_control_names", "_control_counts", "_other_names", "_other_counts"] # '_other' are antibodies that set aside
    _var_keywords = ["_control_id"]

    def __init__(
        self,
        barcode_metadata: Union[dict, pd.DataFrame],
        feature_metadata: Union[dict, pd.DataFrame],
        matrices: Dict[str, csr_matrix],
        metadata: dict,
        barcode_multiarrays: Dict[str, np.ndarray] = None,
        feature_multiarrays: Dict[str, np.ndarray] = None,
        cur_matrix: str = "raw.count",
    ) -> None:
        assert metadata["modality"] == "citeseq"
        super().__init__(barcode_metadata, feature_metadata, matrices, metadata, barcode_multiarrays, feature_multiarrays, cur_matrix)
        
        # Prepare for the control antibody list if not loaded from Zarr
        if "_control_names" not in self.metadata:
            assert len(self.matrices) == 1 and "raw.count" in self.matrices
            self.metadata["_control_names"] = np.array(["None"], dtype = object)
            self.metadata["_control_counts"] = csr_matrix((self._shape[0], 1), dtype = np.int32)
            self.metadata["_obs_keys"] = ["_control_counts"]
            self.feature_metadata["_control_id"] = np.zeros(self._shape[1], dtype = np.int32)


    def from_anndata(self, data: anndata.AnnData, genome: str = None, modality: str = None) -> None:
        raise ValueError("Cannot convert an AnnData object to a CITESeqData object!")

    
    def to_anndata(self) -> anndata.AnnData:
        raise ValueError("Cannot convert a CITESeqData object ot an AnnData object!")


    def __getitem__(self, index: INDEX) -> UnimodalDataView:
        barcode_index, feature_index = _parse_index(self, index)
        return UnimodalDataView(self, barcode_index, feature_index, self._cur_matrix, obj_name = "CITESeqData")


    def set_aside(self, params: List[str]) -> None:
        """ Move antibodies in params from the raw.count matrix
        """
        assert len(self.matrices) == 1 and "raw.count" in self.matrices
        assert "_other_names" not in self.metadata

        locs = self.feature_metadata.index.get_indexer(params) 
        if (locs < 0).sum() > 0:
            raise ValueError(f"Detected unknown antibodies {params[locs < 0]}!")
        self.metadata["_other_names"] = self.feature_metadata.index.values[locs] # with loc: List[int], this should be a copy not a reference
        self.metadata["_other_counts"] = self.matrices["raw.count"][:, locs]

        obs_keys = self.metadata.get("_obs_keys", [])
        obs_keys.append("_other_counts")
        self.metadata["_obs_keys"] = obs_keys
        
        idx = np.ones(self._shape[1], dtype = bool)
        idx[locs] = False
        self._inplace_subset_var(idx)


    def load_control_list(self, control_csv: str) -> None:
        assert "raw.count" in self.matrices
        assert self.metadata["_control_names"].size == 1

        ctrls = {"None": 0}
        series = pd.read_csv(control_csv, header=0, index_col=0, squeeze=True)
        for antibody, control in series.iteritems():
            if antibody not in self.feature_metadata.index:
                continue

            pos = ctrls.get(control, None)
            if pos is None:
                if control in self.feature_metadata.index:
                    pos = len(ctrls)
                    ctrls[control] = pos
                else:
                    logger.warning(f"Detected and ignored unknown control antibody '{control}'!")
                    pos = 0

            self.feature_metadata.loc[antibody, "_control_id"] = pos

        ctrl_names = np.empty(len(ctrls), dtype = object)
        for ctrl_name, pos in ctrls.items():
            ctrl_names[pos] = ctrl_name
            
        locs = self.feature_metadata.index.get_indexer(pd.Index(ctrl_names[1:], copy = False))
        idx = np.ones(self._shape[1], dtype = bool)
        idx[locs] = False

        self.metadata["_control_names"] = ctrl_names
        self.metadata["_control_counts"] = hstack([self.metadata["_control_counts"], 
                                                   self.matrices["raw.count"][:, locs]], 
                                                   format = "csr")
        self._inplace_subset_var(idx)

        # Delete all other matrices in case users do transform before loading the controls
        for keyword in list(self.matrices):
            if keyword != "raw.count":
                del self.matrices[keyword]
        self._cur_matrix = "raw.count"


    def log_transform(self, select: bool = True) -> None:
        """ ln(x+1)"""
        if "raw.count" not in self.matrices:
            raise ValueError("raw.count matrix must exist in order to calculate the log transformed matrix!")

        log_mat = np.maximum(np.log1p(self.matrices["raw.count"].toarray(), dtype = np.float32) \
                             - np.log1p(self.metadata["_control_counts"].toarray()[:, self.feature_metadata["_control_id"].values], dtype = np.float32),
                             0.0)
        self.matrices["log.transformed"] = csr_matrix(log_mat)
        if select:
            self._cur_matrix = "log.transformed"


    def arcsinh_transform(self, cofactor: float = 5.0, jitter = False, random_state = 0, select: bool = True) -> None:
        """Conduct arcsinh transform on the raw.count matrix.
        
        Add arcsinh transformed matrix 'arcsinh.transformed'. If jitter == True, instead add a 'arcsinh.jitter' matrix in dense format, jittering by adding a randomized value in U([-0.5, 0.5)). Mimic Cytobank.
        
        Parameters
        ----------
        cofactor: ``float``, optional, default: ``5.0``
            Cofactor used in cytobank, arcsinh(x / cofactor).

        jitter: ``bool``, optional, default: ``False``
            Add a 'arcsinh.jitter' matrix in dense format, jittering by adding a randomized value in U([-0.5, 0.5)).

        random_state: ``int``, optional, default: ``0``
            Random seed for generating jitters.

        select: ``bool``, optional, default: ``True``
            If True, select the transformed matrix as the major matrix (X).

        Returns
        -------
        ``None``

        Update ``self.matrices``:
            * ``self.matrices['arcsinh.transformed']``: if jitter == False, csr_matrix.
            * ``self.matrices['arcsinh.jitter']``: if jitter == True, np.ndarray.

        Examples
        --------
        >>> citeseq_data.arcsinh_transform(jitter = True)
        """
        if "raw.count" not in self.matrices:
            raise ValueError("raw.count matrix must exist in order to calculate the arcsinh transformed matrix!")

        ctrl_ids = self.feature_metadata["_control_id"].values
        idx = ctrl_ids > 0

        signal = self.matrices["raw.count"].toarray()
        control = self.metadata["_control_counts"].toarray()[:, ctrl_ids]

        if jitter:
            np.random.seed(random_state)
            jitters = np.random.uniform(low = -0.5, high = 0.5, size = signal.shape)
            signal = np.add(signal, jitters, dtype = np.float32)

        signal = np.arcsinh(signal / cofactor, dtype = np.float32)
        control = np.arcsinh(control / cofactor, dtype = np.float32)
        arcsinh_mat = signal - control
        arcsinh_mat[:, idx] = np.maximum(arcsinh_mat[:, idx], 0.0)

        if jitter:
            self.matrices["arcsinh.jitter"] = arcsinh_mat
        else:
            self.matrices["arcsinh.transformed"] = csr_matrix(arcsinh_mat)

        if select:
            self._cur_matrix = "arcsinh.jitter" if jitter else "arcsinh.transformed"
