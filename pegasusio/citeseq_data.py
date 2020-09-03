import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from typing import List, Dict, Union

import logging
logger = logging.getLogger(__name__)

import anndata
from pegasusio import UnimodalData
from .views import INDEX, _parse_index, UnimodalDataView


# According to communications with Biolegend, isotype control as background is not recommended.

class CITESeqData(UnimodalData):
    _matrix_keywords = ["arcsinh.transformed", "raw.count", "arcsinh.jitter"] # 'arcsinh.jitter' is in dense format, np.ndarray
    _uns_keywords = ["_other_names", "_other_counts"] # '_other' are antibodies that set aside

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

        signal = self.matrices["raw.count"].copy()

        if jitter:
            np.random.seed(random_state)
            jitters = np.random.uniform(low = -0.5, high = 0.5, size = signal.shape)
            signal = np.add(signal.toarray(), jitters, dtype = np.float32)
            self.matrices["arcsinh.jitter"] = np.arcsinh(signal / cofactor, dtype = np.float32)
        else:
            signal.data = np.arcsinh(signal.data / cofactor, dtype = np.float32)
            self.matrices["arcsinh.transformed"] = signal

        if select:
            self._cur_matrix = "arcsinh.jitter" if jitter else "arcsinh.transformed"
