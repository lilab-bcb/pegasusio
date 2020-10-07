import numpy as np
import pandas as pd
from typing import List, Dict, Union

import logging
logger = logging.getLogger(__name__)

import anndata
from pegasusio import UnimodalData
from .views import INDEX, _parse_index, UnimodalDataView


class NanostringData(UnimodalData):
    _matrix_keywords = ["Q3Norm", "LogMatrix"] # all in dense format, np.ndarray

    def __init__(
        self,
        barcode_metadata: Union[dict, pd.DataFrame],
        feature_metadata: Union[dict, pd.DataFrame],
        matrices: Dict[str, np.ndarray],
        metadata: dict,
        barcode_multiarrays: Dict[str, np.ndarray] = None,
        feature_multiarrays: Dict[str, np.ndarray] = None,
        cur_matrix: str = "Q3Norm",
    ) -> None:
        assert metadata["modality"] == "nanostring"
        super().__init__(barcode_metadata, feature_metadata, matrices, metadata, barcode_multiarrays, feature_multiarrays, cur_matrix)


    def from_anndata(self, data: anndata.AnnData, genome: str = None, modality: str = None) -> None:
        raise ValueError("Cannot convert an AnnData object to a NanostringData object!")

    
    def to_anndata(self) -> anndata.AnnData:
        raise ValueError("Cannot convert a NanostringData object ot an AnnData object!")


    def __getitem__(self, index: INDEX) -> UnimodalDataView:
        barcode_index, feature_index = _parse_index(self, index)
        return UnimodalDataView(self, barcode_index, feature_index, self._cur_matrix, obj_name = "NanostringData")


    def log_norm(self, select: bool = True) -> None:
        """Conduct log normalization on the Q3Norm matrix: log(x + 1).
        
        Add log-transformed matrix 'LogMatrix'.
        
        Parameters
        ----------
        select: ``bool``, optional, default: ``True``
            If True, select the transformed matrix as the major matrix (X).

        Returns
        -------
        ``None``

        Update ``self.matrices``:
            * ``self.matrices['LogMatrix']``: np.ndarray.

        Examples
        --------
        >>> nanostring_data.log_norm()
        """
        if "Q3Norm" not in self.matrices:
            raise ValueError("Q3Norm matrix must exist in order to calculate the log transformed matrix!")

        self.matrices["LogMatrix"] = np.log1p(self.matrices["Q3Norm"])

        if select:
            self._cur_matrix = "LogMatrix"
