import numpy as np
import pandas as pd
from typing import List, Dict, Union

import logging
logger = logging.getLogger(__name__)

import anndata
from pegasusio import UnimodalData
from .views import INDEX, _parse_index, UnimodalDataView


class NanostringData(UnimodalData):
    _matrix_keywords = ["RawData", "Q3Norm", "HKNorm", "LogMatrix"] # all in dense format, np.ndarray

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


    def norm_hk(self, select: bool = True) -> None:
        """ Normalize raw protein data using house keeping (HK) genes.
            See https://genomebiology.biomedcentral.com/articles/10.1186/gb-2002-3-7-research0034 for geometric mean normalization.

        Parameters
        ----------
        select: ``bool``, optional, default: ``True``
            If True, select the normalized matrix as the major matrix (X).

        Returns
        -------
        ``None``

        Update ``self.matrices``:
            * ``self.matrices['HKNorm']``: np.ndarray.

        Examples
        --------
        >>> nanostring_data.norm_hk()
        """
        from scipy.stats.mstats import gmean

        gms = gmean(self.obsm["controls"], axis = 1)
        norm_factors = gmean(gms) / gms
        self.matrices["HKNorm"] = self.matrices["RawData"] * norm_factors.reshape(-1, 1)
        self.obs["norm_factor"] = norm_factors

        if select:
            self._cur_matrix = "HKNorm"


    def log_transform(self, select: bool = True) -> None:
        """Conduct log transformation on the selected matrix: log(x + 1). Selected matrix can be either Q3Norm or HKNorm
        
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
        if self.current_matrix() not in ["Q3Norm", "HKNorm"]:
            raise ValueError("Either Q3Norm or HKNorm matrix must be selected in order to run the 'log_norm' function!")

        self.matrices["LogMatrix"] = np.log1p(self.X)

        if select:
            self._cur_matrix = "LogMatrix"
