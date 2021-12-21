import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Dict, Optional, Union

import logging

from pegasusio.unimodal_data import UnimodalData
logger = logging.getLogger(__name__)


class SpatialData(UnimodalData):
    def __init__(
        self,
        barcode_metadata: Optional[Union[dict, pd.DataFrame]] = None,
        feature_metadata: Optional[Union[dict, pd.DataFrame]] = None,
        matrices: Optional[Dict[str, csr_matrix]] = None,
        metadata: Optional[dict] = None,
        barcode_multiarrays: Optional[Dict[str, np.ndarray]] = None,
        feature_multiarrays: Optional[Dict[str, np.ndarray]] = None,
        barcode_multigraphs: Optional[Dict[str, csr_matrix]] = None,
        feature_multigraphs: Optional[Dict[str, csr_matrix]] = None,
        cur_matrix: str = "raw.data",
        img = None,
    ) -> None:
        assert metadata["modality"] == "visium"
        super().__init__(
            barcode_metadata,
            feature_metadata,
            matrices,
            metadata,
            barcode_multiarrays,
            feature_multiarrays,
            barcode_multigraphs,
            feature_multigraphs,
            cur_matrix,
        )
        self._img = img

    @property
    def img(self) -> Optional[pd.DataFrame]:
        return self._img

    @img.setter
    def img(self, img: pd.DataFrame):
        self._img = img

    def __repr__(self) -> str:
        repr_str = super().__repr__()
        key = "img"
        fstr = self._gen_repr_str_for_attrs(key)
        if fstr != "":
            repr_str += f"\n    {key}: {fstr}"
        return repr_str