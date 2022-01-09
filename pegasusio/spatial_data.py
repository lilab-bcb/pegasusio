import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Dict, Optional, Union

import logging

from pegasusio.unimodal_data import UnimodalData

logger = logging.getLogger(__name__)


class SpatialData(UnimodalData):
    """
    Class to implement data structure to
    manipulate spatial data with the spatial image (img) field
    This class extends UnimodalData with additional
    functions specific to the img field
    """

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
        cur_matrix: str = "X",
        image_metadata: Optional[pd.DataFrame] = None,
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
        self.image_metadata = image_metadata

    @property
    def img(self) -> Optional[pd.DataFrame]:
        return self.image_metadata

    @img.setter
    def img(self, img: pd.DataFrame):
        self.image_metadata = img

    def __repr__(self) -> str:
        repr_str = super().__repr__()
        key = "img"
        fstr = self._gen_repr_str_for_attrs(key)
        if fstr != "":
            repr_str += f"\n    {key}: {fstr}"
        return repr_str
