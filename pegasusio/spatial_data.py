import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from typing import Dict, Union, Optional
from pegasusio import UnimodalData

import logging
logger = logging.getLogger(__name__)


_spatial_modalities = ['visium', 'merfish']

class SpatialData(UnimodalData):
    _matrix_keywords = ["raw.count"]
    _spatial_data_columns = ['in_tissue', 'array_row', 'array_col']
    _spatial_coordinates_columns = ['pxl_col_in_fullres', 'pxl_row_in_fullres']
    _image_data_columns = ['sample_id', 'image_id', 'data', 'scaleFactor']

    def __init__(
        self,
        barcode_metadata: Union[dict, pd.DataFrame],
        feature_metadata: Union[dict, pd.DataFrame],
        matrices: Dict[str, np.ndarray],
        metadata: dict,
        barcode_multiarrays: Optional[Dict[str, np.ndarray]] = None,
        feature_multiarrays: Optional[Dict[str, np.ndarray]] = None,
        barcode_multigraphs: Optional[Dict[str, csr_matrix]] = None,
        feature_multigraphs: Optional[Dict[str, csr_matrix]] = None,
        cur_matrix: str = "raw.data",
        image_data: Optional[pd.DataFrame] = None,
    ):
        assert metadata["modality"] in _spatial_modalities, "Must be data in Spatial modality!"
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

        if image_data is None:
            self.image_data = pd.DataFrame(columns=self._image_data_columns)


    def spatial_data(self) -> pd.DataFrame:
        return self.barcode_metadata[self._spatial_data_columns]

    def spatial_coords(self) -> pd.DataFrame:
        return self.barcode_metadata[self._spatial_coordinates_columns]
