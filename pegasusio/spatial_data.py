import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections.abc import MutableMapping
from copy import deepcopy
from natsort import natsorted
from typing import List, Dict, Union, Set, Tuple

import logging

from pegasusio.unimodal_data import UnimodalData
logger = logging.getLogger(__name__)

import anndata

from pegasusio import run_gc
from pegasusio import modalities
from .views import INDEX, _parse_index, UnimodalDataView
from .datadict import DataDict



class SpatialData(UnimodalData):
       def __init__(
        self,
        barcode_metadata: Union[dict, pd.DataFrame],
        feature_metadata: Union[dict, pd.DataFrame],
        matrices: Dict[str, csr_matrix],
        metadata: dict,
        barcode_multiarrays: Dict[str, np.ndarray] = None,
        feature_multiarrays: Dict[str, np.ndarray] = None,
        barcode_multigraphs: Dict[str, csr_matrix] = None,
        feature_multigraphs: Dict[str, csr_matrix] = None,
        cur_matrix: str = "raw.data",
        img = pd.DataFrame
    ) -> None:
        assert metadata["modality"] == "visium"
        super().__init__(barcode_metadata,
        feature_metadata, 
        matrices, 
        metadata, 
        barcode_multiarrays, 
        feature_multiarrays, 
        barcode_multigraphs, 
        feature_multigraphs, 
        cur_matrix)
        self.img = img
    
        @property
        def img(self) -> Union[pd.DataFrame, None]:
            return self.img

        # Set the img field if needed
        @img.setter
        def img(self, img: pd.DataFrame):
            self.img = img



