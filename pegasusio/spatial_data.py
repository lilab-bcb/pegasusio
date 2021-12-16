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
        img=pd.DataFrame,
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
        self.img = img

        @property
        def img(self) -> Union[pd.DataFrame, None]:
            return self.img

        # Set the img field if needed
        @img.setter
        def img(self, img: pd.DataFrame):
            self.img = img

        def __repr__(self) -> str:
            repr_str = f"{self.__class__.__name__} object with n_obs x n_vars = {self.barcode_metadata.shape[0]} x {self.feature_metadata.shape[0]}"
            repr_str += (
                f"\n    Genome: {self.get_genome()}; Modality: {self.get_modality()}"
            )
            mat_word = "matrices" if len(self.matrices) > 1 else "matrix"
            repr_str += f"\n    It contains {len(self.matrices)} {mat_word}: {str(list(self.matrices))[1:-1]}"
            repr_str += (
                f"\n    It currently binds to matrix '{self._cur_matrix}' as X\n"
                if len(self.matrices) > 0
                else "\n    It currently binds to no matrix\n"
            )
            for key in ["obs", "var", "obsm", "varm", "obsp", "varp", "uns"]:
                fstr = self._gen_repr_str_for_attrs(key)
                if fstr != "":
                    repr_str += f"\n    {key}: {fstr}"

        return repr_str
