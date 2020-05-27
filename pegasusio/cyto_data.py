import numpy as np
import pandas as pd
from typing import List, Dict, Union

import logging
logger = logging.getLogger(__name__)

import anndata
from pegasusio import UnimodalData
from .views import INDEX, _parse_index, UnimodalDataView


class CytoData(UnimodalData):
    _matrix_keywords = ["arcsinh.transformed", "raw.data", "arcsinh.jitter"] # all in dense format, np.ndarray
    _obsm_keywords = ["_parameters", "_controls"] # _parameters store parameters that are not used, e.g. Time
    _uns_keywords = ["_parameter_names", "_control_names"]
    _var_keywords = ["_control_id"]

    def __init__(
        self,
        barcode_metadata: Union[dict, pd.DataFrame],
        feature_metadata: Union[dict, pd.DataFrame],
        matrices: Dict[str, np.ndarray],
        metadata: dict,
        barcode_multiarrays: Dict[str, np.ndarray] = None,
        feature_multiarrays: Dict[str, np.ndarray] = None,
        cur_matrix: str = "raw.data",
    ) -> None:
        assert metadata["modality"] == "cyto"
        super().__init__(barcode_metadata, feature_metadata, matrices, metadata, barcode_multiarrays, feature_multiarrays, cur_matrix)

        # Prepare for the control list if not loaded from Zarr
        if "_control_names" not in self.metadata:
            assert len(self.matrices) == 1 and "raw.data" in self.matrices
            self.metadata["_control_names"] = np.array(["None"], dtype = object)
            self.barcode_multiarrays["_controls"] = np.zeros((self._shape[0], 1), dtype = np.int32)
            self.feature_metadata["_control_id"] = np.zeros(self._shape[1], dtype = np.int32)


    def from_anndata(self, data: anndata.AnnData, genome: str = None, modality: str = None) -> None:
        raise ValueError("Cannot convert an AnnData object to a CytoData object!")

    
    def to_anndata(self) -> anndata.AnnData:
        raise ValueError("Cannot convert a CytoData object ot an AnnData object!")


    def __getitem__(self, index: INDEX) -> UnimodalDataView:
        barcode_index, feature_index = _parse_index(self, index)
        return UnimodalDataView(self, barcode_index, feature_index, self._cur_matrix, obj_name = "CytoData")


    def set_aside(self, params: List[str] = ["Time"]) -> None:
        """ Move parameters in params from the raw.data matrix
        """
        assert len(self.matrices) == 1 and "raw.data" in self.matrices
        assert "_parameter_names" not in self.metadata

        locs = self.feature_metadata.index.get_indexer(params) 
        if (locs < 0).sum() > 0:
            raise ValueError(f"Detected unknown parameters {params[locs < 0]}!")
        self.barcode_multiarrays["_parameters"] = self.matrices["raw.data"][:, locs]
        self.metadata["_parameter_names"] = self.feature_metadata.index.values[locs] # with loc: List[int], this should be a copy not a reference
        idx = np.ones(self._shape[1], dtype = bool)
        idx[locs] = False
        self._inplace_subset_var(idx)


    def load_control_list(self, control_csv: str) -> None:
        """ Load control csv and move control from matrix to obsm
        """
        assert "raw.data" in self.matrices
        assert self.metadata["_control_names"].size == 1

        ctrls = {"None": 0}
        series = pd.read_csv(control_csv, header=0, index_col=0, squeeze=True)
        for parameter, control in series.iteritems():
            if parameter not in self.feature_metadata.index:
                continue

            pos = ctrls.get(control, None)
            if pos is None:
                if control in self.feature_metadata.index:
                    pos = len(ctrls)
                    ctrls[control] = pos
                else:
                    logger.warning(f"Detected and ignored unknown control parameter '{control}'!")
                    pos = 0

            self.feature_metadata.loc[parameter, "_control_id"] = pos

        ctrl_names = np.empty(len(ctrls), dtype = object)
        for ctrl_name, pos in ctrls.items():
            ctrl_names[pos] = ctrl_name

        locs = self.feature_metadata.index.get_indexer(pd.Index(ctrl_names[1:], copy = False))
        idx = np.ones(self._shape[1], dtype = bool)
        idx[locs] = False

        self.metadata["_control_names"] = ctrl_names
        self.barcode_multiarrays["_controls"] = np.hstack((self.barcode_multiarrays["_controls"], self.matrices["raw.data"][:, locs]))
        self._inplace_subset_var(idx)

        # Delete all other matrices in case users do transform before loading the controls
        for keyword in list(self.matrices):
            if keyword != "raw.data":
                del self.matrices[keyword]
        self._cur_matrix = "raw.data"


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
            * ``self.matrices['arcsinh.transformed']``: if jitter == False, np.ndarray.
            * ``self.matrices['arcsinh.jitter']``: if jitter == True, np.ndarray.

        Examples
        --------
        >>> cyto_data.arcsinh_transform()
        """
        if "raw.data" not in self.matrices:
            raise ValueError("raw.data matrix must exist in order to calculate the arcsinh transformed matrix!")

        ctrl_ids = self.feature_metadata["_control_id"].values
        idx = ctrl_ids > 0

        signal = self.matrices["raw.data"]
        control = self.barcode_multiarrays["_controls"][:, ctrl_ids]

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
            self.matrices["arcsinh.transformed"] = arcsinh_mat

        if select:
            self._cur_matrix = "arcsinh.jitter" if jitter else "arcsinh.transformed"
