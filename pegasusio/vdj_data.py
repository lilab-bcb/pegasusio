import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import List, Dict, Union, Set, Tuple

import logging
logger = logging.getLogger(__name__)

import anndata
from pegasusio import UnimodalData
from .views import INDEX, _parse_index, UnimodalDataView


class VDJDataView(UnimodalDataView):
    def __init__(self, vdjdata: "VDJData", barcode_index: List[int], feature_index: List[int], cur_matrix: str):
        super().__init__(vdjdata, barcode_index, feature_index, cur_matrix)
        for keyword in self.parent._uns_keywords:
            self.metadata[keyword] = self.parent.metadata[keyword]
        self.obs # must call .obs in order to initialize self.barcode_metadata
        for chain in self.parent._features[self.metadata["modality"]]:
            keyword = "n" + chain
            if keyword in self.barcode_metadata:
                self.barcode_metadata.drop(columns = keyword, inplace = True)

            pos_arr = []
            for pos, vname in enumerate(self.var_names):
                if vname.startswith(chain):
                    pos_arr.append(pos)
            if len(pos_arr) > 0:
                self.barcode_metadata["n" + chain] = self.X[:, pos_arr].getnnz(axis = 1)

    def __repr__(self) -> str:
        repr_str = "View of VDJData object with n_obs x n_vars = {} x {}".format(self._shape[0], self._shape[1])
        repr_str += "\n    It contains {} matrices: {}".format(len(self.parent.matrices), str(list(self.parent.matrices))[1:-1])
        repr_str += "\n    It currently binds to matrix '{}' as X\n".format(self._cur_matrix) if len(self.parent.matrices) > 0 else "\n    It currently binds to no matrix\n"
        repr_str += "\n    obs: {}".format(str(list(self.barcode_metadata))[1:-1])
        for key in ["var", "obsm", "varm"]:
            repr_str += "\n    {}: {}".format(key, str(list(getattr(self.parent, key).keys()))[1:-1])
        repr_str += "\n    uns: {}".format(str(list(self.metadata))[1:-1])

        return repr_str

    def __getitem__(self, index: INDEX) -> "VDJDataView":
        barcode_index, feature_index = _parse_index(self, index)
        return VDJDataView(self.parent, barcode_index, feature_index, self._cur_matrix)

    def get_chain(self, chain: str) -> pd.DataFrame:
        if chain not in self.var_names:
            raise ValueError("Chain {} is unknown!".format(chain))

        data = {}
        fpos = self.var_names.get_loc(chain)
        for keyword in self.parent._matrix_keywords:
            uns_key = "_" + keyword
            self.select_matrix(keyword)
            if uns_key in self.uns:
                idx = self.X[:, fpos].toarray().ravel()
                data[keyword] = self.uns[uns_key][idx]
            else:
                data[keyword] = self.X[:, fpos].toarray().ravel()
        df = pd.DataFrame(data = data, index = self.obs_names, columns = self.parent._matrix_keywords)

        return df



class VDJData(UnimodalData):
    _matrix_keywords = ["high_confidence", "length", "reads", "umis", "v_gene", "d_gene", "j_gene", "c_gene", "cdr3", "cdr3_nt"]
    _uns_keywords = ["_v_gene", "_d_gene", "_j_gene", "_c_gene", "_cdr3", "_cdr3_nt"]
    _features = {"tcr": ["TRA", "TRB", "TRD", "TRG", "Multi"], "bcr": ["IGK", "IGL", "IGH", "Multi"]}
    _n_contigs = 10


    def __init__(
        self,
        barcode_metadata: Union[dict, pd.DataFrame],
        feature_metadata: Union[dict, pd.DataFrame],
        matrices: Dict[str, csr_matrix],
        metadata: dict,
        cur_matrix: str = "umis",
    ) -> None:
        assert metadata["modality"] in {"tcr", "bcr"}
        super().__init__(barcode_metadata, feature_metadata, matrices, None, None, metadata, cur_matrix)
        for chain in VDJData._features[self.uns["modality"]]:
            pos = self.var_names.get_loc(chain)
            self.obs["n" + chain] = self.X[:, pos:pos+VDJData._n_contigs].getnnz(axis = 1)


    def __repr__(self) -> str:
        return super().__repr__().replace("UnimodalData", "VDJData", 1)


    def get_chain(self, chain: str) -> pd.DataFrame:
        if chain not in self.var_names:
            raise ValueError("Chain {} is unknown!".format(chain))

        data = {}
        fpos = self.var_names.get_loc(chain)
        for keyword in VDJData._matrix_keywords:
            uns_key = "_" + keyword
            if uns_key in self.uns:
                idx = self.matrices[keyword][:, fpos].toarray().ravel()
                data[keyword] = self.uns[uns_key][idx]
            else:
                data[keyword] = self.matrices[keyword][:, fpos].toarray().ravel()
        df = pd.DataFrame(data = data, index = self.obs_names, columns = VDJData._matrix_keywords)

        return df


    def from_anndata(self, data: anndata.AnnData, genome: str = None, modality: str = None) -> None:
        raise ValueError("Cannot convert an AnnData object to a VDJData object!")

    
    def to_anndata(self) -> anndata.AnnData:
        raise ValueError("Cannot convert a VDJData object ot an AnnData object!")


    def __getitem__(self, index: INDEX) -> UnimodalDataView:
        barcode_index, feature_index = _parse_index(self, index)
        return VDJDataView(self, barcode_index, feature_index, self._cur_matrix)


