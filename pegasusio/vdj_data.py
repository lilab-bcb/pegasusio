import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import List, Dict, Union

import anndata
from pegasusio import UnimodalData
from .views import INDEX, CINDEX, _parse_index, UnimodalDataView


class VDJDataView(UnimodalDataView):
    def __init__(self, vdjdata: "VDJData", barcode_index: CINDEX, feature_index: CINDEX, cur_matrix: str):
        super().__init__(vdjdata, barcode_index, feature_index, cur_matrix, obj_name = "VDJData")
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
        return super().__repr__({"obs": str(list(self.barcode_metadata))[1:-1]})

    def __getitem__(self, index: INDEX) -> "VDJDataView":
        barcode_index, feature_index = _parse_index(self, index)
        return VDJDataView(self.parent, barcode_index, feature_index, self._cur_matrix)

    def get_chain(self, chain: str) -> pd.DataFrame:
        if chain not in self.var_names:
            raise ValueError(f"Chain '{chain}' is unknown!")

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
        barcode_multiarrays: Dict[str, np.ndarray] = None,
        feature_multiarrays: Dict[str, np.ndarray] = None,
        cur_matrix: str = "umis",
    ) -> None:
        assert metadata["modality"] in {"tcr", "bcr"}
        super().__init__(barcode_metadata, feature_metadata, matrices, metadata, barcode_multiarrays, feature_multiarrays, cur_matrix)
        for chain in VDJData._features[self.uns["modality"]]:
            pos = self.var_names.get_loc(chain)
            self.obs["n" + chain] = self.X[:, pos:pos+VDJData._n_contigs].getnnz(axis = 1)


    def get_chain(self, chain: str) -> pd.DataFrame:
        if chain not in self.var_names:
            raise ValueError("Chain '{chain}' is unknown!")

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


    def construct_clonotype(self, min_umis: int = 2) -> None:
        if self.get_modality() == "tcr":
            df1 = self.get_chain("TRA")
            df2 = self.get_chain("TRB")
            idx = (df1["umis"] >= min_umis) & (df2["umis"] >= min_umis)
            self.obs["has_clonotype"] = idx
            self.obs["clonotype"] = ""
            self.obs.loc[idx, "clonotype"] = df1.loc[idx, "cdr3"] + "|" + df2.loc[idx, "cdr3"]

            df3 = self.get_chain("TRD")
            df4 = self.get_chain("TRG")
            idx2 = (df3["umis"] >= min_umis) & (df4["umis"] >= min_umis) & (~idx)
            if idx2.sum() > 0:
                self.obs.loc[idx2, "has_clonotype"] = True
                self.obs.loc[idx2, "clonotype"] = df3.loc[idx2, "cdr3"] + "|" + df4.loc[idx2, "cdr3"]
        else:
            assert self.get_modality() == "bcr"
            df1 = self.get_chain("IGH")
            df2 = self.get_chain("IGK")
            df3 = self.get_chain("IGL")
            idx1 = df1["umis"] >= min_umis
            idx2 = (df2["umis"] >= df3["umis"]) & (df2["umis"] >= min_umis)
            idx3 = (df2["umis"] < df3["umis"]) & (df3["umis"] >= min_umis)
            idxkh = idx1 & idx2
            idxlh = idx1 & idx3
            self.obs["has_clonotype"] = idxkh | idxlh
            self.obs["clonotype"] = ""
            self.obs.loc[idxkh, "clonotype"] = df2.loc[idxkh, "cdr3"] + "|" + df1.loc[idxkh, "cdr3"]
            self.obs.loc[idxlh, "clonotype"] = df3.loc[idxlh, "cdr3"] + "|" + df1.loc[idxlh, "cdr3"]


    def from_anndata(self, data: anndata.AnnData, genome: str = None, modality: str = None) -> None:
        raise ValueError("Cannot convert an AnnData object to a VDJData object!")

    
    def to_anndata(self) -> anndata.AnnData:
        raise ValueError("Cannot convert a VDJData object ot an AnnData object!")


    def __getitem__(self, index: INDEX) -> VDJDataView:
        barcode_index, feature_index = _parse_index(self, index)
        return VDJDataView(self, barcode_index, feature_index, self._cur_matrix)
