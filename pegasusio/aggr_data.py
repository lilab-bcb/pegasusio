import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack, coo_matrix
from typing import List, Dict, Union
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)

from pegasusio import timer, run_gc
from pegasusio import UnimodalData, CITESeqData, CytoData, VDJData, MultimodalData


def _get_fillna_dict(df: pd.DataFrame) -> dict:
    """ Generate a fillna dict for columns in a df """
    fillna_dict = {}
    for column in df:
        if df[column].dtype.kind == "b":
            fillna_dict[column] = False
        elif df[column].dtype.kind in {"i", "u", "f", "c"}:
            fillna_dict[column] = 0
        elif df[column].dtype.kind == "S":
            fillna_dict[column] = b""
        elif df[column].dtype.kind in {"O", "U"}:
            fillna_dict[column] = ""
        else:
            raise ValueError(f"{column} has unsupported dtype {df[column].dtype}!")

    return fillna_dict


class AggrData:
    def __init__(self):
        self.aggr = defaultdict(list)


    def add_data(self, data: MultimodalData) -> None:
        for key in data.list_data():
            self.aggr[key].append(data.get_data(key))


    @run_gc
    def _merge_matrices(self, feature_metadata: pd.DataFrame, unilist: List[UnimodalData], modality: str) -> Dict[str, csr_matrix]:
        """ Merge all matrices together """
        f2idx = pd.Series(data=range(feature_metadata.shape[0]), index=feature_metadata.index)

        mat_keys = set()
        no_reorg = True
        for unidata in unilist:
            mat_keys.update(unidata.matrices)
            if no_reorg and (feature_metadata.shape[0] > unidata.feature_metadata.shape[0] or (feature_metadata.index != unidata.feature_metadata.index).sum() > 0):
                no_reorg = False

        if modality == "bcr" or modality == "tcr":
            for key in VDJData._uns_keywords:
                mat_keys.discard(key[1:])


        matrices = {}
        if no_reorg:
            for mat_key in mat_keys:
                mat_list = []
                for unidata in unilist:
                    mat = unidata.matrices.pop(mat_key, None)
                    if mat is not None:
                        mat_list.append(mat)
                matrices[mat_key] = vstack(mat_list) if modality != "cyto" else np.vstack(mat_list)
        else:
            colmap = []
            for unidata in unilist:
                colmap.append(f2idx[unidata.feature_metadata.index].values)

            if modality == "cyto":
                # matrices are dense.
                for mat_key in mat_keys:
                    mat_list = []
                    for i, unidata in enumerate(unilist):
                        mat = unidata.matrices.pop(mat_key, None)
                        if mat is not None:
                            newmat = np.zeros((mat.shape[0], feature_metadata.shape[0]), dtype = mat.dtype)
                            newmat[:, colmap[i]] = mat
                            mat_list.append(newmat)
                    matrices[mat_key] = np.vstack(mat_list)
            else:
                for mat_key in mat_keys:
                    data_list = []
                    row_list = []
                    col_list = []
                    row_base = 0
                    for i, unidata in enumerate(unilist):
                        mat = unidata.matrices.pop(mat_key, None)
                        if mat is not None:
                            mat = mat.tocoo(copy = False) # convert to coo format
                            data_list.append(mat.data)
                            row_list.append(mat.row + row_base)
                            col_list.append(colmap[i][mat.col])
                            row_base += mat.shape[0]
                    data = np.concatenate(data_list)
                    row = np.concatenate(row_list)
                    col = np.concatenate(col_list)
                    matrices[mat_key] = coo_matrix((data, (row, col)), shape=(row_base, feature_metadata.shape[0])).tocsr(copy = False)

        return matrices


    @run_gc
    def _vdj_update_metadata_matrices(self, metadata: dict, matrices: dict, unilist: List[UnimodalData]) -> None:
        metadata.pop("uns_dict", None)
        for key in VDJData._uns_keywords:
            values = set()
            for unidata in unilist:
                values.update(unidata.metadata[key])
            values.discard("None") # None must be 0 for sparse matrix
            metadata[key] = np.array(["None"] + list(values), dtype = np.object)
            val2num = dict(zip(metadata[key], range(metadata[key].size)))

            mat_key = key[1:]
            mat_list = []
            for unidata in unilist:
                mat = unidata.matrices.pop(mat_key)
                old2new = np.array([val2num[val] for val in unidata.metadata[key]], dtype = np.int32)
                mat.data = old2new[mat.data]
                mat_list.append(mat)
            matrices[mat_key] = vstack(mat_list)


    @run_gc
    def _aggregate_unidata(self, unilist: List[UnimodalData]) -> UnimodalData:
        if len(unilist) == 1:
            del unilist[0].metadata["_sample"]
            return unilist[0]

        modality = unilist[0].get_modality()

        barcode_metadata_dfs = [unidata.barcode_metadata for unidata in unilist]
        barcode_metadata = pd.concat(barcode_metadata_dfs, axis=0, sort=False, copy=False)
        fillna_dict = _get_fillna_dict(barcode_metadata)
        barcode_metadata.fillna(value=fillna_dict, inplace=True)


        var_dict = {}
        for unidata in unilist:
            idx = unidata.feature_metadata.columns.difference(["featureid"])
            if idx.size > 0:
                var_dict[unidata.metadata["_sample"]] = unidata.feature_metadata[idx]
                unidata.feature_metadata.drop(columns = idx, inplace = True)

        feature_metadata = unilist[0].feature_metadata
        for other in unilist[1:]:
            keys = ["featurekey"] + feature_metadata.columns.intersection(other.feature_metadata.columns).values.tolist()
            feature_metadata = feature_metadata.merge(other.feature_metadata, on=keys, how="outer", sort=False, copy=False)  # If sort is True, feature keys will be changed even if all channels share the same feature keys.
        fillna_dict = _get_fillna_dict(feature_metadata)
        feature_metadata.fillna(value=fillna_dict, inplace=True)


        matrices = self._merge_matrices(feature_metadata, unilist, modality)


        uns_dict = {}
        metadata = {"genome": unilist[0].metadata["genome"], "modality": unilist[0].metadata["modality"]}
        for unidata in unilist:
            assert unidata.metadata.pop("genome") == metadata["genome"]
            assert unidata.metadata.pop("modality") == metadata["modality"]
            if modality == "citeseq":
                for key in CITESeqData._uns_keywords:
                    unidata.metadata.pop(key, None)
                unidata.metadata.pop("_obs_keys", None)
            elif modality == "cyto":
                for key in CytoData._uns_keywords:
                    unidata.metadata.pop(key, None)
            sample_name = unidata.metadata.pop("_sample")
            if len(unidata.metadata) > 0:
                uns_dict[sample_name] = unidata.metadata.mapping

        if len(var_dict) > 0:
            metadata["var_dict"] = var_dict
        if len(uns_dict) > 0:
            metadata["uns_dict"] = uns_dict


        unidata = None
        if isinstance(unilist[0], CITESeqData):
            unidata = CITESeqData(barcode_metadata, feature_metadata, matrices, metadata)
        elif isinstance(unilist[0], CytoData):
            unidata = CytoData(barcode_metadata, feature_metadata, matrices, metadata)
        elif isinstance(unilist[0], VDJData):
            self._vdj_update_metadata_matrices(metadata, matrices, unilist)
            unidata = VDJData(barcode_metadata, feature_metadata, matrices, metadata)
        else:
            unidata = UnimodalData(barcode_metadata, feature_metadata, matrices, metadata)

        return unidata


    @timer(logger=logger)
    def aggregate(self) -> MultimodalData:
        """ Aggregate all data together """
        data = MultimodalData()
        for key in list(self.aggr):
            unidata = self._aggregate_unidata(self.aggr.pop(key))
            data.add_data(unidata)
        return data
