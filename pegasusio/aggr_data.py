import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack, coo_matrix
from typing import List, Dict, Union
from collections import defaultdict
from pandas.api.types import is_categorical_dtype, is_string_dtype
from natsort import natsorted

import logging
logger = logging.getLogger(__name__)

from pegasusio import timer, run_gc
from pegasusio import UnimodalData, CITESeqData, CytoData, VDJData, MultimodalData


def _fillna(df: pd.DataFrame) -> None:
    """ Fill NA with default values """
    isna = df.isna()
    hasna = isna.sum(axis=0)
    for column in hasna.index[hasna > 0]:
        if is_categorical_dtype(df[column]):
            df[column] = df[column].astype(str)

        default_value = None
        if df[column].dtype.kind == "O" and type(df[column].unique()[0]) == bool:
            default_value = False
        elif df[column].dtype.kind in {"i", "u", "f", "c"}:
            default_value = 0
        elif df[column].dtype.kind == "S":
            default_value = b""
        elif df[column].dtype.kind in {"O", "U"}:
            default_value = ""
        else:
            raise ValueError(f"{column} has unsupported dtype {df[column].dtype}!")

        df.loc[isna[column], column] = default_value

        if type(default_value) == bool:
            df[column] = df[column].astype(bool)

        if df[column].dtype.kind == "f":
            int_type = getattr(np, df[column].dtype.name.replace('float', 'int'))
            values = df[column].values
            int_values = values.astype(int_type)
            if np.array_equal(values, int_values):
                df[column] = int_values


def _check_categorical(df:pd.DataFrame) -> None:
    for col in df.columns:
        if not is_categorical_dtype(df[col]) and is_string_dtype(df[col]):
            keywords = set(df[col])
            if len(keywords) <= df.shape[0] / 10.0: # at least 10x reduction
                df[col] = pd.Categorical(df[col], categories = natsorted(keywords))


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
            metadata[key] = np.array(["None"] + list(values), dtype = object)
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
        _fillna(barcode_metadata)
        _check_categorical(barcode_metadata)

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
        _fillna(feature_metadata)
        _check_categorical(feature_metadata)

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
