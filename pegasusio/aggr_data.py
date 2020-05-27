import numpy as np
import pandas as pd
import warnings
from scipy.sparse import csr_matrix, vstack
from typing import List, Dict, Union, Set, Tuple
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)

from pegasusio import timer, run_gc
from pegasusio import UnimodalData, CITESeqData, CytoData, VDJData, MultimodalData


class AggrData:
    def __init__(self):
        self.aggr = defaultdict(list)


    def add_data(self, data: MultimodalData) -> None:
        for key in data.list_data():
            self.aggr[key] = data.get_data(key)


    def _get_fillna_dict(self, df: pd.DataFrame) -> dict:
        """ Generate a fillna dict for columns in a df """
        fillna_dict = {}
        for column in df:
            fillna_dict[column] = "" if df[column].dtype.kind in {"O", "S"} else 0
        return fillna_dict


    @run_gc
    def _merge_matrices(self, feature_metadata: pd.DataFrame, unlist: List[UnimodalData]) -> Dict[str, csr_matrix]:
        """ After running this function, all matrices in unilist are deleted. """
        matrices_list = defaultdict(list)
        f2idx = pd.Series(data=range(feature_metadata.shape[0]), index=feature_metadata.index)

        for unidata in unilist:
            if feature_metadata.shape[0] > unidata.feature_metadata.shape[0] or (feature_metadata.index != unidata.feature_metadata.index).sum() > 0:
                mat_shape = (unidata.shape[0], f2idx.size)
                for mat_key in unidata.list_keys():
                    unidata.select_matrix(mat_key)
                    mat = csr_matrix(mat_shape, dtype = unidata.X.dtype)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mat[:, f2idx[unidata.feature_metadata.index].values] = unidata.X
                    matrices_list[mat_key].append(mat)
            else:
                for mat_key in unidata.list_keys():
                    unidata.select_matrix(mat_key)
                    matrices_list[mat_key].append(unidata.X)

        matrices = {}
        for mat_key in matrices_list:
            matrices[mat_key] = vstack(matrices_list[mat_key])

        return matrices


    @run_gc
    def _vdj_update_metadata_matrices(self, metadata: dict, matrices: dict, unilist: List[UnimodalData]) -> None:
        for key in VDJData._uns_keywords:
            values = set()
            for unidata in unilist:
                values.update(unidata.metadata[key])
            val2num = dict(zip(values, range(len(values))))
            mat_key = key[1:]
            mat = matrices[mat_key]
            start = end = 0
            for unidata in unilist:
                old2new = np.array([val2num[val] for val in unidata.metadata[key]], dtype = np.int32)
                end = start + unidata.matrices[mat_key].data.size
                mat[start:end] = old2new[mat[start:end]]
                start = end
            metadata[key] = list(values)


    @run_gc
    def _aggregate_unidata(self, unilist: List[UnimodalData]) -> UnimodalData:
        if len(unilist) == 1:
            return unilist[0]

        barcode_metadata_dfs = [unidata.barcode_metadata for unidata in unilist]
        barcode_metadata = pd.concat(barcode_metadata_dfs, axis=0, sort=False, copy=False)
        fillna_dict = self._get_fillna_dict(barcode_metadata)
        barcode_metadata.fillna(value=fillna_dict, inplace=True)

        feature_metadata = unilist[0].feature_metadata
        for other in unilist[1:]:
            keys = ["featurekey"] + feature_metadata.columns.intersection(other.feature_metadata.columns).values.tolist()
            feature_metadata = feature_metadata.merge(other.feature_metadata, on=keys, how="outer", sort=False, copy=False)  # If sort is True, feature keys will be changed even if all channels share the same feature keys.
        fillna_dict = self._get_fillna_dict(feature_metadata)
        feature_metadata.fillna(value=fillna_dict, inplace=True)

        matrices = self._merge_matrices(feature_metadata, unilist)
        metadata = unilist[0].metadata.mapping

        unidata = None
        if isinstance(unilist[0], CITESeqData):
            metadata["_control_counts"] = csr_matrix((barcode_metadata.shape[0], 1), dtype = np.int32)
            unidata = CITESeqData(barcode_metadata, feature_metadata, matrices, metadata)
        elif isinstance(unilist[0], CytoData):
            barcode_multiarrays = {"_controls": np.zeros((barcode_metadata.shape[0], 1), dtype = np.int32)}
            unidata = CytoData(barcode_metadata, feature_metadata, matrices, metadata, barcode_multiarrays = barcode_multiarrays)
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
