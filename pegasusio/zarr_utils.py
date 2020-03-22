# This code is inspired by HCA zarr python codes and AnnData zarr codes

import os
import shutil
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_string_dtype, is_scalar, is_dict_like
from scipy.sparse import csr_matrix, issparse
import zarr
from zarr import Blosc
from natsort import natsorted
from typing import List, Dict, Tuple, Union

from pegasusio import UnimodalData, MultimodalData



CHUNKSIZE = 1000000
COMPRESSOR = Blosc(cname = 'lz4', clevel = 5)


def calc_chunk(shape: tuple) -> tuple:
    ndim = len(shape)
    chunks = [0] * ndim
    ords = np.argsort(shape)
    chunk_size = CHUNKSIZE * 1.0
    chunk_value = -1
    for i, idx in enumerate(ords):
        if chunk_value < 0:
            if shape[idx] ** (ndim - i) < chunk_size:
                chunks[idx] = shape[idx]
                chunk_size /= chunks[idx]
            else:
                chunk_value = int(np.ceil(chunk_size ** (1.0 / (ndim - i))))
                chunks[idx] = chunk_value
        else:
            chunks[idx] = chunk_value

    return tuple(chunks)



class ZarrFile:
    def __init__(self, path: str, mode: str = 'r', storage_type: str = None) -> None:
        """ Initialize a Zarr file, if mode == 'w', create an empty one, otherwise, load from path
        path : `str`, path for the zarr object.
        storage_type : `str`, currently only support 'ZipStore' and 'NestedDirectoryStore'. If None, use 'NestedDirectoryStore' by default.
        """
        self.store = self.root = None

        if storage_type is None:
            storage_type = 'NestedDirectoryStore'

        if mode == 'w':
            # Create a new zarr file
            if storage_type == 'ZipStore':
                if os.path.isdir(path):
                    shutil.rmtree(path)
                self.store = zarr.ZipStore(path, mode = 'w')
            else:
                if os.path.isfile(path):
                    os.unlink(path)
                zarr.NestedDirectoryStore(path)
            self.root = zarr.group(self.store, overwrite = True)
            self.root.attrs['store'] = storage_type
        else:
            # Load existing zarr file
            try:
                self.root = zarr.open(path, mode = mode)
            except ValueError:
                self.store = zarr.ZipStore(path, mode = mode)
                self.root = zarr.open_group(self.store, mode = mode)


    def __del__(self):
        if self.store is not None and hasattr(self.store, 'close'):
            self.store.close()



    def read_csr(self, group: zarr.Group) -> csr_matrix:
        return csr_matrix((group['data'], group['indices'], group['indptr']), shape = group.attrs['shape'])


    def read_series(self, group: zarr.Group, name: str) -> Union[pd.Categorical, np.recarray]:
        if 'ordered' in group[name].attrs:
            # categorical column
            return pd.Categorical.from_codes(group[name][...], categories = group['_categories/{0}'.format(name)][...], ordered = group[name].attrs['ordered'])
        else:
            return group[name][...]


    def read_dataframe(self, group: zarr.Group) -> Union[pd.DataFrame, np.ndarray]:
        if group.attrs['data_type'] == 'data_frame':
            columns = [col for col in group.array_keys() if col != '_index']
            df = pd.DataFrame(data = {col: self.read_series(group, col) for col in columns},
                index = pd.Index(self.read_series(group, '_index'), name = group.attrs['index_name']),
                columns = columns)
            return df
        else:
            array = np.rec.fromarrays([self.read_series(group, col) for col in group.array_keys()],
                names = group.attrs['columns'])
            return array


    def read_array(self, group: zarr.Group, name: str) -> Union[np.ndarray, np.recarray]:
        if name in group.group_keys():
            return self.read_dataframe(group[name])
        else:
            return group[name][...]


    def read_mapping(self, group: zarr.Group) -> dict:
        res_dict = {}

        if 'scalar' in group.attrs:
            res_dict.update(group.attrs['scalar'])
        
        for key in group.array_keys():
            res_dict[key] = self.read_array(group, key)
        
        for key in group.group_keys():
            sub_group = group[key]
            data_type = sub_group.attrs['data_type']
            value = None
            if data_type == 'data_frame' or data_type == 'record_array':
                value = self.read_dataframe(sub_group)
            elif data_type == 'csr_matrix':
                value = self.read_csr(sub_group)
            else:
                assert data_type == 'dict'
                value = self.read_mapping(sub_group)
            res_dict[key] = value
        
        return res_dict


    def read_unimodal_data(self, group: zarr.Group, ngene: int = None, select_singlets: bool = False) -> UnimodalData:
        """ Read UnimodalData
            ngene: filter barcodes with < ngene
            select_singlets: only select singlets
            The above two option only works if experiment_type == "rna"
        """
        unidata = UnimodalData(barcode_metadata = self.read_dataframe(group['barcode_metadata']),
                            feature_metadata = self.read_dataframe(group['feature_metadata']),
                            matrices = self.read_mapping(group['matrices']),
                            barcode_multiarrays = self.read_mapping(group['barcode_multiarrays']),
                            feature_multiarrays = self.read_mapping(group['feature_multiarrays']),
                            metadata = self.read_mapping(group['metadata']))

        assert "genome" in unidata.metadata
        assert "experiment_type" in unidata.metadata
        
        if unidata.metadata["experiment_type"] == "rna":
            unidata.filter(ngene, select_singlets)

        return unidata


    def read_multimodal_data(self, ngene: int = None, select_singlets: bool = False) -> MultimodalData:
        """ Read MultimodalData
            ngene: filter barcodes with < ngene
            select_singlets: only select singlets
            The above two option only works for dataset with experiment_type == "rna"
        """
        data = MultimodalData()
        need_trim = (ngene is not None) or select_singlets
        selected_barcodes = None

        for key, group in self.root.groups():
            unidata = self.read_unimodal_data(group, ngene, select_singlets)
            if need_trim and unidata.uns.get("experiment_type", "rna") == "rna":
                selected_barcodes = unidata.obs_names if selected_barcodes is None else selected_barcodes.union(unidata.obs_names)
            data.add_data(key, unidata)

        if need_trim:
            for key in data.list_data():
                unidata = data.get_data(key)
                if unidata.uns.get("experiment_type", "rna") != "rna":
                    selected = unidata.obs_names.isin(selected_barcodes)
                    unidata.trim(selected)

        return data



    def write_csr(self, group: zarr.Group, name: str, matrix: csr_matrix) -> None:
        sub_group = group.create_group(name, overwrite = True)
        sub_group.attrs.update(data_type = 'csr_matrix', shape = matrix.shape)
        sub_group.create_dataset('data', data = matrix.data, shape = matrix.data.shape, chunks = calc_chunk(matrix.data.shape), dtype = matrix.data.dtype, compressor = COMPRESSOR, overwrite = True)
        sub_group.create_dataset('indices', data = matrix.indices, shape = matrix.indices.shape, chunks = calc_chunk(matrix.indices.shape), dtype = matrix.indices.dtype, compressor = COMPRESSOR, overwrite = True)
        sub_group.create_dataset('indptr', data = matrix.indptr, shape = matrix.indptr.shape, chunks = calc_chunk(matrix.indptr.shape), dtype = matrix.indptr.dtype, compressor = COMPRESSOR, overwrite = True)


    def write_series(self, group: zarr.Group, name: str, array: np.ndarray, data_type: str) -> None:
        if data_type == 'data_frame':
            if is_string_dtype(array):
                keywords = np.unique(array)
                if keywords.size <= array.size / 2.0: # at least half
                    array = pd.Categorical(array, categories = natsorted(keywords))
            if is_categorical_dtype(array):
                # write category keys
                categories = group.require_group('_categories')
                values = array.categories.values
                if isinstance(values[0], bytes):
                    values = np.array([x.decode() for x in values], dtype = object)
                dtype = str if values.dtype.kind == 'O' else values.dtype
                categories.create_dataset(name, data = values, shape = values.shape, chunks = calc_chunk(values.shape), dtype = dtype, compressor = COMPRESSOR, overwrite = True)
                # write codes
                codes_arr = group.create_dataset(name, data = array.codes, shape = array.codes.shape, chunks = calc_chunk(array.codes.shape), dtype = array.codes.dtype, compressor = COMPRESSOR, overwrite = True)
                codes_arr.attrs['ordered'] = array.ordered
                return None

        dtype = str if array.dtype.kind == 'O' else array.dtype
        group.create_dataset(name, data = array, shape = array.shape, chunks = calc_chunk(array.shape), dtype = dtype, compressor = COMPRESSOR, overwrite = True)


    def write_dataframe(self, group: zarr.Group, name: str, df: Union[pd.DataFrame, np.recarray]) -> None:
        data_type = 'data_frame' if isinstance(df, pd.DataFrame) else 'record_array'

        sub_group = group.create_group(name, overwrite = True) 
        attrs_dict = {'data_type' : data_type}
        cols = list(df.columns if data_type == 'data_frame' else df.dtype.names)
        if data_type == 'data_frame':
            attrs_dict['index_name'] = df.index.name if df.index.name is not None else 'index'
            sub_group.create_group('_categories', overwrite = True) # create a group to store category keys for catigorical columns
            self.write_series(sub_group, '_index', df.index.values, data_type)

        for col in cols:
            self.write_series(sub_group, col, (df[col].values if data_type == 'data_frame' else df[col]), data_type)

        sub_group.attrs.update(**attrs_dict)


    def write_array(self, group: zarr.Group, name: str, array: np.ndarray) -> None:
        if array.dtype.kind == 'V':
            self.write_dataframe(group, name, array)
        else:
            dtype = str if array.dtype.kind == 'O' else array.dtype
            group.create_dataset(name, data = array, shape = array.shape, chunks = calc_chunk(array.shape), dtype = dtype, compressor = COMPRESSOR, overwrite = True)


    def write_mapping(self, group: zarr.Group, name: str, mapping: dict) -> None:
        sub_group = group.create_group(name, overwrite = True)
        scalar_dict = {}

        for key, value in mapping.items():
            if is_scalar(value):
                scalar_dict[key] = value
            elif isinstance(value, np.ndarray):
                self.write_array(sub_group, key, value)
            elif isinstance(value, pd.DataFrame):
                self.write_dataframe(sub_group, key, value)
            elif is_dict_like(value):
                self.write_mapping(sub_group, key, value)
            elif issparse(value):
                assert isinstance(value, csr_matrix)
                self.write_csr(sub_group, key, value)
            else:
                # assume value is either list or tuple, converting it to np.ndarray
                self.write_array(sub_group, key, np.array(value))

        attrs_dict = {'data_type' : 'dict'}
        if len(scalar_dict) > 0:
            attrs_dict['scalar'] = scalar_dict
        sub_group.attrs.update(**attrs_dict)



    def write_unimodal_data(self, group: zarr.Group, name: str, data: UnimodalData) -> None:
        """ Write UnimodalData
        """
        sub_group = group.create_group(name, overwrite = True)
        sub_group.attrs['data_type'] = 'UnimodalData'

        self.write_dataframe(sub_group, 'barcode_metadata', data.barcode_metadata)
        self.write_dataframe(sub_group, 'feature_metadata', data.feature_metadata)
        self.write_mapping(sub_group, 'matrices', data.matrices)
        self.write_mapping(sub_group, 'barcode_multiarrays', data.barcode_multiarrays)
        self.write_mapping(sub_group, 'feature_multiarrays', data.feature_multiarrays)
        self.write_mapping(sub_group, 'metadata', data.metadata)


    def write_multimodal_data(self, data: MultimodalData) -> None:
        """ Write MultimodalData
        """
        for key in data.list_data():
            unidata = data.get_data(key)
            self.write_unimodal_data(self.root, key, unidata)
