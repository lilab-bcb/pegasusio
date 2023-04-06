# This code is inspired by HCA zarr python codes and AnnData zarr codes

import os
import shutil
import PIL
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_string_dtype, is_scalar, is_dict_like
from scipy.sparse import csr_matrix, issparse
import zarr
from zarr import Blosc
from natsort import natsorted
from typing import List, Dict, Tuple, Union

import logging
logger = logging.getLogger(__name__)

from pegasusio import modalities
from pegasusio import UnimodalData, VDJData, CITESeqData, CytoData, MultimodalData, SpatialData


CHUNKSIZE = 1000000.0
COMPRESSOR = Blosc(cname = 'lz4', clevel = 5)


def calc_chunk(shape: tuple) -> tuple:
    ndim = len(shape)
    chunks = [1] * ndim
    ords = np.argsort(shape)
    if shape[ords[0]] > 0:
        chunk_size = CHUNKSIZE
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


def check_and_remove_existing_path(path: str) -> None:
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
            logger.warning(f"Detected and removed pre-existing directory {path}.")
        else:
            assert os.path.isfile(path)
            os.unlink(path)
            logger.warning(f"Detected and removed pre-existing file {path}.")


class ZarrFile:
    def __init__(self, path: str, mode: str = 'r', storage_type: str = None) -> None:
        """ Initialize a Zarr file, if mode == 'w', create an empty one, otherwise, load from path
        path : `str`, path for the zarr object.
        storage_type : `str`, currently only support 'ZipStore' and 'NestedDirectoryStore'. If None, use 'NestedDirectoryStore' by default.
        """
        self.store = self.root = None
        self.write_empty_chunks = False

        if storage_type is None:
            storage_type = 'NestedDirectoryStore'

        if mode == 'w':
            # Create a new zarr file
            check_and_remove_existing_path(path)
            self.store = zarr.ZipStore(path, mode = 'w') if storage_type == 'ZipStore' else zarr.NestedDirectoryStore(path)
            self.root = zarr.group(self.store, overwrite = True)
            self.write_empty_chunks = (storage_type == 'ZipStore')
        else:
            # Load existing zarr file
            self.store = zarr.NestedDirectoryStore(path) if os.path.isdir(path) else zarr.ZipStore(path, mode = 'r')
            if mode == 'a' and isinstance(self.store, zarr.ZipStore):
                self._to_directory()
            self.root = zarr.open(self.store, mode = mode)


    def __del__(self):
        if self.store is not None and hasattr(self.store, 'close'):
            self.store.close()


    def _to_zip(self):
        if not isinstance(self.store, zarr.ZipStore):
            zip_path = self.store.path + '.zip'
            zip_store = zarr.ZipStore(zip_path, mode = 'w')
            zarr.copy_store(self.store, zip_store)
            zip_store.close()

            shutil.rmtree(self.store.path)

            self.store = zarr.ZipStore(zip_path, mode = 'r')
            self.root = zarr.open_group(self.store, mode = 'r')


    def _to_directory(self):
        orig_path = self.store.path

        if not orig_path.endswith('.zip'):
            self.store.close()
            zip_path = orig_path + '.zip'
            check_and_remove_existing_path(zip_path)
            os.replace(orig_path, zip_path)
            self.store = zarr.ZipStore(zip_path, mode = 'r')
        else:
            zip_path = orig_path

        dest_path = zip_path[:-4]
        check_and_remove_existing_path(dest_path)
        dir_store = zarr.NestedDirectoryStore(dest_path)
        zarr.copy_store(self.store, dir_store)
        self.store.close()
        os.remove(zip_path)

        self.store = dir_store
        self.root = zarr.open_group(self.store)

        logger.info(f"Converted ZipStore zarr file {orig_path} to NestedDirectoryStore {dest_path}.")

    def read_csr(self, group: zarr.Group) -> csr_matrix:
        return csr_matrix((group['data'][...], group['indices'][...], group['indptr'][...]), shape = group.attrs['shape'])

    def read_series(self, group: zarr.Group, name: str) -> Union[pd.Categorical, np.ndarray]:
        if 'ordered' in group[name].attrs:
            # categorical column
            return pd.Categorical.from_codes(group[name][...], categories = group[f'_categories/{name}'][...], ordered = group[name].attrs['ordered'])
        else:
            if isinstance(group[name], zarr.hierarchy.Group):
                ll = []
                for data in group[name].arrays():
                    ll.append(PIL.Image.fromarray(data[1][...]))
                return ll
            else:
                return group[name][...]

    def read_dataframe(self, group: zarr.Group) -> pd.DataFrame:
        columns = group.attrs.get('columns', None)
        if columns is None:
            columns = [col for col in group.array_keys() if col != '_index']
        data = {col: self.read_series(group, col) for col in columns}
        _index = self.read_series(group, '_index')
        index = pd.Index(_index, name = group.attrs['index_name'], dtype = _index.dtype)
        df = pd.DataFrame(data = data, index = index) # if add columns = columns, the generation will be slow
        return df

    def read_array(self, group: zarr.Group, name: str) -> np.ndarray:
        return group[name][...]

    def read_record_array(self, group: zarr.Group) -> np.recarray:
        columns = group.attrs.get('columns', None)
        if columns is None:
            columns = [col for col in group.array_keys()]

        array = np.rec.fromarrays([group[col][...] for col in columns], names = columns)
        return array


    def read_composite_list(self, group: zarr.Group) -> list:
        assert '_size' in group.attrs
        res_list = [None] * group.attrs['_size']

        if 'scalar' in group.attrs:
            scalar_dict = group.attrs['scalar']
            for i, value in scalar_dict.items():
                res_list[int(i)] = value

        for key in group. array_keys():
            res_list[int(key)] = self.read_array(group, key)

        for key in group.group_keys():
            sub_group = group[key]
            data_type = sub_group.attrs['data_type']
            value = None
            if data_type == 'data_frame':
                value = self.read_dataframe(sub_group)
            elif data_type == 'record_array':
                value = self.read_record_array(sub_group)
            elif data_type == 'csr_matrix':
                value = self.read_csr(sub_group)
            elif data_type == 'dict':
                value = self.read_mapping(sub_group)
            else:
                assert data_type == 'composite_list'
                value = self.read_composite_list(sub_group)
            res_list[int(key)] = value

        return res_list


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
            if data_type == 'data_frame':
                value = self.read_dataframe(sub_group)
            elif data_type == 'record_array':
                value = self.read_record_array(sub_group)
            elif data_type == 'csr_matrix':
                value = self.read_csr(sub_group)
            elif data_type == 'dict':
                value = self.read_mapping(sub_group)
            else:
                assert data_type == 'composite_list'
                value = self.read_composite_list(sub_group)
            res_dict[key] = value

        return res_dict


    def read_unimodal_data(self, group: zarr.Group) -> UnimodalData:
        """ Read UnimodalData
        """
        metadata = self.read_mapping(group['metadata'])
        assert 'genome' in metadata
        if 'modality' not in metadata:
            assert metadata.get('experiment_type', 'none') in modalities
            metadata['modality'] = metadata.pop('experiment_type')

        DataClass = UnimodalData
        modality = metadata['modality']
        if modality == 'tcr' or modality == 'bcr':
            DataClass = VDJData
        elif modality == 'citeseq':
            DataClass = CITESeqData
        elif modality == 'cyto':
            DataClass = CytoData
        elif modality == 'visium':
            DataClass = SpatialData

        unidata = DataClass(
            barcode_metadata=self.read_dataframe(group["barcode_metadata"]),
            feature_metadata=self.read_dataframe(group["feature_metadata"]),
            matrices=self.read_mapping(group["matrices"]),
            metadata=metadata,
            barcode_multiarrays=self.read_mapping(group["barcode_multiarrays"]),
            feature_multiarrays=self.read_mapping(group["feature_multiarrays"]),
            barcode_multigraphs=self.read_mapping(group["barcode_multigraphs"])
            if "barcode_multigraphs" in group
            else dict(),  # for backward-compatibility
            feature_multigraphs=self.read_mapping(group["feature_multigraphs"])
            if "feature_multigraphs" in group
            else dict(),
        )
        if isinstance (unidata, SpatialData):
            unidata.image_metadata = self.read_dataframe(group["image_metadata"]) if "image_metadata" in group else dict()

        if group.attrs.get("_cur_matrix", None) is not None:
            unidata.select_matrix(group.attrs["_cur_matrix"])

        return unidata


    def read_multimodal_data(self, attach_zarrobj = False) -> MultimodalData:
        """ Read MultimodalData
        """
        data = MultimodalData()
        for key, group in self.root.groups():
            unidata = self.read_unimodal_data(group)
            data.add_data(unidata)

        if self.root.attrs.get('_selected', None) is not None:
            data.select_data(self.root.attrs['_selected'])

        if attach_zarrobj:
            data._zarrobj = self

        return data


    def write_csr(self, parent: zarr.Group, name: str, matrix: csr_matrix) -> None:
        group = parent.create_group(name, overwrite = True)
        group.attrs.update(data_type = 'csr_matrix', shape = matrix.shape)
        group.create_dataset('data', data = matrix.data, shape = matrix.data.shape, chunks = calc_chunk(matrix.data.shape), dtype = matrix.data.dtype, compressor = COMPRESSOR, overwrite = True, write_empty_chunks = self.write_empty_chunks)
        group.create_dataset('indices', data = matrix.indices, shape = matrix.indices.shape, chunks = calc_chunk(matrix.indices.shape), dtype = matrix.indices.dtype, compressor = COMPRESSOR, overwrite = True, write_empty_chunks = self.write_empty_chunks)
        group.create_dataset('indptr', data = matrix.indptr, shape = matrix.indptr.shape, chunks = calc_chunk(matrix.indptr.shape), dtype = matrix.indptr.dtype, compressor = COMPRESSOR, overwrite = True, write_empty_chunks = self.write_empty_chunks)


    def write_series(self, group: zarr.Group, name: str, array: np.ndarray) -> None:
        if not is_categorical_dtype(array) and name != '_index' and is_string_dtype(array):
            keywords = set(array)
            if len(keywords) <= array.size / 10.0: # at least 10x reduction
                array = pd.Categorical(array, categories = natsorted(keywords))

        if is_categorical_dtype(array):
            # write category keys
            categories = group.require_group('_categories', overwrite = False)
            values = array.categories.values
            if isinstance(values[0], bytes):
                values = np.array([x.decode() for x in values], dtype = object)
            self.write_array(categories, name, values)
            # write codes
            codes_arr = group.create_dataset(name, data = array.codes, shape = array.codes.shape, chunks = calc_chunk(array.codes.shape), dtype = array.codes.dtype, compressor = COMPRESSOR, overwrite = True, write_empty_chunks = self.write_empty_chunks)
            codes_arr.attrs['ordered'] = bool(array.ordered)
        else:
            self.write_array(group, name, array)


    def write_dataframe(self, parent: zarr.Group, name: str, df: pd.DataFrame) -> None:
        group = parent.create_group(name, overwrite = True)
        attrs_dict = {'data_type' : 'data_frame', 'columns' : list(df.columns)}
        attrs_dict['index_name'] = df.index.name if df.index.name is not None else 'index'
        self.write_series(group, '_index', df.index.values)
        for col in df.columns:
            if df[col].size > 0 and isinstance(df[col].values[0], PIL.Image.Image):
                colgroup = group.create_group(col, overwrite = True)
                x = 0
                for data in df[col].values:
                    x = x+1
                    self.write_series(colgroup, col + str(x), np.array(data))
            else:
                self.write_series(group, col, df[col].values)
        group.attrs.update(**attrs_dict)

    def write_array(self, group: zarr.Group, name: str, array: np.ndarray) -> None:
        dtype = str if array.dtype.kind == 'O' else array.dtype
        group.create_dataset(name, data = array, shape = array.shape, chunks = calc_chunk(array.shape), dtype = dtype, compressor = COMPRESSOR, overwrite = True, write_empty_chunks = self.write_empty_chunks)

    def write_record_array(self, parent: zarr.Group, name: str, array: np.recarray) -> None:
        group = parent.create_group(name, overwrite = True)
        attrs_dict = {'data_type' : 'record_array', 'columns' : list(array.dtype.names)}
        for col in array.dtype.names:
            self.write_array(group, col, array[col])
        group.attrs.update(**attrs_dict)


    def write_composite_list(self, parent: zarr.Group, name: str, values: Union[list, tuple]) -> None:
        group = parent.create_group(name, overwrite = True)
        scalar_dict = {}

        for i, value in enumerate(values):
            if is_scalar(value):
                if type(value).__module__ == 'numpy':
                    value = value.item()
                scalar_dict[str(i)] = value
            elif isinstance(value, np.ndarray):
                if value.dtype.kind == 'V':
                    self.write_record_array(group, str(i), value)
                else:
                    self.write_array(group, str(i), value)
            elif isinstance(value, pd.DataFrame):
                self.write_dataframe(group, str(i), value)
            elif is_dict_like(value):
                self.write_mapping(group, str(i), value)
            elif issparse(value):
                assert isinstance(value, csr_matrix)
                self.write_csr(group, str(i), value)
            else:
                # assume value is either list or tuple
                assert type(value) in {list, tuple}
                is_comp_list = sum([is_scalar(x) for x in value]) < len(value)
                if is_comp_list:
                    self.write_composite_list(group, str(i), value)
                else:
                    # converting it to np.ndarray
                    self.write_array(group, str(i), value.astype(str) if is_categorical_dtype(value) else np.array(value))

        attrs_dict = {'data_type': 'composite_list', '_size': len(values)}
        if len(scalar_dict) > 0:
            attrs_dict['scalar'] = scalar_dict
        group.attrs.update(**attrs_dict)


    def write_mapping(self, parent: zarr.Group, name: str, mapping: dict, overwrite = True) -> None:
        group = None
        if overwrite:
            group = parent.create_group(name, overwrite = True)
        else:
            group = parent.require_group(name, overwrite = False) # throw an error if name in array_keys()

        scalar_dict = group.attrs.pop('scalar', {})

        def _write_one_pair(key, value):
            if is_scalar(value):
                if type(value).__module__ == 'numpy':
                    value = value.item()
                scalar_dict[key] = value
            elif isinstance(value, np.ndarray):
                if value.dtype.kind == 'V':
                    self.write_record_array(group, key, value)
                else:
                    self.write_array(group, key, value)
            elif isinstance(value, pd.DataFrame):
                self.write_dataframe(group, key, value)
            elif is_dict_like(value):
                self.write_mapping(group, key, value)
            elif issparse(value):
                assert isinstance(value, csr_matrix)
                self.write_csr(group, key, value)
            else:
                # assume value is either list or tuple
                assert type(value) in {list, tuple}
                is_comp_list = sum([is_scalar(x) for x in value]) < len(value)
                if is_comp_list:
                    self.write_composite_list(group, key, value)
                else:
                    # converting it to np.ndarray
                    self.write_array(group, key, value.astype(str) if is_categorical_dtype(value) else np.array(value))

        if overwrite:
            for key, value in mapping.items():
                _write_one_pair(key, value)
        else:
            for key in mapping.deleted:
                if key in scalar_dict:
                    del scalar_dict[key]
                else:
                    del group[key]
            for key in mapping.modified:
                _write_one_pair(key, mapping[key])

        attrs_dict = {'data_type' : 'dict'}
        if len(scalar_dict) > 0:
            attrs_dict['scalar'] = scalar_dict
        group.attrs.update(**attrs_dict)


    def write_unimodal_data(self, parent: zarr.Group, name: str, data: UnimodalData, overwrite: bool = True) -> None:
        """ Write UnimodalData
            overwrite means if overwrite the whole unimodal data; meaning is different from require_group overwrite --- if the name in array_keys(), take it and make it as a group
        """
        group = None
        if overwrite:
            group = parent.create_group(name, overwrite = True)
        else:
            group = parent.require_group(name, overwrite = False) # throw an error if name in array_keys()

        attrs_dict = {'data_type': 'UnimodalData', '_cur_matrix': data.current_matrix()}
        group.attrs.update(**attrs_dict)

        self.write_dataframe(group, 'barcode_metadata', data.barcode_metadata)
        self.write_dataframe(group, 'feature_metadata', data.feature_metadata)

        if hasattr(data, 'img'):
            self.write_dataframe(group, 'image_metadata', data.image_metadata)

        if overwrite or data.matrices.is_dirty():
            self.write_mapping(group, 'matrices', data.matrices, overwrite = overwrite)
        if overwrite or data.metadata.is_dirty():
            self.write_mapping(group, 'metadata', data.metadata, overwrite = overwrite)
        if overwrite or data.barcode_multiarrays.is_dirty():
            self.write_mapping(group, 'barcode_multiarrays', data.barcode_multiarrays, overwrite = overwrite)
        if overwrite or data.feature_multiarrays.is_dirty():
            self.write_mapping(group, 'feature_multiarrays', data.feature_multiarrays, overwrite = overwrite)
        if overwrite or data.barcode_multigraphs.is_dirty():
            self.write_mapping(group, 'barcode_multigraphs', data.barcode_multigraphs, overwrite = overwrite)
        if overwrite or data.feature_multigraphs.is_dirty():
            self.write_mapping(group, 'feature_multigraphs', data.feature_multigraphs, overwrite = overwrite)


    def write_multimodal_data(self, data: MultimodalData, overwrite: bool = True) -> None:
        """ Write MultimodalData
        """
        if overwrite:
            for key in data.list_data():
                self.write_unimodal_data(self.root, key, data.get_data(key), overwrite = True)
        else:
            for key in data.data.deleted:
                del self.root[key]
            for key in data.data.accessed:
                self.write_unimodal_data(self.root, key, data.get_data(key), overwrite = key in data.data.modified)
        self.root.attrs['_selected'] = data._selected
