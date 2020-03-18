import gzip
import logging
import os.path
import re
from typing import List, Tuple, Dict

import anndata
import numpy as np
import pandas as pd
import tables
from scipy.io import mmread
from scipy.sparse import csr_matrix, issparse

from pegasusio import UnimodalData, MultimodalData
from .hdf5_utils import load_10x_h5_file, load_pegasus_h5_file

logger = logging.getLogger("pegasusio")
from pegasus.utils import decorators as pg_deco






def _load_csv_file_sparse(input_csv, genome, sep, dtype, ngene=None, chunk_size=1000):
    """
     Read a csv file in chunks
    """

    import scipy.sparse
    features = []
    dense_arrays = []
    sparse_arrays = []

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero")

    with (
        gzip.open(input_csv, mode="rt")
        if input_csv.endswith(".gz")
        else open(input_csv)
    ) as fin:
        barcodes = next(fin).strip().split(sep)[1:]
        for line in fin:
            fields = line.strip().split(sep)
            features.append(fields[0])
            dense_arrays.append(np.array(fields[1:], dtype=dtype))
            if len(dense_arrays) == chunk_size:
                sparse_arrays.append(
                    scipy.sparse.csr_matrix(np.stack(dense_arrays, axis=0))
                )
                dense_arrays = []

    if len(dense_arrays) > 0:
        sparse_arrays.append(scipy.sparse.csr_matrix(np.stack(dense_arrays, axis=0)))
        dense_arrays = None
    mat = scipy.sparse.vstack(sparse_arrays)
    barcode_metadata = {"barcodekey": barcodes}
    feature_metadata = {"featurekey": features, "featurename": features}
    data = MemData()
    array2d = Array2D(barcode_metadata, feature_metadata, mat.T)
    array2d.filter(ngene=ngene)
    data.addData(genome, array2d)
    return data


def load_csv_file(
    input_csv: str,
    genome: str,
    sep: str = ",",
    ngene: int = None,
    chunk_size: int = None,
) -> "MemData":
    """Load count matrix from a CSV-style file, such as CSV file or DGE style tsv file.

     Parameters
     ----------

     input_csv : `str`
         The CSV file, gzipped or not, containing the count matrix.
     genome : `str`
         The genome reference.
     sep: `str`, optional (default: ',')
         Separator between fields, either ',' or '\t'.
     ngene : `int`, optional (default: None)
         Minimum number of genes to keep a barcode. Default is to keep all barcodes.
     chunk_size: `int`, optional (default: None)
        Chunk size for reading dense matrices as sparse
     Returns
     -------

     An MemData object containing a genome-Array2D pair.

     Examples
     --------
     >>> io.load_csv_file('example_ADT.csv', genome = 'GRCh38')
     >>> io.load_csv_file('example.umi.dge.txt.gz', genome = 'GRCh38', sep = '\t')
     """

    path = os.path.dirname(input_csv)
    base = os.path.basename(input_csv)
    is_hca_csv = base == "expression.csv"

    if sep == "\t":
        # DGE, columns are cells, which is around thousands and we can use pandas.read_csv
        if chunk_size is not None:
            return _load_csv_file_sparse(
                input_csv,
                genome,
                sep,
                "float32" if base.startswith("expression") else "int",
                ngene=ngene,
                chunk_size=chunk_size)
        df = pd.read_csv(input_csv, header=0, index_col=0, sep=sep)
        mat = csr_matrix(df.values.T)
        barcode_metadata = {"barcodekey": df.columns.values}
        feature_metadata = {
            "featurekey": df.index.values,
            "featurename": df.index.values,
        }
    else:
        if chunk_size is not None and not is_hca_csv:
            return _load_csv_file_sparse(
                input_csv,
                genome,
                sep,
                "float32" if base.startswith("expression") else "int",
                ngene=ngene,
                chunk_size=chunk_size,
            )
        # For CSV files, wide columns prevent fast pd.read_csv loading
        converter = (
            float if base.startswith("expression") else int
        )  # If expression -> float otherwise int

        barcodes = []
        names = []
        stacks = []
        with (
            gzip.open(input_csv, mode="rt")
            if input_csv.endswith(".gz")
            else open(input_csv)
        ) as fin:
            barcodes = next(fin).strip().split(sep)[1:]
            for line in fin:
                fields = line.strip().split(sep)
                names.append(fields[0])
                stacks.append([converter(x) for x in fields[1:]])

        mat = csr_matrix(np.stack(stacks, axis=1 if not is_hca_csv else 0))
        barcode_metadata = {"barcodekey": barcodes}
        feature_metadata = {"featurekey": names, "featurename": names}

        if is_hca_csv:
            barcode_file = os.path.join(path, "cells.csv")
            if os.path.exists(barcode_file):
                barcode_metadata = pd.read_csv(barcode_file, sep=",", header=0)
                assert "cellkey" in barcode_metadata
                barcode_metadata.rename(columns={"cellkey": "barcodekey"}, inplace=True)

            feature_file = os.path.join(path, "genes.csv")
            if os.path.exists(feature_file):
                feature_metadata = pd.read_csv(feature_file, sep=",", header=0)

    data = MemData()
    array2d = Array2D(barcode_metadata, feature_metadata, mat)
    array2d.filter(ngene=ngene)
    data.addData(genome, array2d)

    return data




def infer_file_format(input_file: str) -> Tuple[str, str, str]:
    """ Infer file format from input_file name

    This function infer file format by inspecting the file name.

    Parameters
    ----------

    input_file : `str`
        Input file name.

    Returns
    -------
    `str`
        File format, choosing from '10x', 'pegasus', 'h5ad', 'loom', 'mtx', 'dge', 'csv' and 'tsv'.
    `str`
        The path covering all input files. Most time this is the same as input_file. But for HCA mtx and csv, this should be parent directory.
    `str`
        Type of the path, either 'file' or 'directory'.
    """

    file_format = None
    copy_path = input_file
    copy_type = "file"

    if input_file.endswith(".h5"):
        file_format = "10x"
    elif input_file.endswith(".h5sc"):
        file_format = "pegasus"
    elif input_file.endswith(".h5ad"):
        file_format = "h5ad"
    elif input_file.endswith(".loom"):
        file_format = "loom"
    elif (
        input_file.endswith(".mtx")
        or input_file.endswith(".mtx.gz")
        or os.path.splitext(input_file)[1] == ""
    ):
        file_format = "mtx"
        if os.path.splitext(input_file)[1] != "":
            copy_path = os.path.dirname(input_file)
        copy_type = "directory"
    elif input_file.endswith("dge.txt.gz"):
        file_format = "dge"
    elif input_file.endswith(".csv") or input_file.endswith(".csv.gz"):
        file_format = "csv"
        if os.path.basename(input_file) == "expression.csv":
            copy_path = os.path.dirname(input_file)
            copy_type = "directory"
    elif input_file.endswith(".txt") or input_file.endswith(".tsv") or input_file.endswith(
        ".txt.gz") or input_file.endswith(".tsv.gz"):
        file_format = "tsv"
    else:
        raise ValueError("Unrecognized file type for file {}!".format(input_file))

    return file_format, copy_path, copy_type


@pg_deco.TimeLogger()
def read_input(
    input_file: str,
    genome: str = None,
    return_type: str = "AnnData",
    concat_matrices: bool = False,
    h5ad_mode: str = "a",
    ngene: int = None,
    select_singlets: bool = False,
    channel_attr: str = None,
    chunk_size: int = None,
    black_list: List[str] = [],
) -> "MemData or AnnData or List[AnnData]":
    """Load data into memory.

    This function is used to load input data into memory. Inputs can be in 10x genomics v2 & v3 formats (hdf5 or mtx), HCA DCP mtx and csv formats, Drop-seq dge format, and CSV format.

    Parameters
    ----------

    input_file : `str`
        Input file name.
    genome : `str`, optional (default: None)
        A string contains comma-separated genome names. pegasus will read all matrices matching the genome names. If genome is None, all matrices will be considered. For formats like loom, mtx, dge, csv and tsv, genome is used to provide genome name. In this case if genome is None, except mtx format, '' is used as the genome name instead.
    return_type : `str`
        Return object type, can be either 'MemData' or 'AnnData'.
    concat_matrices : `boolean`, optional (default: False)
        If input file contains multiple matrices, turning this option on will concatenate them into one AnnData object. Otherwise return a list of AnnData objects.
    h5ad_mode : `str`, optional (default: 'a')
        If input is in h5ad format, the backed mode for loading the data. Mode could be 'a', 'r', 'r+', where 'a' refers to load the whole matrix into memory.
    ngene : `int`, optional (default: None)
        Minimum number of genes to keep a barcode. Default is to keep all barcodes.
    select_singlets : `bool`, optional (default: False)
        If this option is on, only keep DemuxEM-predicted singlets when loading data.
    channel_attr : `str`, optional (default: None)
        Use channel_attr to represent different samples. This will set a 'Channel' column field with channel_attr.
    chunk_size: `int`, optional (default: None)
        Chunk size for reading dense matrices as sparse
    black_list : `List[str]`, optional (default: [])
        Attributes in black list will be poped out.

    Returns
    -------
    `MemData` object or `anndata` object or a list of `anndata` objects
        An `MemData` object or `anndata` object or a list of `anndata` objects containing the count matrices.

    Examples
    --------
    >>> adata = pg.read_input('example_10x.h5', genome = 'mm10')
    >>> adata = pg.read_input('example.h5ad', h5ad_mode = 'r+')
    >>> adata = pg.read_input('example_ADT.csv')
    """

    input_file = os.path.expanduser(os.path.expandvars(input_file))
    file_format, _, _ = infer_file_format(input_file)

    if file_format == "pegasus":
        data = load_pegasus_h5_file(
            input_file, ngene=ngene, select_singlets=select_singlets
        )
    elif file_format == "10x":
        data = load_10x_h5_file(input_file, ngene=ngene)
    elif file_format == "h5ad":
        data = anndata.read_h5ad(
            input_file,
            chunk_size=chunk_size,
            backed=(None if h5ad_mode == "a" else h5ad_mode),
        )
        if channel_attr is not None:
            data.obs["Channel"] = data.obs[channel_attr]
        for attr in black_list:
            if attr in data.obs:
                data.obs.drop(columns = attr, inplace = True)
    elif file_format == "mtx":
        data = load_mtx_file(input_file, genome, ngene=ngene)
    elif file_format == "loom":
        if genome is None:
            genome = ''
        data = load_loom_file(input_file, genome, ngene=ngene)
        if isinstance(data, anndata.AnnData):
            file_format = "h5ad"
            if channel_attr is not None:
                data.obs["Channel"] = data.obs[channel_attr]
            for attr in black_list:
                if attr in data.obs:
                    data.obs.drop(columns = attr, inplace = True)
    else:
        assert file_format == "dge" or file_format == "csv" or file_format == "tsv"
        if genome is None:
            genome = ''
        data = load_csv_file(
            input_file,
            genome,
            sep=("\t" if file_format == "dge" or file_format == "tsv" else ","),
            ngene=ngene,
            chunk_size=chunk_size,
        )

    if file_format != "h5ad":
        data.restrain_keywords(genome)
        if return_type == "AnnData":
            data = data.convert_to_anndata(
                concat_matrices=concat_matrices,
                channel_attr=channel_attr,
                black_list=black_list,
            )
    else:
        assert return_type == "AnnData"
        if select_singlets:
            assert 'demux_type' in data.obs
            data._inplace_subset_obs((data.obs['demux_type'] == 'singlet').values)

    return data






@pg_deco.TimeLogger()
def write_output(
    data: "MemData or AnnData",
    output_file: str,
    whitelist: List = ["obs", "obsm", "uns", "var", "varm"],
) -> None:
    """ Write data back to disk.

    This function is used to write data back to disk.

    Parameters
    ----------

    data : `MemData` or `AnnData`
        data to write back, can be either an MemData or AnnData object.
    output_file : `str`
        output file name. If data is MemData, output_file should ends with suffix '.h5sc'. Otherwise, output_file can end with either '.h5ad', '.loom', or '.mtx.gz'. If output_file ends with '.loom', a LOOM file will be generated. If no suffix is detected, an appropriate one will be appended.
    whitelist : `list`, optional, default = ["obs", "obsm", "uns", "var", "varm"]
        List that indicates changed fields when writing h5ad file in backed mode. For example, ['uns/Groups', 'obsm/PCA'] will only write Groups in uns, and PCA in obsm; the rest of the fields will be unchanged.

    Returns
    -------
    `None`

    Examples
    --------
    >>> pg.write_output(adata, 'test.h5ad')
    """

    if (not isinstance(data, MemData)) and (not isinstance(data, anndata.AnnData)):
        raise ValueError("data is neither an MemData nor AnnData object!")

    # Identify and correct file suffix
    file_name, _, suffix = output_file.rpartition(".")
    if suffix == 'gz' and file_name.endswith('.mtx'):
        suffix = 'mtx.gz'
    if file_name == "":
        file_name = output_file
        suffix = "h5sc" if isinstance(data, MemData) else "h5ad"
    if isinstance(data, MemData) and suffix != "h5sc" and suffix != "h5":
        logging.warning(
            "Detected file suffix for this MemData object is neither .h5sc nor .h5. We will assume output_file is a file name and append .h5sc suffix."
        )
        file_name = output_file
        suffix = "h5sc"
    if isinstance(data, anndata.AnnData) and (suffix not in ["h5ad", "loom", "mtx.gz"]):
        logging.warning(
            "Detected file suffix for this AnnData object is neither .h5ad or .loom. We will assume output_file is a file name and append .h5ad suffix."
        )
        file_name = output_file
        suffix = "h5ad"
    output_file = file_name + "." + suffix

    # Eliminate objects starting with fmat_ from uns
    if isinstance(data, anndata.AnnData):
        keys = list(data.uns)
        for keyword in keys:
            if keyword.startswith("fmat_"):
                data.uns.pop(keyword)

    # Write outputs
    if suffix == "mtx.gz":
        _write_mtx(data, output_file)
    elif suffix == "h5sc" or suffix == "h5":
        data.write_h5_file(output_file)
    elif suffix == "loom":
        _write_loom(data, output_file)
    elif (
        not data.isbacked
        or (data.isbacked and data.file._file.mode != "r+")
    ):
        data.write(output_file, compression="gzip")
    else:
        assert data.file._file.mode == "r+"
        import h5py

        h5_file = data.file._file
        # Fix old h5ad files in which obsm/varm were stored as compound datasets
        for key in ["obsm", "varm"]:
            if key in h5_file.keys() and isinstance(h5_file[key], h5py.Dataset):
                del h5_file[key]
                whitelist.append(key)

        _update_backed_h5ad(
            h5_file, _to_dict_fixed_width_arrays(data), _parse_whitelist(whitelist)
        )
        h5_file.close()
