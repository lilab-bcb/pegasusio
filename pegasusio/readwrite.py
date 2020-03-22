import os
import time
import anndata
from typing import Tuple, Set
import logging
logger = logging.getLogger("pegasusio")

from pegasusio import UnimodalData, MultimodalData

from .hdf5_utils import load_10x_h5_file, load_pegasus_h5_file, load_loom_file, write_loom_file
from .text_utils import load_mtx_file, write_mtx_file, load_csv_file, write_scp_file
from .zarr_utils import ZarrFile



def infer_file_type(input_file: str) -> Tuple[str, str, str]:
    """ Infer file format from input_file name

    This function infer file type by inspecting the file name.

    Parameters
    ----------

    input_file : `str`
        Input file name.

    Returns
    -------
    `str`
        File type, choosing from 'zarr', 'h5sc'(obsoleted), 'h5ad', 'loom', '10x', 'mtx', 'csv' and 'tsv'.
    `str`
        The path covering all input files. Most time this is the same as input_file. But for HCA mtx and csv, this should be parent directory.
    `str`
        Type of the path, either 'file' or 'directory'.

    Note: The last two `str`s are mainly used for transfer to cloud
    """
    file_type = None
    copy_path = input_file
    copy_type = "file"

    if input_file.endswith(".zarr"):
        file_type = "zarr"
    elif input_file.endswith(".h5sc"):
        file_type = "h5sc"
    elif input_file.endswith(".h5ad"):
        file_type = "h5ad"
    elif input_file.endswith(".loom"):
        file_type = "loom"
    elif input_file.endswith(".h5"):
        file_type = "10x"
    elif (
        input_file.endswith(".mtx")
        or input_file.endswith(".mtx.gz")
        or os.path.splitext(input_file)[1] == ""
    ):
        file_type = "mtx"
        if os.path.splitext(input_file)[1] != "":
            copy_path = os.path.dirname(input_file)
        copy_type = "directory"
    elif input_file.endswith(".csv") or input_file.endswith(".csv.gz"):
        file_type = "csv"
        if os.path.basename(input_file) == "expression.csv":
            copy_path = os.path.dirname(input_file)
            copy_type = "directory"
    elif input_file.endswith(".txt") or input_file.endswith(".tsv") or input_file.endswith(
        ".txt.gz") or input_file.endswith(".tsv.gz"):
        file_type = "tsv"
    else:
        raise ValueError("Unrecognized file type for file {}!".format(input_file))

    return file_type, copy_path, copy_type


def read_input(
    input_file: str,
    file_type: str = None,
    genome: str = None,
    exptype: str = None,
    ngene: int = None,
    select_singlets: bool = False,
    black_list: Set[str] = None,
    select_data: Set[str] = None,
    select_exptype: Set[str] = None 
) -> MultimodalData:
    """Load data into memory.

    This function is used to load input data into memory. Inputs can be in 10x genomics v2 & v3 formats (hdf5 or mtx), HCA DCP mtx and csv formats, Drop-seq dge format, and CSV format.

    Parameters
    ----------

    input_file : `str`
        Input file name.
    file_type : `str`, optional (default: None)
        File type, choosing from 'zarr', 'h5sc'(obsoleted), 'h5ad', 'loom', '10x', 'mtx', 'csv', or 'tsv'. If None, inferred from input_file
    genome : `str`, optional (default: None)
        For formats like loom, mtx, dge, csv and tsv, genome is used to provide genome name. In this case if genome is None, except mtx format, "unknown" is used as the genome name instead.
    exptype : `str`, optional (default: None)
        Default experiment type, choosing from 'rna', 'citeseq', 'hashing', 'tcr', 'bcr', 'crispr' or 'atac'. If None, use 'rna' as default.
    ngene : `int`, optional (default: None)
        Minimum number of genes to keep a barcode. Default is to keep all barcodes.
    select_singlets : `bool`, optional (default: False)
        If this option is on, only keep DemuxEM-predicted singlets when loading data.
    black_list : `Set[str]`, optional (default: None)
        Attributes in black list will be poped out.
    select_data: `Set[str]`, optional (default: None)
        Only select data with keys in select_data.
    select_exptype: `Set[str]`, optional (default: None)
        Only select data with experiment type in select_exptype.

    Returns
    -------

    A MultimodalData object.

    Examples
    --------
    >>> data = io.read_input('example_10x.h5')
    >>> data = io.read_input('example.h5ad')
    >>> data = io.read_input('example_ADT.csv', genome = 'hashing_HTO', exptype = 'hashing')
    """
    start = time.perf_counter()

    input_file = os.path.expanduser(os.path.expandvars(input_file))
    if file_type is None:
        file_type, _, _ = infer_file_type(input_file)

    if file_type == "zarr":
        zf = ZarrFile(input_file)
        data = zf.read_multimodal_data(ngene = ngene, select_singlets = select_singlets)
        del zf
    elif file_type == "h5sc":
        data = load_pegasus_h5_file(input_file, ngene=ngene, select_singlets=select_singlets)
    elif file_type == "h5ad":
        data = MultimodalData(anndata.read_h5ad(input_file), genome = genome, exptype = exptype)
    elif file_type == "loom":
        data = load_loom_file(input_file, genome = genome, exptype = exptype, ngene = ngene)
    elif file_type == "10x":
        data = load_10x_h5_file(input_file, ngene=ngene)
    elif file_type == "mtx":
        data = load_mtx_file(input_file, genome = genome, exptype = exptype, ngene = ngene)
    else:
        assert file_type == "csv" or file_type == "tsv"
        data = load_csv_file(input_file, sep = "," if file_type == "csv" else "\t", genome = genome, exptype = exptype, ngene = ngene)

    data.subset_data(select_data, select_exptype)
    data.scan_black_list(black_list)    

    end = time.perf_counter()
    logger.info("{} file '{}' is loaded, time spent = {:.2f}s.".format(file_type, input_file, end - start))

    return data



def write_output(
    data: MultimodalData,
    output_file: str,
    file_type: str = None,
    zarr_zipstore: bool = False,
    is_sparse: bool = True,
    precision: int = 2
) -> None:
    """ Write data back to disk.

    This function is used to write data back to disk.

    Parameters
    ----------

    data : MutimodalData
        data to write back.
    output_file : `str`
        output file name. Note that for mtx files, output_file specifies a directory. For scp format, file_type must be specified. 
    file_type : `str`, optional (default: None)
        File type can be 'zarr', 'h5ad', 'loom', 'mtx' or 'scp'. If file_type is None, it will be inferred based on output_file.
    zarr_zipstore: `bool`, optional (default: False)
        Only apply to zarr output, using ZipStore to have one file instead of a folder.
    is_sparse : `bool`, optional (default: True)
        Only used for writing out SCP-compatible files, if write expression as a sparse matrix.
    precision : `int`, optional (default: 2)
        Precision after decimal point for values in mtx and scp expression matrix.

    Returns
    -------
    `None`

    Examples
    --------
    >>> io.write_output(data, 'test.zarr')
    """
    start = time.perf_counter()

    output_file = os.path.expanduser(os.path.expandvars(output_file))

    def _infer_output_file_type(output_File: str) -> str:
        if output_file.endswith(".zarr"):
            return "zarr"
        elif output_file.endswith(".h5ad"):
            return "h5ad"
        elif output_file.endswith(".loom"):
            return "loom"
        else:
            return "mtx"

    file_type = _infer_output_file_type(output_file) if file_type is None else file_type

    if file_type == "zarr":
        zf = ZarrFile(output_file, mode = "w", storage_type = "ZipStore" if zarr_zipstore else None)
        zf.write_multimodal_data(data)
        del zf
    elif file_type == "h5ad":
        data.to_anndata().write(output_file, compression="gzip")
    elif file_type == "loom":
        write_loom_file(data, output_file)
    elif file_type == "mtx":
        write_mtx_file(data, output_file, precision = precision)
    elif file_type == "scp":
        write_scp_file(data, output_file, is_sparse = is_sparse, precision = precision)
    else:
        raise ValueError("Unknown file type {}!".format(file_type))

    end = time.perf_counter()
    logger.info("{} file '{}' is written. Time spent = {:.2f}s.".format(file_type, output_file, end - start))
