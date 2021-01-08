import os
import gzip
import anndata
from typing import Tuple, Set, Union, List
from pandas.api.types import is_list_like

import logging
logger = logging.getLogger(__name__)

from pegasusio import timer
from pegasusio import UnimodalData, MultimodalData


from .hdf5_utils import load_10x_h5_file, load_loom_file, write_loom_file
from .text_utils import load_mtx_file, write_mtx_file, load_csv_file, write_scp_file
from .zarr_utils import ZarrFile
from .vdj_utils import load_10x_vdj_file
from .cyto_utils import load_fcs_file
from .nanostring_utils import load_nanostring_files


def infer_file_type(input_file: Union[str, List[str]]) -> Tuple[str, str, str]:
    """ Infer file format from input_file name

    This function infer file type by inspecting the file name.

    Parameters
    ----------

    input_file : `str` or `List[str]`
        Input file name.

    Returns
    -------
    `str`
        File type, choosing from 'zarr', 'h5ad', 'loom', '10x', 'mtx', 'csv', 'tsv', 'fcs' or 'nanostring'.
    `str`
        The path covering all input files. Most time this is the same as input_file. But for HCA mtx and csv, this should be parent directory.
    `str`
        Type of the path, either 'file' or 'directory'.

    Note: The last two `str`s are mainly used for transfer to cloud
    """
    file_type = None
    copy_path = input_file
    copy_type = "file"

    if is_list_like(input_file):
        # now it can only be nanostring
        file_type = "nanostring"
        return file_type, copy_path, copy_type

    if input_file.endswith(".zarr") or input_file.endswith(".zarr.zip"):
        file_type = "zarr"
    elif input_file.endswith(".h5ad"):
        file_type = "h5ad"
    elif input_file.endswith(".loom"):
        file_type = "loom"
    elif input_file.endswith(".h5"):
        file_type = "10x"
    elif input_file.endswith(".fcs"):
        file_type = "fcs"
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
        raise ValueError(f"Unrecognized file type for file {input_file}!")

    return file_type, copy_path, copy_type


def is_vdj_file(input_csv: str, file_type: str) -> bool:
    if file_type != "csv":
        return False
    with (gzip.open(input_csv, "rt") if input_csv.endswith(".gz") else open(input_csv)) as fin:
        line = next(fin)
        return line.find(",chain,v_gene,d_gene,j_gene,c_gene,") >= 0


@timer(logger=logger)
def read_input(
    input_file: str,
    file_type: str = None,
    mode: str = "r",
    genome: str = None,
    modality: str = None,
    black_list: Set[str] = None,
    select_data: Set[str] = None,
    select_genome: Set[str] = None,
    select_modality: Set[str] = None,
) -> MultimodalData:
    """Load data into memory.

    This function is used to load input data into memory. Inputs can be in 'zarr', 'h5ad', 'loom', '10x', 'mtx', 'csv', 'tsv', 'fcs' (for flow/mass cytometry data) or 'nanostring' (Nanostring GeoMx spatial data) formats.

    Parameters
    ----------

    input_file : `str`
        Input file name.
    file_type : `str`, optional (default: None)
        File type, choosing from 'zarr', 'h5ad', 'loom', '10x', 'mtx', 'csv', 'tsv', 'fcs' (for flow/mass cytometry data) or 'nanostring'. If None, inferred from input_file.
    mode: `str`, optional (default: 'r')
        File open mode, options are 'r' or 'a'. If mode == 'a', file_type must be zarr and ngene/select_singlets cannot be set.
    genome : `str`, optional (default: None)
        For formats like loom, mtx, dge, csv and tsv, genome is used to provide genome name. In this case if genome is None, except mtx format, "unknown" is used as the genome name instead.
    modality : `str`, optional (default: None)
        Default modality, choosing from 'rna', 'atac', 'tcr', 'bcr', 'crispr', 'hashing', 'citeseq', 'cyto' (flow cytometry / mass cytometry) or 'nanostring'. If None, use 'rna' as default.
    black_list : `Set[str]`, optional (default: None)
        Attributes in black list will be poped out.
    select_data: `Set[str]`, optional (default: None)
        Only select data with keys in select_data. Select_data, select_genome and select_modality are mutually exclusive.
    select_genome: `Set[str]`, optional (default: None)
        Only select data with genomes in select_genome. Select_data, select_genome and select_modality are mutually exclusive.
    select_modality: `Set[str]`, optional (default: None)
        Only select data with modalities in select_modality. Select_data, select_genome and select_modality are mutually exclusive.

    Returns
    -------

    A MultimodalData object.

    Examples
    --------
    >>> data = io.read_input('example_10x.h5')
    >>> data = io.read_input('example.h5ad')
    >>> data = io.read_input('example_ADT.csv', genome = 'hashing_HTO', modality = 'hashing')
    """
    
    if is_list_like(input_file):
        input_file = [os.path.expanduser(os.path.expandvars(x)) for x in input_file]
    else:
        input_file = os.path.expanduser(os.path.expandvars(input_file))

    if file_type is None:
        file_type, _, _ = infer_file_type(input_file)

    if mode == "a":
        if file_type != "zarr":
            raise ValueError("Only Zarr file can have mode 'a'!")
        zf = ZarrFile(input_file, mode = mode)
        data = zf.read_multimodal_data(attach_zarrobj = True)
    else:
        if file_type == "zarr":
            zf = ZarrFile(input_file)
            data = zf.read_multimodal_data()
        elif file_type == "h5ad":
            data = MultimodalData(anndata.read_h5ad(input_file), genome = genome, modality = modality)
        elif file_type == "loom":
            data = load_loom_file(input_file, genome = genome, modality = modality)
        elif file_type == "10x":
            data = load_10x_h5_file(input_file)
        elif file_type == "fcs":
            data = load_fcs_file(input_file, genome = genome)
        elif file_type == "nanostring":
            input_matrix = input_file[0]
            segment_file = input_file[1]
            annotation_file = input_file[2] if len(input_file) > 2 else None
            data = load_nanostring_files(input_matrix, segment_file, annotation_file = annotation_file, genome = genome)
        elif file_type == "mtx":
            data = load_mtx_file(input_file, genome = genome, modality = modality)
        else:
            assert file_type == "csv" or file_type == "tsv"
            if is_vdj_file(input_file, file_type):
                data = load_10x_vdj_file(input_file, genome = genome, modality = modality)
            else:
                data = load_csv_file(input_file, sep = "," if file_type == "csv" else "\t", genome = genome, modality = modality)

    data.subset_data(select_data, select_genome, select_modality)
    data.kick_start()
    data.scan_black_list(black_list)

    logger.info(f"{file_type} file '{input_file}' is loaded.")

    return data


@timer(logger=logger)
def write_output(
    data: Union[MultimodalData, UnimodalData],
    output_file: str,
    file_type: str = None,
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
        File type can be 'zarr' (as folder), 'zarr.zip' (as a ZIP file), 'h5ad', 'loom', 'mtx' or 'scp'. If file_type is None, it will be inferred based on output_file.
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
    if isinstance(data, UnimodalData):
        data = MultimodalData(data)

    output_file = os.path.expanduser(os.path.expandvars(output_file))

    def _infer_output_file_type(output_File: str) -> str:
        if output_file.endswith(".zarr"):
            return "zarr"
        elif output_file.endswith(".zarr.zip"):
            return "zarr.zip"
        elif output_file.endswith(".h5ad"):
            return "h5ad"
        elif output_file.endswith(".loom"):
            return "loom"
        else:
            name, sep, suf = output_file.rpartition(".")
            return "mtx" if sep == "" else suf

    file_type = _infer_output_file_type(output_file) if file_type is None else file_type
    if file_type not in {"zarr", "zarr.zip", "h5ad", "loom", "mtx", "scp"}:
        raise ValueError(f"Unsupported output file type '{file_type}'!")

    _tmp_multi = data._clean_tmp() # for each unidata, remove uns keys starting with '_tmp' and store these values to _tmp_multi

    if file_type.startswith("zarr"):
        zf = ZarrFile(output_file, mode = "w", storage_type = "ZipStore" if file_type == "zarr.zip" else None)
        zf.write_multimodal_data(data)
        del zf
    elif file_type == "h5ad":
        data.to_anndata().write(output_file, compression="gzip")
    elif file_type == "loom":
        write_loom_file(data, output_file)
    elif file_type == "mtx":
        write_mtx_file(data, output_file, precision = precision)
    else:
        assert file_type == "scp"
        write_scp_file(data, output_file, is_sparse = is_sparse, precision = precision)

    data._addback_tmp(_tmp_multi)
    logger.info(f"{file_type} file '{output_file}' is written.")
