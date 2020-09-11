import os
import numpy as np
import pandas as pd
from subprocess import check_call
from typing import List, Tuple, Dict, Set, Union, Optional

from pegasusio import timer
from pegasusio import MultimodalData, AggrData
from pegasusio import infer_file_type, read_input

import logging
logger = logging.getLogger(__name__)


def _find_digits(value: str) -> Tuple[str, int]:
    pos = len(value) - 1
    while pos >= 0 and value[pos].isdigit():
        pos -= 1
    pos += 1
    assert pos < len(value)
    return (value[:pos], int(value[pos:]))


def _parse_restriction_string(rstr: str) -> Tuple[str, bool, Set[str]]:
    pos = rstr.index(":")
    name = rstr[:pos]
    isin = True
    if rstr[pos + 1] == "~":
        isin = False
        pos += 1
    content = set()
    for item in rstr[pos + 1:].split(","):
        values = item.split("-")
        if len(values) == 1:
            content.add(values[0])
        else:
            prefix, fr = _find_digits(values[0])
            assert values[1].isdigit()
            to = int(values[1]) + 1
            for i in range(fr, to):
                content.add(prefix + str(i))
    return (name, isin, content)


def _parse_genome_string(genome_str: str) -> Tuple[str, Dict[str, str]]:
    genome = None
    genome_dict = {}

    if genome_str is not None:
        fields = genome_str.split(",")
        for field in fields:
            items = field.split(":")
            if len(items) == 1:
                genome = items[0]
            else:
                genome_dict[items[0]] = items[1]

    return genome, genome_dict


@timer(logger=logger)
def aggregate_matrices(
    csv_file: Union[str, Dict[str, np.ndarray], pd.DataFrame],
    restrictions: Optional[Union[List[str], str]] = [],
    attributes: Optional[Union[List[str], str]] = [],
    default_ref: Optional[str] = None,
    append_sample_name: Optional[bool] = True,
    select_singlets: Optional[bool] = False,
    remap_string: Optional[str] = None,
    subset_string: Optional[str] = None,
    min_genes: Optional[int] = None,
    max_genes: Optional[int] = None,
    min_umis: Optional[int] = None,
    max_umis: Optional[int] = None,
    mito_prefix: Optional[str] = None,
    percent_mito: Optional[float] = None,
) -> MultimodalData:
    """Aggregate channel-specific count matrices into one big count matrix.

    This function takes as input a csv_file, which contains at least 2 columns — Sample, sample name; Location, file that contains the count matrices (e.g. filtered_gene_bc_matrices_h5.h5), and merges matrices from the same genome together. If multi-modality exists, a third Modality column might be required. An aggregated Multimodal Data will be returned.

    Parameters
    ----------

    csv_file : `str`
        The CSV file containing information about each channel. Alternatively, a dictionary or pd.Dataframe can be passed.
    restrictions : `list[str]` or `str`, optional (default: [])
        A list of restrictions used to select channels, each restriction takes the format of name:value,…,value or name:~value,..,value, where ~ refers to not. If only one restriction is provided, it can be provided as a string instead of a list.
    attributes : `list[str]` or `str`, optional (default: [])
        A list of attributes need to be incorporated into the output count matrix. If only one attribute is provided, this attribute can be provided as a string instead of a list.
    default_ref : `str`, optional (default: None)
        Default reference name to use. If there is no Reference column in the csv_file, a Reference column will be added with default_ref as its value. This argument can also be used for replacing genome names. For example, if default_ref is 'hg19:GRCh38,GRCh38', we will change any genome with name 'hg19' to 'GRCh38' and if no genome is provided, 'GRCh38' is the default.
    append_sample_name : `bool`, optional (default: True)
        By default, append sample_name to each channel. Turn this option off if each channel has distinct barcodes.
    select_singlets : `bool`, optional (default: False)
        If we have demultiplexed data, turning on this option will make pegasus only include barcodes that are predicted as singlets.
    remap_string: ``str``, optional, default ``None``
        Remap singlet names using <remap_string>, where <remap_string> takes the format "new_name_i:old_name_1,old_name_2;new_name_ii:old_name_3;...". For example, if we hashed 5 libraries from 3 samples sample1_lib1, sample1_lib2, sample2_lib1, sample2_lib2 and sample3, we can remap them to 3 samples using this string: "sample1:sample1_lib1,sample1_lib2;sample2:sample2_lib1,sample2_lib2". In this way, the new singlet names will be in metadata field with key 'assignment', while the old names will be kept in metadata field with key 'assignment.orig'.
    subset_string: ``str``, optional, default ``None``
        If select singlets, only select singlets in the <subset_string>, which takes the format "name1,name2,...". Note that if --remap-singlets is specified, subsetting happens after remapping. For example, we can only select singlets from sampe 1 and 3 using "sample1,sample3".
    min_genes: ``int``, optional, default: None
       Only keep cells with at least ``min_genes`` genes.
    max_genes: ``int``, optional, default: None
       Only keep cells with less than ``max_genes`` genes.
    min_umis: ``int``, optional, default: None
       Only keep cells with at least ``min_umis`` UMIs.
    max_umis: ``int``, optional, default: None
       Only keep cells with less than ``max_umis`` UMIs.
    mito_prefix: ``str``, optional, default: None
       Prefix for mitochondrial genes.
    percent_mito: ``float``, optional, default: None
       Only keep cells with percent mitochondrial genes less than ``percent_mito`` % of total counts. Only when both mito_prefix and percent_mito set, the mitochondrial filter will be triggered.

    Returns
    -------
    `MultimodalData` object.
        The aggregated count matrix as an MultimodalData object.

    Examples
    --------
    >>> data = aggregate_matrix('example.csv', restrictions=['Source:pbmc', 'Donor:1'], attributes=['Source', 'Platform', 'Donor'])
    """
    if isinstance(csv_file, str):
        df = pd.read_csv(csv_file, header=0, index_col=False) # load sample sheet
    elif isinstance(csv_file, dict):
        df = pd.DataFrame(csv_file)
    else:
        df = csv_file

    # Remove duplicated items
    if isinstance(restrictions, str):
        restrictions = [restrictions]
    restrictions = set(restrictions)
    if isinstance(attributes, str):
        attributes = [attributes]
    attributes = set(attributes)

    # Select data
    rvec = [_parse_restriction_string(x) for x in restrictions]

    idx = pd.Series([True] * df.shape[0], index=df.index, name="selected")
    for name, isin, content in rvec:
        assert name in df.columns
        if isin:
            idx = idx & df[name].isin(content)
        else:
            idx = idx & (~(df[name].isin(content)))

    if idx.sum() == 0:
        raise ValueError("No data pass the restrictions!")

    df = df.loc[idx].sort_values(by = "Sample") # sort by sample_name

    # parse default_ref
    def_genome, genome_dict = _parse_genome_string(default_ref)

    # Load data
    tot = 0
    dest_paths = [] # record localized file paths so that we can remove them later
    curr_sample = ""
    curr_row = curr_data = None
    aggrData = AggrData()

    for idx_num, row in df.iterrows():
        input_file = os.path.expanduser(os.path.expandvars(row["Location"].rstrip(os.sep))) # extend all user variables
        file_type, copy_path, copy_type = infer_file_type(input_file) # infer file type

        if row["Location"].lower().startswith('gs://'): # if Google bucket
            base_name = os.path.basename(copy_path)
            dest_path = f"{idx_num}_tmp_{base_name}" # id_num will make sure dest_path is unique in the sample sheet
            if not os.path.exists(dest_path):  # if dest_path exists, we may try to localize it once and may have the file cached
                if copy_type == "directory":
                    check_call(["mkdir", "-p", dest_path])
                    call_args = ["gsutil", "-m", "rsync", "-r", copy_path, dest_path]
                else:
                    call_args = ["gsutil", "-m", "cp", copy_path, dest_path]
                check_call(call_args)
            dest_paths.append(dest_path)

            if input_file == copy_path:
                input_file = dest_path
            else:
                input_file = os.path.join(dest_path, os.path.basename(input_file))

        genome = row.get("Reference", None)
        if (genome is not None) and (not isinstance(genome, str)): # to avoid NaN
            genome = None
        if genome is None:
            genome = def_genome
        modality = row.get("Modality", None)
        data = read_input(input_file, file_type = file_type, genome = genome, modality = modality)
        if len(genome_dict) > 0:
            data._update_genome(genome_dict)

        if row["Sample"] != curr_sample:
            if curr_data is not None:
                curr_data._propogate_genome()
                curr_data.filter_data(select_singlets = select_singlets, remap_string = remap_string, subset_string = subset_string, min_genes = min_genes, max_genes = max_genes, min_umis = min_umis, max_umis = max_umis, mito_prefix = mito_prefix, percent_mito = percent_mito)
                curr_data._update_barcode_metadata_info(curr_row, attributes, append_sample_name)
                aggrData.add_data(curr_data)
            curr_data = data
            curr_row = row
            curr_sample = row["Sample"]
        else:
            curr_data.update(data)

        tot += 1

    if curr_data is not None:
        curr_data._propogate_genome()
        curr_data.filter_data(select_singlets = select_singlets, remap_string = remap_string, subset_string = subset_string, min_genes = min_genes, max_genes = max_genes, min_umis = min_umis, max_umis = max_umis, mito_prefix = mito_prefix, percent_mito = percent_mito)
        curr_data._update_barcode_metadata_info(curr_row, attributes, append_sample_name)
        aggrData.add_data(curr_data)

    # Merge data
    aggregated_data = aggrData.aggregate()
    attributes.add("Channel")
    aggregated_data._convert_attributes_to_categorical(attributes)
    logger.info(f"Aggregated {tot} files.")

    # Delete temporary file
    if len(dest_paths) > 0:
        for dest_path in dest_paths:
            check_call(["rm", "-rf", dest_path])
        logger.info("Temporary files are deleted.")

    return aggregated_data
