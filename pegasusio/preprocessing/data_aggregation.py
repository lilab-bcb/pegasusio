import os
import pandas as pd
from subprocess import check_call
from typing import List, Tuple, Dict

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


def _parse_genome_string(genome_str: str) -> Tuple[str, Dict[str, str]]
    genome = genome_dict = None
    if genome_str is not None:        
        if genome_str.find(":") < 0:
            genome = genome_str
        else:
            genome_dict = {}
            for item in genome_str.split(","):
                key, value = item.split(":")
                genome_dict[key] = value
    return genome, genome_dict


@timer(logger=logger)
def aggregate_matrices(
    csv_file: str,
    restrictions: List[str] = [],
    attributes: List[str] = [],
    default_ref: str = None,
    append_sample_name: bool = True,
    ngene: int = None,
    select_singlets: bool = False,
) -> MultimodalData:
    """Aggregate channel-specific count matrices into one big count matrix.

    This function takes as input a csv_file, which contains at least 2 columns — Sample, sample name; Location, file that contains the count matrices (e.g. filtered_gene_bc_matrices_h5.h5), and merges matrices from the same genome together. If multi-modality exists, a third Modality column is required. An aggregated Multimodal Data will be returned.

    In the Reference column, you can replace genome names by using "hg19:GRCh38,mm9:mm10", which will replace all hg19 to GRCh38 and all mm9 to mm10. However, in this case, the genome passed to read_input will be None.

    Parameters
    ----------

    csv_file : `str`
        The CSV file containing information about each channel.
    restrictions : `list[str]`, optional (default: [])
        A list of restrictions used to select channels, each restriction takes the format of name:value,…,value or name:~value,..,value, where ~ refers to not.
    attributes : `list[str]`, optional (default: [])
        A list of attributes need to be incorporated into the output count matrix.
    default_ref : `str`, optional (default: None)
        Default reference name to use. If there is no Reference column in the csv_file, a Reference column will be added with default_ref as its value.
    append_sample_name : `bool`, optional (default: True)
        By default, append sample_name to each channel. Turn this option off if each channel has distinct barcodes.
    select_singlets : `bool`, optional (default: False)
        If we have demultiplexed data, turning on this option will make pegasus only include barcodes that are predicted as singlets.
    ngene : `int`, optional (default: None)
        The minimum number of expressed genes to keep one barcode.

    Returns
    -------
    `MultimodalData` object.

    Examples
    --------
    >>> pio.aggregate_matrix('example.csv', 'example_10x.h5', ['Source:pbmc', 'Donor:1'], ['Source', 'Platform', 'Donor'])
    """

    df = pd.read_csv(csv_file, header=0, index_col=False) # load sample sheet

    # Remove duplicated items
    restrictions = set(restrictions)
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

    df = df.loc[idx]

    if df.shape[0] == 0:
        raise ValueError("No data pass the restrictions!")

    # If Reference does not exist and default_ref is not None, add one Reference column
    if ("Reference" not in df.columns) and (default_ref is not None):
        df["Reference"] = default_ref

    # Load data
    tot = 0
    dest_paths = [] # record localized file paths so that we can remove them later
    aggrData = AggrData()

    for idx_num, row in df.iterrows():
        input_file = os.path.expanduser(os.path.expandvars(row["Location"].rstrip(os.sep))) # extend all user variables
        file_type, copy_path, copy_type = infer_file_type(input_file) # infer file type

        if row["Location"].lower().startswith('gs://'): # if Google bucket
            base_name = os.path.basename(copy_path)
            dest_path = idx_num + "_tmp_" + base_name # id_num will make sure dest_path is unique in the sample sheet
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

        genome, genome_dict = _parse_genome_string(row.get("Reference", None))
        modality = row.get("Modality", None)


        data = read_input(input_file, file_type = file_type, genome = genome, modality = modality, ngene = ngene, select_singlets = select_singlets)
        data._update_barcode_metadata_info(row, attributes, append_sample_name)

        if (file_type in ["10x", "h5sc", "zarr"]) and (genome_dict is not None): # Process the case where some data are in hg19 and others in GRCh38 and people want to merge them. May need to remove in the future
            data._update_genome(genome_dict)

        aggrData.add_data(data)

        tot += 1
        logger.info(f"Loaded {input_file}.")

    # Merge data
    aggregated_data = aggrData.aggregate()
    logger.info(f"Aggregated {tot} files.")

    # Delete temporary file
    if len(dest_paths) > 0:
        for dest_path in dest_paths:
            check_call(["rm", "-rf", dest_path])
        logger.info("Temporary files are deleted.")

    return aggregated_data
