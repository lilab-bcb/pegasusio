import numpy as np
import pandas as pd
import itertools
from scipy.sparse import csr_matrix

from pegasusio import VDJData, MultimodalData


def load_10x_vdj_file(input_csv: str, genome: str = None, modality: str = None) -> MultimodalData:
    """Load VDJ data from a 10x CSV file

    Parameters
    ----------

    input_csv : `str`
        The CSV file, gzipped or not, containing the count matrix.
    genome : `str`, optional (default None)
        The genome reference. If None, use "unknown" instead.
    modality: `str`, optional (default None)
        Modality. It should be automatically detected from the CSV file. If not None and the detected modality is not the same as the one users' provide, report an error.

    Returns
    -------

    A MultimodalData object containing a (genome, VDJData) pair.

    Examples
    --------
    >>> io.load_csv_file('vdj_t_all_contig_annotations.csv', genome = 'GRCh38_tcr')
    """
    try:
        from pegasusio.cylib.funcs import convert_10x_vdj_to_vdjdata
    except ModuleNotFoundError:
        print("No module named 'pegasusio.cylib.funcs'")
        
    df = pd.read_csv(input_csv, na_filter = False) # Otherwise, '' will be converted to NaN
    idx = df["productive"] == (True if df["productive"].dtype.kind == "b" else "True")
    df = df[idx]
    df.sort_values(by = ["barcode", "umis"], ascending = [True, False], inplace = True, kind = "mergesort") # sort barcode and make sure it is stable

    feature_name = [x for x in df["chain"].value_counts().index if x != "Multi"][0]
    modal = None
    if feature_name in VDJData._features["tcr"]:
        modal = "tcr"
    elif feature_name in VDJData._features["bcr"]:
        modal = "bcr"
    else:
        raise ValueError(f"Unknown feature '{feature_name}' detected!")

    if (modality is not None) and (modality != modal):
        raise ValueError(f"Detected modality '{modal}' does not match user-provided modality '{modality}'!")
    modality = modal

    # Set up feature keys
    feature_metadata = {"featurekey": [x + (str(y + 1) if y > 0 else "") for x, y in itertools.product(VDJData._features[modality], range(VDJData._n_contigs))]}
    fid2pos = {}
    for i, value in enumerate(feature_metadata["featurekey"]):
        fid2pos[value] = i

    n_barcodes = df["barcode"].nunique()

    barcodes, is_cell, mats, strarrs = convert_10x_vdj_to_vdjdata(df["barcode"].values,
                                                                  df[VDJData._matrix_keywords[0:4] + ["is_cell"]].values.astype(np.int32),
                                                                  df[VDJData._matrix_keywords[4:] + ["chain"]].values,
                                                                  fid2pos, n_barcodes, VDJData._n_contigs)

    barcode_metadata = {"barcodekey": barcodes, "is_cell": is_cell}

    matrices = {}
    for i, keyword in enumerate(VDJData._matrix_keywords):
        mat = mats[i]
        if keyword == "high_confidence":
            mat = mat.astype(np.bool_)
        matrices[keyword] = csr_matrix(mat)

    genome = "unknown" if genome is None else genome
    metadata = {"genome": genome, "modality": modality}
    for i, keyword in enumerate(VDJData._uns_keywords):
        metadata[keyword] = strarrs[i]

    vdjdata = VDJData(barcode_metadata, feature_metadata, matrices, metadata)
    vdjdata.separate_channels()
    data = MultimodalData(vdjdata)

    return data
