import os
import numpy as np
import pandas as pd

from pegasusio import NanostringData, MultimodalData

import logging
logger = logging.getLogger(__name__)



def load_nanostring_files(input_matrix: str, segment_file: str, annotation_file: str = None, genome: str = None) -> MultimodalData:
    """Load Cyto data from a FCS file, support v2.0, v3.0 and v3.1.

    Parameters
    ----------

    input_matrix : `str`
        Input Q3 normalized data matrix.
    segment_file: `str`
        Segment file containing segmentation information for each ROI.
    annotation_file: `str`, optional (default None)
        An optional annotation file providing tissue type information etc.
    genome : `str`, optional (default None)
        The genome reference. If None, use "unknown" instead.

    Returns
    -------

    A MultimodalData object containing a (genome, CytoData) pair.

    Examples
    --------
    >>> io.load_fcs_file('example.fcs', genome = 'GRCh38')
    """
    if not os.path.isfile(input_matrix):
        raise FileNotFoundError(f"File {input_matrix} does not exist!")
    df = pd.read_csv(input_matrix, sep = '\t', header = 0, index_col = 0)

    barcodekey = pd.Index([x.replace('.', '-') for x in df.columns.values])
    barcode_metadata = {"barcodekey": barcodekey.values} # I guess the original matrix is processed in R because '-' -> '.'.
    feature_metadata = {"featurekey": df.index.values}
    matrix = np.transpose(df.values) # float64, do we need to convert it to float32?

    if not os.path.isfile(segment_file):
        raise FileNotFoundError(f"File {segment_file} does not exist!")
    df = pd.read_csv(segment_file, sep = '\t', header = 0, index_col = 0)

    idx = barcodekey.isin(df.index)
    if idx.sum() < barcodekey.size:
        logger.warning(f"Cannot find {barcodekey[~idx]} from the segment property file! Number of AOIs reduces to {idx.sum()}.")
        barcodekey = barcodekey[idx]
        barcode_metadata["barcodekey"] = barcodekey.values
        matrix = matrix[idx]
    if idx.sum() < df.shape[0]:
        logger.warning(f"Sample IDs {','.join(x for x in df.index[~df.index.isin(barcodekey)])} from the segment property file are not located in the matrix file!")
    df = df.reindex(barcodekey)

    for key in ["primer plate well", "slide name", "scan name", "panel", "segment", "aoi"]:
        if key in df.columns:
            df.loc[df[key].isna(), key] = "None"
            barcode_metadata[key] = df[key].values
    if "roi" in df.columns:
        rois = df["roi"].copy()
        rois[rois.isna()] = -1.0
        rois = rois.astype(int).astype(str)
        rois[rois == "-1"] = "None"
        barcode_metadata["roi"] = rois.values
    for key in ["area", "SequencingSaturation"]:
        if key in df.columns:
            df.loc[df[key].isna(), key] = 0.0
            barcode_metadata[key] = df[key].values.astype(np.float32)
    for key in ["RawReads", "TrimmedReads", "StitchedReads", "AlignedReads", "DeduplicatedReads"]:
        if key in df.columns:
            df.loc[df[key].isna(), key] = 0
            barcode_metadata[key] = df[key].values.astype(np.int32)

    if annotation_file is not None:
        if not os.path.isfile(annotation_file):
            raise FileNotFoundError(f"File {annotation_file} does not exist!")
        df = pd.read_csv(annotation_file, sep = '\t', header = 0, index_col = 0)
        assert barcodekey.isin(df.index).sum() == barcodekey.size
        df = df.reindex(barcodekey)
        for key in df.columns:
            barcode_metadata[key] = df[key].values

    genome = "unknown" if genome is None else genome
    metadata = {"genome": genome, "modality": "nanostring"}

    nanodata = NanostringData(barcode_metadata, feature_metadata, {"Q3Norm": matrix}, metadata)
    data = MultimodalData(nanodata)

    return data
