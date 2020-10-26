import os
import numpy as np
import pandas as pd

from pegasusio import NanostringData, MultimodalData

import logging
logger = logging.getLogger(__name__)



def load_nanostring_files(input_matrix: str, segment_file: str, annotation_file: str = None, genome: str = None) -> MultimodalData:
    """Load Nanostring GeoMx input files.

    Parameters
    ----------

    input_matrix : `str`
        Input Q3 normalized data matrix.
    segment_file: `str`
        Segment file containing segmentation information for each ROI. If segment_file == 'protein', load GeoMx protein results from nCounter. 
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
    is_protein = segment_file == "protein"

    if not os.path.isfile(input_matrix):
        raise FileNotFoundError(f"File {input_matrix} does not exist!")

    genome = "unknown" if genome is None else genome
    metadata = {"genome": genome, "modality": "nanostring"}
    barcode_multiarrays = None

    if not is_protein:
        df = pd.read_csv(input_matrix, sep = '\t', header = 0, index_col = 0)

        barcodekey = pd.Index([x.replace('.', '-') for x in df.columns.values])
        barcode_metadata = {"barcodekey": barcodekey.values} # I guess the original matrix is processed in R because '-' -> '.'.
        feature_metadata = {"featurekey": df.index.values}
        matrices = {"Q3Norm": np.transpose(df.values)} # float64, do we need to convert it to float32?
        cur_matrix = "Q3Norm"

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
    else:
        df = pd.read_csv(input_matrix, sep = ',', header = None, index_col = 0)

        barcodekey = pd.Index(df.loc["Segment displayed name", 4:].values)
        barcode_metadata = {"barcodekey": barcodekey.values, 
                            "segment": pd.Categorical(df.loc["Segment (Name/ Label)", 4:].values), 
                            "AOI surface area": df.loc["AOI surface area", 4:].values.astype(np.float64), 
                            "AOI nuclei count": df.loc["AOI nuclei count", 4:].values.astype(np.int32)
                           }
        probe_pos = df.index.get_loc("#Probe Group") + 1
        df_probes = df.iloc[probe_pos:]
        rawmat = np.transpose(df_probes.loc[:, 4:].values.astype(np.float64))
        idx_signal = df_probes[2] == "Endogenous"
        idx_control = df_probes[2] == "Control"
        idx_negative = df_probes[2] == "Negative"

        feature_metadata = {"featurekey": df_probes.loc[idx_signal, 3].values, "ProbeAnnotation": df_probes.index[idx_signal].values}
        matrices = {"RawData": rawmat[:, idx_signal]} # float64
        metadata["control_names"] = df_probes.loc[idx_control, 3].values
        metadata["negative_names"] = df_probes.loc[idx_negative, 3].values
        barcode_multiarrays = {"controls": rawmat[:, idx_control], "negatives": rawmat[:, idx_negative]}

        cur_matrix = "RawData"

    if annotation_file is not None:
        if not os.path.isfile(annotation_file):
            raise FileNotFoundError(f"File {annotation_file} does not exist!")
        df = pd.read_csv(annotation_file, sep = '\t', header = 0, index_col = 0)
        assert barcodekey.isin(df.index).sum() == barcodekey.size
        df = df.reindex(barcodekey)
        for key in df.columns:
            barcode_metadata[key] = pd.Categorical(df[key].values)


    nanodata = NanostringData(barcode_metadata, feature_metadata, matrices, metadata, barcode_multiarrays = barcode_multiarrays, cur_matrix = cur_matrix)
    data = MultimodalData(nanodata)

    return data
