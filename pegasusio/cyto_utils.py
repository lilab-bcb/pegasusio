import os
import numpy as np
import pandas as pd

from pegasusio import CytoData, MultimodalData


def load_fcs_file(input_fcs: str, genome: str = None) -> MultimodalData:
    """Load Cyto data from a FCS file, support v2.0, v3.0 and v3.1.

    Parameters
    ----------

    input_fcs : `str`
        The FCS file.
    genome : `str`, optional (default None)
        The genome reference. If None, use "unknown" instead.

    Returns
    -------

    A MultimodalData object containing a (genome, CytoData) pair.

    Examples
    --------
    >>> io.load_fcs_file('example.fcs', genome = 'GRCh38')
    """
    try:
        from pegasusio.cylib.io import read_fcs
    except ModuleNotFoundError:
        print("No module named 'pegasusio.cylib.io'")

    if not os.path.isfile(input_fcs):
        raise FileNotFoundError(f"File {input_fcs} does not exist!")
    feature_metadata, matrix, metadata = read_fcs(input_fcs)
    barcode_metadata = {"barcodekey": [f"event{i}" for i in range(1, matrix.shape[0] + 1)]}
    genome = "unknown" if genome is None else genome
    metadata["genome"] = genome
    metadata["modality"] = "cyto"

    cytodata = CytoData(barcode_metadata, feature_metadata, {"raw.data": matrix}, metadata)
    data = MultimodalData(cytodata)

    return data
