import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import h5py
from typing import Dict

import logging
logger = logging.getLogger(__name__)

from pegasusio import modalities, UnimodalData, CITESeqData, MultimodalData


def load_10x_h5_file_v2(h5_in: h5py.Group) -> MultimodalData:
    """Load 10x v2 format matrix from hdf5 file

    Parameters
    ----------

    h5_in : h5py.Group
        An instance of h5py.Group class that is connected to a 10x v2 formatted hdf5 file.

    Returns
    -------

    A MultimodalData object containing (genome, UnimodalData) pair per genome.

    Examples
    --------
    >>> io.load_10x_h5_file_v2(h5_in)
    """
    data = MultimodalData()
    for genome in h5_in.keys():
        group = h5_in[genome]

        M, N = group["shape"][...]
        mat = csr_matrix(
            (
                group["data"][...],
                group["indices"][...],
                group["indptr"][...],
            ),
            shape=(N, M),
        )

        barcodes = group["barcodes"][...].astype(str)
        ids = group["genes"][...].astype(str)
        names = group["gene_names"][...].astype(str)

        unidata = UnimodalData({"barcodekey": barcodes},
            {"featurekey": names, "featureid": ids},
            {"X": mat},
            {"modality": "rna", "genome": genome}
        )
        unidata.separate_channels()

        data.add_data(unidata)

    return data


def load_10x_h5_file_v3(h5_in: h5py.Group) -> MultimodalData:
    """Load 10x v3 format matrix from hdf5 file, allowing detection of crispr and citeseq libraries

    Parameters
    ----------

    h5_in : h5py.Group
        An instance of h5py.Group class that is connected to a 10x v3 formatted hdf5 file.

    Returns
    -------

    A MultimodalData object containing (genome, UnimodalData) pair per genome.

    Examples
    --------
    >>> io.load_10x_h5_file_v3(h5_in)
    """
    M, N = h5_in["matrix/shape"][...]
    bigmat = csr_matrix(
        (
            h5_in["matrix/data"][...],
            h5_in["matrix/indices"][...],
            h5_in["matrix/indptr"][...],
        ),
        shape=(N, M),
    )
    barcodes = h5_in["matrix/barcodes"][...].astype(str)
    df = pd.DataFrame(data = {"genome": h5_in["matrix/features/genome"][...].astype(str),
                              "feature_type": h5_in["matrix/features/feature_type"][...].astype(str),
                              "id": h5_in["matrix/features/id"][...].astype(str),
                              "name": h5_in["matrix/features/name"][...].astype(str)})

    genomes = list(df["genome"].unique())
    if "" in genomes:
        genomes.remove("")
    default_genome = genomes[0] if len(genomes) == 1 else None

    data = MultimodalData()
    gb = df.groupby(by = ["genome", "feature_type"])
    for name, group in gb:
        barcode_metadata = {"barcodekey": barcodes}
        feature_metadata = {"featurekey": group["name"].values, "featureid": group["id"].values}
        mat = bigmat[:, gb.groups[name]]

        genome = name[0] if (name[0] != "" or default_genome is None) else default_genome
        modality = "custom"
        if name[1] == "Gene Expression":
            modality = "rna"
        elif name[1] == "CRISPR Guide Capture":
            modality = "crispr"
        elif name[1] == "Antibody Capture":
            modality = "citeseq"

        if modality == "citeseq":
            unidata = CITESeqData(barcode_metadata, feature_metadata, {"raw.count": mat}, {"genome": genome, "modality": modality})
        else:
            unidata = UnimodalData(barcode_metadata, feature_metadata, {"X": mat}, {"genome": genome, "modality": modality})
        unidata.separate_channels()

        data.add_data(unidata)

    return data


def load_10x_h5_file(input_h5: str) -> MultimodalData:
    """Load 10x format matrix (either v2 or v3) from hdf5 file

    Parameters
    ----------

    input_h5 : `str`
        The matrix in 10x v2 or v3 hdf5 format.

    Returns
    -------

    A MultimodalData object containing (genome, UnimodalData) pair per genome.

    Examples
    --------
    >>> io.load_10x_h5_file('example_10x.h5')
    """
    data = None
    with h5py.File(input_h5, 'r') as h5_in:
        load_file = load_10x_h5_file_v3 if "matrix" in h5_in.keys() else load_10x_h5_file_v2
        data = load_file(h5_in)

    return data


def load_loom_file(input_loom: str, genome: str = None, modality: str = None) -> MultimodalData:
    """Load count matrix from a LOOM file.

    Parameters
    ----------

    input_loom : `str`
        The LOOM file, containing the count matrix.
    genome : `str`, optional (default None)
        The genome reference. If None, use "unknown" instead. If not None and input loom contains genome attribute, the attribute will be overwritten.
    modality: `str`, optional (default None)
        Modality. If None, use "rna" instead. If not None and input loom contains modality attribute, the attribute will be overwritten.

    Returns
    -------

    A MultimodalData object containing a (genome, UmimodalData) pair.

    Examples
    --------
    >>> io.load_loom_file('example.loom', genome = 'GRCh38')
    """
    col_trans = {"CellID": "barcodekey", "obs_names": "barcodekey", "cell_names": "barcodekey"}
    row_trans = {
        "Gene": "featurekey", "var_names": "featurekey", "gene_names": "featurekey",
        "Accession": "featureid",  "gene_ids": "featureid", "ensembl_ids": "featureid",
    }

    import loompy
    with loompy.connect(input_loom) as ds:
        barcode_metadata = {}
        barcode_multiarrays = {}
        for key, arr in ds.col_attrs.items():
            key = col_trans.get(key, key)
            if arr.ndim == 1:
                barcode_metadata[key] = arr
            elif arr.ndim > 1:
                barcode_multiarrays[key] = arr
            else:
                raise ValueError(f"Detected column attribute '{key}' has ndim = {arr.ndim}!")

        feature_metadata = {}
        feature_multiarrays = {}
        for key, arr in ds.row_attrs.items():
            key = row_trans.get(key, key)
            if arr.ndim == 1:
                feature_metadata[key] = arr
            elif arr.ndim > 1:
                feature_multiarrays[key] = arr
            else:
                raise ValueError(f"Detected row attribute '{key}' has ndim = {arr.ndim}!")

        matrices = {}
        for key, mat in ds.layers.items():
            key = "X" if key == "" else key
            matrices[key] = mat.sparse().T.tocsr()

        metadata = dict(ds.attrs)
        if genome is not None:
            metadata["genome"] = genome
        elif "genome" not in metadata:
            metadata["genome"] = "unknown"

        if modality is not None:
            metadata["modality"] = modality
        elif "modality" not in metadata:
            if metadata.get("experiment_type", "none") in modalities:
                metadata["modality"] = metadata.pop("experiment_type")
            else:
                metadata["modality"] = "rna"

        unidata = UnimodalData(barcode_metadata, feature_metadata, matrices, metadata, barcode_multiarrays, feature_multiarrays)
        unidata.separate_channels()

    data = MultimodalData(unidata)
    return data


def write_loom_file(data: MultimodalData, output_file: str) -> None:
    """ Write a MultimodalData to loom file. Will assert data only contain one type of experiment.
    """
    keys = data.list_data()
    if len(keys) > 1:
        raise ValueError(f"Data contain multiple modalities: {','.join(keys)}!")
    data.select_data(keys[0])
    matrices = data.list_keys()
    assert "X" in matrices
    if len(matrices) == 0:
        raise ValueError("Could not write empty matrix to a loom file!")

    def _process_attrs(key_name: str, attrs: pd.DataFrame, attrs_multi: dict) -> Dict[str, object]:
        res_dict = {key_name: attrs.index.values}
        for key in attrs.columns:
            res_dict[key] = np.array(attrs[key].values)
        for key, value in attrs_multi.items():
            if value.ndim > 1: # value.ndim == 1 refers to np.recarray, which will not be written to a loom file.
                res_dict[key] = value if value.shape[1] > 1 else value[:, 0]
        return res_dict

    row_attrs = _process_attrs("Gene", data.var, data.varm)
    col_attrs = _process_attrs("CellID", data.obs, data.obsm)

    accession_key = "featureid" if "featureid" in row_attrs else ("gene_ids" if "gene_ids" in row_attrs else None)
    if accession_key is not None:
        row_attrs["Accession"] = row_attrs.pop(accession_key)

    layers = {}
    for matkey in matrices:
        layers["" if matkey == "X" else matkey] = data.get_matrix(matkey).T

    file_attrs = {}
    for key, value in data.uns.items():
        if isinstance(value, str):
            file_attrs[key] = value

    import loompy
    loompy.create(output_file, layers, row_attrs, col_attrs, file_attrs = file_attrs)

    logger.info(f"{output_file} is written.")
