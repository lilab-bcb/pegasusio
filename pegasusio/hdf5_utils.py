import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import h5py

from pegasusio import UnimodalData, MultimodalData



def load_10x_h5_file_v2(h5_in: h5py.Group, ngene: int = None) -> MultimodalData:
    """Load 10x v2 format matrix from hdf5 file

    Parameters
    ----------

    h5_in : h5py.Group
        An instance of h5py.Group class that is connected to a 10x v2 formatted hdf5 file.
    ngene : `int`, optional (default: None)
        Minimum number of genes to keep a barcode. Default is to keep all barcodes.

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
        	metadata = {"experiment_type": "rna", "genome": genome}
        )
        unidata.filter(ngene=ngene)
        unidata.separate_channels()

        data.add_data(genome, unidata)

    return data


def load_10x_h5_file_v3(h5_in: h5py.Group, ngene: int = None) -> MultimodalData:
    """Load 10x v3 format matrix from hdf5 file

    Parameters
    ----------

    h5_in : h5py.Group
        An instance of h5py.Group class that is connected to a 10x v3 formatted hdf5 file.
    ngene : `int`, optional (default: None)
        Minimum number of genes to keep a barcode. Default is to keep all barcodes.

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
    genomes = h5_in["matrix/features/genome"][...].astype(str)
    ids = h5_in["matrix/features/id"][...].astype(str)
    names = h5_in["matrix/features/name"][...].astype(str)

    data = MultimodalData()
    for genome in np.unique(genomes):
        idx = genomes == genome

        barcode_metadata = {"barcodekey": barcodes}
        feature_metadata = {"featurekey": names[idx], "featureid": ids[idx]}
        mat = bigmat[:, idx].copy()
        unidata = UnimodalData(barcode_metadata, feature_metadata, {"X": mat}, metadata = {"experiment_type": "rna", "genome": genome})
        unidata.filter(ngene=ngene)
        unidata.separate_channels()

        data.add_data(genome, unidata)

    return data


def load_10x_h5_file(input_h5: str, ngene: int = None) -> MultimodalData:
    """Load 10x format matrix (either v2 or v3) from hdf5 file

    Parameters
    ----------

    input_h5 : `str`
        The matrix in 10x v2 or v3 hdf5 format.
    ngene : `int`, optional (default: None)
        Minimum number of genes to keep a barcode. Default is to keep all barcodes.

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
        data = load_file(h5_in, ngene)

    return data


def load_pegasus_h5_file(
    input_h5: str, ngene: int = None, select_singlets: bool = False
) -> MultimodalData:
    """Load matrices from pegasus-format hdf5 file (deprecated)

    Parameters
    ----------

    input_h5 : `str`
        pegasus-format hdf5 file.
    ngene : `int`, optional (default: None)
        Minimum number of genes to keep a barcode. Default is to keep all barcodes.
    select_singlets: `bool`, optional (default: False)
        If only load singlets.

    Returns
    -------

    A MultimodalData object containing (genome, UnimodalData) pair per genome.

    Examples
    --------
    >>> io.load_pegasus_h5_file('example.h5sc')
    """
    cite_seq_name = None
    selected_barcodes = None

    data = MultimodalData()
    with h5py.File(input_h5, 'r') as h5_in:
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

            barcode_metadata = {}
            sub_group = group["_barcodes"]
            for key in sub_group.keys():
                if key != "barcodekey":
                    continue
                values = sub_group[key][...]
                if values.dtype.kind == "S":
                    values = values.astype(str)
                barcode_metadata[key] = values

            feature_metadata = {}
            sub_group = group["_features"]
            for key in sub_group.keys():
                values = sub_group[key][...]
                if values.dtype.kind == "S":
                    values = values.astype(str)
                if key == "featurename":
                    key = "featurekey"
                elif key == "featurekey":
                    key = "featureid"                    
                feature_metadata[key] = values

            is_citeseq = genome.startswith("CITE_Seq")
            unidata = UnimodalData(barcode_metadata, feature_metadata, {"X": mat}, metadata = {"experiment_type": "citeseq" if is_citeseq else "rna", "genome": genome})

            if is_citeseq:
                cite_seq_name = genome
            else:
                unidata.filter(ngene, select_singlets)
                selected_barcodes = unidata.obs_names

            data.add_data(genome, unidata)

    if (cite_seq_name is not None) and (selected_barcodes is not None):
        unidata = data.get_data(cite_seq_name)
        selected = unidata.obs_names.isin(selected_barcodes)
        unidata.trim(selected)

    return data
