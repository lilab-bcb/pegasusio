import os

import pandas as pd

from pegasusio import MultimodalData, SpatialData
from .hdf5_utils import load_10x_h5_file


def process_spatial_metadata(df):
    df['in_tissue'] = df['in_tissue'].apply(lambda n: True if n == 1 else False)
    df['barcodekey'] = df['barcodekey'].map(lambda s: s.split('-')[0])
    df.set_index('barcodekey', inplace=True)

def load_visium_folder(input_path) -> MultimodalData:
    file_list = os.listdir(input_path)

    # Load count matrix.
    hdf5_filename = "raw_feature_bc_matrix.h5"
    assert hdf5_filename in file_list, "Raw count hdf5 file is missing!"
    rna_data = load_10x_h5_file(f"{input_path}/{hdf5_filename}")

    # Load spatial metadata.
    assert ("spatial" in file_list) and (os.path.isdir(f"{input_path}/spatial")), "Spatial folder is missing!"
    tissue_pos_csv = "spatial/tissue_positions_list.csv"
    spatial_metadata = pd.read_csv(
        f"{input_path}/{tissue_pos_csv}",
        names=['barcodekey', 'in_tissue', 'array_row', 'array_col', 'pxl_col_in_fullres', 'pxl_row_in_fullres'],
    )
    process_spatial_metadata(spatial_metadata)

    barcode_metadata = pd.concat([rna_data.obs, spatial_metadata], axis=1)
    feature_metadata = rna_data.var

    matrices = {'raw.data': rna_data.X}
    metadata = {'genome': rna_data.get_genome(), 'modality': 'visium'}

    spdata = SpatialData(barcode_metadata, feature_metadata, matrices, metadata)
    data = MultimodalData(spdata)

    return data
