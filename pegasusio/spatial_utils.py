import os

import pandas as pd

from pegasusio import MultimodalData, SpatialData
from .hdf5_utils import load_10x_h5_file
import json
import numpy as np
from PIL import Image

def process_spatial_metadata(df):
    df['in_tissue'] = df['in_tissue'].apply(lambda n: True if n == 1 else False)
    df['barcodekey'] = df['barcodekey'].map(lambda s: s.split('-')[0])
    df.set_index('barcodekey', inplace=True)

def is_image(filename):
    return filename.endswith((".png", ".jpg")) 

def load_visium_folder(input_path) -> MultimodalData:
    file_list = os.listdir(input_path)
    sample_id = input_path.split("/")[-1]
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

    #  Store “pxl_col_in_fullres” and ”pxl_row_in_fullres” as a 2D array, 
    # which is the spatial location info of each cell in the dataset.
    obsm = spatial_metadata[['pxl_col_in_fullres', 'pxl_row_in_fullres']]
    barcode_multiarrays = {"spatial_coordinates":obsm.to_numpy()}
    
    #  Store all the other spatial info of cells, i.e. “in_tissue”, “array_row”, and “array_col”
    obs  = spatial_metadata[['in_tissue', 'array_row', 'array_col']]
    barcode_metadata = obs

     # Store image metadata as a Pandas DataFrame, with the following structure:
    img = pd.DataFrame()
    spatial_path = f"{input_path}/spatial"

    with open(f"{spatial_path}/scalefactors_json.json") as fp:
        scale_factors = json.load(fp)

    arr = os.listdir(spatial_path)
    for png in arr:
        if not is_image(png):
            continue
        if "hires" in png:
            with Image.open(f"{spatial_path}/{png}") as data:
                data.load()
            print("read img data:",type(data))
            dict = {"sample_id":sample_id, "image_id":"hires", "data":data, "scaleFactor":scale_factors["tissue_hires_scalef"]}
            img=img.append(dict, ignore_index=True)
        elif "lowres" in png:
            with Image.open(f"{spatial_path}/{png}") as data:
                data.load()
            print("read img data:",type(data))
            dict = {"sample_id":sample_id, "image_id":"lowres", "data":data, "scaleFactor":scale_factors["tissue_lowres_scalef"]}
            img=img.append(dict, ignore_index=True)

    assert not img.empty, "the image data frame is empty"
    spdata = SpatialData(
        barcode_metadata,
        feature_metadata,
        matrices,
        metadata,
        barcode_multiarrays=barcode_multiarrays,
        img=img,
    )
    data = MultimodalData(spdata)

    return data
