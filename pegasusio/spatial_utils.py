import os, re

import pandas as pd

from pegasusio import MultimodalData, SpatialData
from .hdf5_utils import load_10x_h5_file
import json
from PIL import Image


def process_spatial_metadata(df):
    df["in_tissue"] = df["in_tissue"].apply(lambda n: True if n == 1 else False)
    df["barcodekey"] = df["barcodekey"].map(lambda s: s.split("-")[0])
    df.set_index("barcodekey", inplace=True)


def load_visium_folder(input_path) -> MultimodalData:
    """
    Method to read the visium spatial data folder
    into MultimodalData object that contains SpatialData
    """
    file_list = os.listdir(input_path)
    sample_id = input_path.split("/")[-1]
    # Load count matrix.
    hdf5_filename = "filtered_feature_bc_matrix.h5"
    assert (
        hdf5_filename in file_list
    ), "Filtered feature-barcode matrix in HDF5 format is missing!"
    rna_data = load_10x_h5_file(f"{input_path}/{hdf5_filename}")

    # Load spatial metadata.
    assert ("spatial" in file_list) and (
        os.path.isdir(f"{input_path}/spatial")
    ), "Spatial folder is missing!"

    tissue_pos_csv = f"{input_path}/spatial/tissue_positions.csv"
    if os.path.exists(tissue_pos_csv):
        spatial_metadata = pd.read_csv(tissue_pos_csv)
        spatial_metadata.rename(columns={"barcode": "barcodekey"}, inplace=True)
    else:
        tissue_pos_csv = f"{input_path}/spatial/tissue_positions_list.csv"
        assert os.path.exists(tissue_pos_csv), f"Cannot locate file {tissue_pos_csv}!"
        spatial_metadata = pd.read_csv(
            tissue_pos_csv,
            names=[
                "barcodekey",
                "in_tissue",
                "array_row",
                "array_col",
                "pxl_row_in_fullres",
                "pxl_col_in_fullres",
            ],
        )
    process_spatial_metadata(spatial_metadata)

    barcode_metadata = rna_data.obs.join(spatial_metadata, how="left")
    feature_metadata = rna_data.var

    matrices = {"X": rna_data.X}
    metadata = {"genome": rna_data.get_genome(), "modality": "visium"}

    #  Store “pxl_col_in_fullres” and ”pxl_row_in_fullres” as a 2D array, which is the spatial location info of each cell in the dataset.
    barcode_multiarrays = {
        "X_spatial": barcode_metadata[
            ["pxl_col_in_fullres", "pxl_row_in_fullres"]
        ].to_numpy()
    }
    barcode_metadata.drop(
        columns=["pxl_row_in_fullres", "pxl_col_in_fullres"], inplace=True
    )

    # Store image metadata as a Pandas DataFrame, with the following structure:
    image_metadata = pd.DataFrame(
        columns=["sample_id", "image_id", "data", "scale_factor", "spot_diameter"]
    )
    spatial_path = f"{input_path}/spatial"

    with open(f"{spatial_path}/scalefactors_json.json") as fp:
        scale_factors = json.load(fp)

    def get_image_data(
        filepath, sample_id, image_id, scaleFactor, spot_diameter_fullres
    ):
        data = Image.open(filepath)
        return pd.DataFrame(
            {
                "sample_id": [sample_id],
                "image_id": [image_id],
                "data": [data],
                "scale_factor": [scaleFactor],
                "spot_diameter": [spot_diameter_fullres * scaleFactor],
            }
        )

    for png in [f for f in os.listdir(spatial_path) if re.match(".*\.png", f)]:
        if ("_hires_" in png) or ("_lowres_" in png):
            filepath = f"{spatial_path}/{png}"
            res_tag = "hires" if "_hires_" in png else "lowres"
            image_item = get_image_data(
                filepath,
                sample_id,
                res_tag,
                scale_factors[f"tissue_{res_tag}_scalef"],
                scale_factors["spot_diameter_fullres"],
            )
            image_metadata = pd.concat([image_metadata, image_item], ignore_index=True)

    assert not image_metadata.empty, "the image data frame is empty"
    spdata = SpatialData(
        barcode_metadata,
        feature_metadata,
        matrices,
        metadata,
        barcode_multiarrays=barcode_multiarrays,
        image_metadata=image_metadata,
    )
    data = MultimodalData(spdata)

    return data
