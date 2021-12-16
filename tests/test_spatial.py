from operator import mul
import unittest

import pytest
import pandas as pd
from pegasusio import multimodal_data, readwrite, write_output
from pegasusio.spatial_utils import load_visium_folder
from pandas.testing import assert_series_equal

class TestSpatial(unittest.TestCase):

    @pytest.mark.pytest
    def test_spatial_zarr(self):
        data = load_visium_folder("/Users/rocherr/dev/LIB5432879_SAM24387106")
        print("1):",data)
        img = data.img
        obs = data.obs
        write_output(data, "spatial.zarr.zip")
        multimodal_data = readwrite.read_input("spatial.zarr.zip")
        print(multimodal_data)
        assert multimodal_data

        assert pd.DataFrame.equals(multimodal_data.img, img)
        assert pd.DataFrame.equals(multimodal_data.obs, obs)

    