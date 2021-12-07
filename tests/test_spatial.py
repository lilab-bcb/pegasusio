import unittest

import pytest

from pegasusio import readwrite as io
from pegasusio.spatial_utils import load_visium_folder

class TestSpatial(unittest.TestCase):

    @pytest.mark.pytest
    def test_h5ad(self):
        print("hello world")
        data = load_visium_folder("/Users/rocherr/dev/LIB5432879_SAM24387106")
        # print("obsm:\n")
        # print(data.get_data(modality='visium').obsm)
        # print("obs:\n")
        # print(data.get_data(modality='visium').obs)
        # print("obs_names:\n")
        # print(data.get_data(modality='visium').obs_names)
        # print("feature_metadata:\n")
        # print(data.get_data(modality='visium').feature_metadata)
        # print("barcode_metadata:\n")
        # print(data.get_data(modality='visium').barcode_metadata)
        # print("barcode_multiarrays:\n")
        # print(data.get_data(modality='visium').barcode_multiarrays.items())
        # print(data.get_data().obsm)
        print("img:",data.img)

        print("obs:", data.obs)


        print("obsm:", data.obsm["spatial_coordinates"])