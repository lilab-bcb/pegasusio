import unittest

import pegasusio as io

class TestIO(unittest.TestCase):

    def test_h5ad(self):
        data = io.read_input("tests/pegasusio-test-data/case1/pbmc3k.h5ad", genome = 'hg19')
        self.assertEqual(data.shape, (2638, 1838))

if __name__ == '__main__':
    unittest.main()
