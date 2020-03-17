# cython: language_level=3

import numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple split_barcode_channel(str[:] arr):
	cdef Py_ssize_t size = arr.size, i

	barcodes = np.empty(size, dtype = np.object)
	channels = np.empty(size, dtype = np.object)

	cdef str[:] bview = barcodes, cview = channels
	cdef list res

	for i in range(size):
		res = arr[i].rsplit(sep = '-', maxsplit = 1)
		bview[i] = res[0]
		if len(res) > 1:
			cview[i] = res[1]

	return (barcodes, channels)
