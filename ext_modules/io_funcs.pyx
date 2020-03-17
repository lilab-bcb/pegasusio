# cython: language_level=3, c_string_type=str, c_string_encoding=default

import numpy as np

from libc.stdlib cimport malloc, free
from libc.stdio cimport fopen, fclose, getline, FILE, fscanf, sscanf, fprintf
from libc.string cimport strncmp, strlen

cimport cython


cdef const char* header_real = b"%%MatrixMarket matrix coordinate real general"
cdef const char* header_int = b"%%MatrixMarket matrix coordinate integer general"
cdef const char* metadata = b"%metadata_json: {\"software\": \"pegasusio\"}"


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple read_mtx(char* mtx_file):
	cdef FILE* fi = fopen(mtx_file, "r")
	cdef char* line = NULL
	cdef size_t size
	cdef char isreal = 0


	assert getline(&line, &size, fi) >= 0
	if strncmp(line, header_real, strlen(header_real)) == 0:
		isreal = 1
	elif strncmp(line, header_int, strlen(header_int)) == 0:
		isreal = 0
	else:
		raise ValueError("Cannot recognize header line : " + line)

	while getline(&line, &size, fi) >=0:
		if line[0] != b'%':
			break

	cdef size_t M, N, L, i

	assert line[0] != b'%'
	assert sscanf(line, "%zd %zd %zd", &M, &N, &L) >= 0
	free(line)

	row_ind = np.zeros(L, dtype = np.intc)
	col_ind = np.zeros(L, dtype = np.intc)
	data = np.zeros(L, dtype = np.float32 if isreal else np.intc)

	cdef int[:] rview = row_ind, cview = col_ind
	cdef int[:] dview_int
	cdef float[:] dview_float

	cdef int x, y, value_int
	cdef float value_float

	if isreal:
		dview_float = data
		for i in range(L):
			assert fscanf(fi, "%d %d %f", &x, &y, &value_float) >= 0
			rview[i] = x - 1
			cview[i] = y - 1
			dview_float[i] = value_float
	else:
		dview_int = data
		for i in range(L):
			assert fscanf(fi, "%d %d %d", &x, &y, &value_int) >= 0
			rview[i] = x - 1
			cview[i] = y - 1
			dview_int[i] = value_int

	fclose(fi)

	return row_ind, col_ind, data, (M, N)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void write_mtx(char* mtx_file, object data, int[:] indices, int[:] indptr, int M, int N, int precision = 2):
	""" Input is csr_matrix internal representation, cell by gene; Output will be gene by cell
	"""
	cdef FILE* fo = fopen(mtx_file, "w")
	cdef str fmt_str = ""
	cdef char is_real = 0

	cdef int[:] data_int
	cdef float[:] data_float

	if data.dtype.kind == 'f':
		fprintf(fo, "%s\n", header_real)
		fmt_str = "%d %d %.{}f\n".format(precision)
		is_real = 1
		data_float = data
	elif data.dtype.kind == 'i':
		fprintf(fo, "%s\n", header_int)
		fmt_str = "%d %d %d\n"
		is_real = 0
		data_int = data
	else:
		raise ValueError("Detected unknown dtype: {}!".format(data.dtype))

	cdef const char* fmt = fmt_str
	cdef Py_ssize_t data_size = data.size
	cdef Py_ssize_t i, j

	fprintf(fo, "%s\n", metadata)
	fprintf(fo, "%d %d %zd\n", N, M, data_size)

	for i in range(M):
		for j in range(indptr[i], indptr[i + 1]):
			if is_real:
				fprintf(fo, fmt, indices[j] + 1, i + 1, data_float[j])
			else:
				fprintf(fo, fmt, indices[j] + 1, i + 1, data_int[j])

	fclose(fo)
