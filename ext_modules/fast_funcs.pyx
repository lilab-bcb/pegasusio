# cython: language_level=3

import numpy as np
cimport cython

ctypedef unsigned char uint8


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
        cview[i] = res[1]

    return (barcodes, channels)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple convert_10x_vdj_to_vdjdata(str[:] tokens, int[:, :] mats_int, str[:, :] mats_str, dict fid2pos, int n_barcodes, int n_contigs):
    cdef Py_ssize_t i, j
    cdef str chain, strval
    cdef int bpos = -1, fpos, num, delta, value
    cdef dict fmap = {}, strmap
    cdef list tracing = []
    cdef int n_features = len(fid2pos), nint = mats_int.shape[1], nstr = mats_str.shape[1], nmat = nint + nstr - 2

    barcodes = np.empty(n_barcodes, dtype = np.object)
    is_cell = np.zeros(n_barcodes, dtype = np.bool_)
    matrices = np.zeros((nmat, n_barcodes, n_features), dtype = np.int32)
    cdef list strarrs = []

    cdef str[:] bview = barcodes
    cdef uint8[:] icview = is_cell
    cdef int[:, :, :] matview = matrices
    cdef str[:] strview


    for i in range(nstr - 1):
        tracing.append({"None": 0, "": 0})

    for i in range(tokens.size):
        if bpos < 0 or bview[bpos] != tokens[i]:
            bpos += 1
            bview[bpos] = tokens[i]
            icview[bpos] = mats_int[i, nint - 1]
            fmap.clear()

        chain = mats_str[i, nstr - 1]
        num = fmap.get(chain, 0)
        if num >= n_contigs:
            raise ValueError(f"There are more than {n_contigs} productive contigs with chain type {chain}!")
        fpos = fid2pos[chain + (str(num + 1) if num > 0 else "")]
        for j in range(nint - 1):
            matview[j, bpos, fpos] = mats_int[i, j]
        delta = nint - 1
        for j in range(nstr - 1):
            strmap = tracing[j]
            strval = mats_str[i, j]
            value = strmap.get(strval, -1)
            if value < 0:
                value = len(strmap) - 1
                strmap[strval] = value
            matview[j + delta, bpos, fpos] = value
        fmap[chain] = num + 1

    for i in range(nstr - 1):
        del tracing[i]["None"]
        strmap = tracing[i]
        strarr = np.empty(len(strmap), dtype = np.object)
        strview = strarr
        for key, value in strmap.items():
            strview[value] = key
        strarrs.append(strarr)

    return barcodes, is_cell, matrices, strarrs
