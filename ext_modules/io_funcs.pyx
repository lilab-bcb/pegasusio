# cython: language_level=3, c_string_type=str, c_string_encoding=default

import numpy as np

from libc.stdlib cimport malloc, free, atoi
from libc.stdio cimport fopen, fclose, getline, FILE, fscanf, sscanf, fprintf, fread, fseek, SEEK_CUR, SEEK_SET#, printf
from libc.string cimport strncmp, strlen, strtok, strcmp, memcpy, memset
from libc.math cimport pow

cimport cython

ctypedef unsigned char uchar

ctypedef fused data_type:
    int
    long
    float
    double

ctypedef fused indices_type:
    int
    long

ctypedef fused indptr_type:
    int
    long



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
    assert sscanf(line, "%zu %zu %zu", &M, &N, &L) >= 0
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
cpdef void write_mtx(char* mtx_file, data_type[:] data, indices_type[:] indices, indptr_type[:] indptr, int M, int N, int precision = 2):
    """ Input is csr_matrix internal representation, cell by gene; Output will be gene by cell
    """
    cdef FILE* fo = fopen(mtx_file, "w")
    cdef str fmt_str = ""

    if (data_type is float) or (data_type is double):
        fprintf(fo, "%s\n", header_real)
        fmt_str = f"%d %d %.{precision}f\n"
    else:
        fprintf(fo, "%s\n", header_int)
        fmt_str = "%d %d %d\n"

    cdef const char* fmt = fmt_str
    cdef Py_ssize_t i, j

    fprintf(fo, "%s\n", metadata)
    fprintf(fo, "%d %d %zd\n", N, M, <long>data.size)

    for i in range(M):
        for j in range(indptr[i], indptr[i + 1]):
            fprintf(fo, fmt, indices[j] + 1, i + 1, data[j])

    fclose(fo)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int is_zero(char* pch):
    cdef int i = 0
    while pch[i] != 0:
        if pch[i] >= b'1' and pch[i] <= b'9':
            return 0
        i += 1
    return 1

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple read_csv(char* csv_file, char* delimiters):
    cdef FILE* fi = fopen(csv_file, "r")
    cdef char* line = NULL
    cdef char* pch = NULL
    cdef size_t size

    cdef str row_key
    cdef list colnames = [], rownames = [], row_ind = [], col_ind = [], data_list = []
    cdef int M = 0, N = 0
    cdef size_t L = 0
    cdef Py_ssize_t i

    assert getline(&line, &size, fi) >= 0
    pch = strtok(line, delimiters)
    assert pch != NULL
    row_key = pch

    pch = strtok(NULL, delimiters)
    while pch != NULL:
        colnames.append(pch)
        N += 1
        pch = strtok(NULL, delimiters)

    if N == 0:
        raise ValueError(f"File {csv_file} contains no columns!")

    colnames[N - 1] = colnames[N - 1].rstrip("\n\r")

    while getline(&line, &size, fi) >=0:
        if line[0] < 32 or line[0] == 127:
            continue
        pch = strtok(line, delimiters)
        assert pch != NULL
        rownames.append(pch)
        for i in range(N):
            pch = strtok(NULL, delimiters)
            assert pch != NULL
            if is_zero(pch) == 0:
                row_ind.append(M)
                col_ind.append(i)
                data_list.append(pch)
                if i == N - 1:
                    data_list[L] = data_list[L].rstrip("\r\n")
                L += 1
        M += 1

    free(line)
    fclose(fi)

    try:
        data = np.array(data_list, dtype = np.intc)
    except ValueError:
        data = np.array(data_list, dtype = np.float32)

    return row_ind, col_ind, data, (M, N), row_key, rownames, colnames


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void write_dense(char* output_file, str[:] barcodes, str[:] features, data_type[:] data, indices_type[:] indices, indptr_type[:] indptr, int M, int N, int precision = 2):
    """ Input must be csr_matrix internal representation, gene by cell (X.T.tocsr()); Output will be gene by cell
    """
    cdef FILE* fo = fopen(output_file, "w")
    cdef str fmt_str = f"\t%.{precision}f" if (data_type is float) or (data_type is double) else "\t%d"

    cdef const char* fmt = fmt_str
    cdef Py_ssize_t i, j, k, fr

    fprintf(fo, "GENE");
    for i in range(N):
        fprintf(fo, "\t%s", <char*>barcodes[i])
    fprintf(fo, "\n")

    for i in range(M):
        fprintf(fo, "%s", <char*>features[i]);
        fr = 0
        for j in range(indptr[i], indptr[i + 1]):
            for k in range(fr, indices[j]):
                fprintf(fo, "\t0")
            fprintf(fo, fmt, data[j])
            fr = indices[j] + 1
        for k in range(fr, N):
            fprintf(fo, "\t0")
        fprintf(fo, "\n")

    fclose(fo)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef swapbytes(uchar* bytes_buffer, int s):
    cdef int i, j
    cdef uchar tmp

    for i in range(s >> 1):
        j = s - i - 1
        tmp = bytes_buffer[i]
        bytes_buffer[i] = bytes_buffer[j]
        bytes_buffer[j] = tmp


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple read_fcs(char* fcs_file):
    """ This function is inspired by the python package fcsparser (https://github.com/eyurtsev/fcsparser)
    """
    cdef FILE* fi = fopen(fcs_file, "rb")

    # parse HEADER segment
    cdef char[7] fcs_format
    cdef char[9] header_field
    cdef size_t text_start, text_end, data_start, data_end, analysis_start, analysis_end

    fcs_format[6] = header_field[8] = 0
    assert fread(<void*>fcs_format, 1, 6, fi) == 6
    if strcmp(fcs_format, "FCS2.0") and strcmp(fcs_format, "FCS3.0") and strcmp(fcs_format, "FCS3.1"):
        fclose(fi)
        raise ValueError(f"Detected unsupported FCS format '{fcs_format}'!")

    assert fseek(fi, 4, SEEK_CUR) == 0
    assert fread(<void*>header_field, 1, 8, fi) == 8
    text_start = atoi(header_field)
    assert fread(<void*>header_field, 1, 8, fi) == 8
    text_end = atoi(header_field)
    assert fread(<void*>header_field, 1, 8, fi) == 8
    data_start = atoi(header_field)
    assert fread(<void*>header_field, 1, 8, fi) == 8
    data_end = atoi(header_field)
    assert fread(<void*>header_field, 1, 8, fi) == 8
    analysis_start = atoi(header_field)
    assert fread(<void*>header_field, 1, 8, fi) == 8
    analysis_end = atoi(header_field)

    if text_start < 58 or text_start >= text_end:
        fclose(fi)
        raise ValueError(f"Detected invalid TEXT segment start and end offsets: [{text_start}, {text_end}]!")

    if text_end == data_start:
        text_end -= 1

    # parse TEXT segment
    cdef char delim
    cdef size_t text_bytes, pos
    cdef uchar* text_segment

    unicode_source = np.zeros(100000, dtype = np.uint8)

    cdef uchar[:] sview = unicode_source
    cdef int s_pos # unicode source position
    cdef dict metadata = {}
    cdef bytes value_bytes
    cdef str key, value

    text_bytes = text_end - text_start + 1
    text_segment = <uchar*>malloc(text_bytes)
    assert text_segment != NULL
    assert fseek(fi, text_start, SEEK_SET) == 0
    assert fread(<void*>text_segment, 1, text_bytes, fi) == text_bytes

    # determine the end position of text segment
    delim = text_segment[0]
    pos = text_bytes - 1
    while text_segment[pos] != delim:
        pos -= 1
    text_bytes = pos + 1

    # generate key:value dictionary
    pos = 1
    s_pos = 0
    key = ""
    while pos < text_bytes:
        if text_segment[pos] == delim:
            if pos + 2 < text_bytes and text_segment[pos + 1] == delim: # delimiter escaping cannot happen at the end of the text segment
                sview[s_pos] = text_segment[pos]
                s_pos += 1
                pos += 1
            else:
                if s_pos == 0:
                    free(text_segment)
                    fclose(fi)
                    raise ValueError("Keywords and keyword values must have lengths greater than zero!")

                value_bytes = bytes(unicode_source[0:s_pos])
                try:
                    value = value_bytes.decode(encoding = 'utf-8')
                except UnicodeDecodeError:
                    value = value_bytes.decode(encoding = 'latin-1')

                s_pos = 0
                if key == "":
                    key = value.upper() # keywords are case insensitive; convert them to upper case
                else:
                    metadata[key] = value
                    key = ""
        else:
            sview[s_pos] = text_segment[pos]
            s_pos += 1

        pos += 1

        if s_pos >= 100000:
            free(text_segment)
            fclose(fi)
            raise NotImplementedError("Keyword or keyward value length cannot exceed 10,000!")

    free(text_segment)

    if key != "":
        fclose(fi)
        raise ValueError(f"Keyword {key} does not have a keyword value!")

    # sanity checks
    cdef size_t _start, _end

    _start = _end = 0
    if "$BEGINDATA" in metadata:
        _start = int(metadata.pop("$BEGINDATA"))
    if "$ENDDATA" in metadata:
        _end = int(metadata.pop("$ENDDATA"))
    if data_start == 0 and data_end == 0:
        data_start = _start
        data_end = _end
    if data_start == 0:
        print("Warning: Could not detect a DATA segment!")

    _start = _end = 0
    if "$BEGINANALYSIS" in metadata:
        _start = int(metadata.pop("$BEGINANALYSIS"))
    if "$ENDANALYSIS" in metadata:
        _end = int(metadata.pop("$ENDANALYSIS"))
    if analysis_start == 0 and analysis_end == 0:
        analysis_start = _start
        analysis_end = _end
    if analysis_start > 0:
        print("Warning: ANALYSIS segment would be omitted since current implementation does not support parsing of this segment.")

    if "$BEGINSTEXT" in metadata:
        if metadata.pop("$BEGINSTEXT") != "0":
            print("Warning: Detected a supplemental TEXT segment. This segment will be ignored under current implementation.")
    if "$ENDSTEXT" in metadata:
        del metadata["$ENDSTEXT"]

    # parse data segment
    cdef int M, N # M parameters and N events; matrix is N x M
    cdef str endian
    cdef char datatype
    cdef dict feature_metadata
    cdef Py_ssize_t i, j, k
    cdef str keyword, tmpstr

    for keyword in ["$NEXTDATA", "$MODE", "$PAR", "$TOT", "$BYTEORD", "$DATATYPE"]:
        if keyword not in metadata:
            fclose(fi)
            raise ValueError(f"Cannot detect requried keyword {keyword} in TEXT segment!")

    if metadata.pop("$NEXTDATA") != "0":
        print("Warning: Detected more than one data sets. Current implementation will only parse the first data set.")

    if metadata.pop("$MODE") != "L":
        fclose(fi)
        raise NotImplementedError("Current implementation can only handle $MODE = L!")

    endian = "little"
    if metadata.pop("$BYTEORD") in ["4,3,2,1", "2,1"]:
        endian = "big"

    datatype = ord(metadata.pop("$DATATYPE")[0])
    if datatype != b'I' and datatype != b'F' and datatype != b'D' and datatype != b'A':
        fclose(fi)
        raise ValueError(f"$DATATYPE {datatype} does not in 'IFDA'!")


    M = int(metadata.pop("$PAR"))
    N = int(metadata.pop("$TOT"))

    feature_metadata = {"featurekey": np.empty(M, dtype = object), "featureid": np.empty(M, dtype = object)}

    cdef str[:] fkview = feature_metadata["featurekey"]
    cdef str[:] fiview = feature_metadata["featureid"]

    PBs = np.zeros(M, dtype = np.intc)
    PRs = np.zeros(M, dtype = np.intc)
    PEs = np.empty(M, dtype = object)
    log_decades = np.zeros(M, dtype = np.float32)
    linear_values = np.zeros(M, dtype = np.float32)
    bit_masks = np.zeros(M, dtype = np.intc)

    cdef int[:] PBsview = PBs
    cdef int[:] PRsview = PRs
    cdef str[:] PEsview = PEs
    cdef float[:] ldview = log_decades
    cdef float[:] lvview = linear_values
    cdef int[:] bmview = bit_masks

    cdef str ldstr, lvstr
    cdef int total_bytes_per_row = 0

    for i in range(1, M + 1):
        for tmpstr in ["B", "E", "N", "R"]:
            keyword = f"$P{i}{tmpstr}"
            if keyword not in metadata:
                fclose(fi)
                raise ValueError(f"Cannot detect requried keyword {keyword} in TEXT segment!")

        pos = i - 1
        fiview[pos] = metadata.pop(f"$P{i}N")
        fkview[pos] = metadata.pop(f"$P{i}S", fiview[pos])
        PEsview[pos] = metadata.pop(f"$P{i}E")

        tmpstr = metadata.pop(f"$P{i}R")
        try:
            PRsview[pos] = int(tmpstr)
        except ValueError:
            PRsview[pos] = int(float(tmpstr))
            print(f"Warning: $P{i}R should be an integer! Rounded {tmpstr} to {PRsview[pos]}.")

        ldstr, lvstr = PEsview[pos].split(",")
        ldview[pos] = float(ldstr)
        lvview[pos] = float(lvstr)
        assert ldview[pos] >= 0.0 and lvview[pos] >= 0.0

        keyword = f"$P{i}B"

        if metadata[keyword] == "*":
            fclose(fi)
            if datatype == b'A':
                raise NotImplementedError(f"Current implementation does not support {keyword} = *!")
            else:
                raise ValueError(f"$DATATYPE '{datatype}' should have {keyword} = *!")

        PBsview[pos] = int(metadata.pop(keyword))

        if datatype == b'F':
            if PBsview[pos] != 32:
                fclose(fi)
                raise ValueError(f"$DATATYPE '{datatype}' must have {keyword} = 32 ({keyword} = {PBsview[pos]})!")
            PBsview[pos] = PBsview[pos] // 8
            if ldview[pos] > 0.0 or lvview[pos] > 0.0:
                fclose(fi)
                raise ValueError(f"$DATATYPE '{datatype}' must have $P{i}E = 0,0 ($P{i}E = {PEsview[pos]})!")
        elif datatype == b'D':
            if PBsview[pos] != 64:
                fclose(fi)
                raise ValueError(f"$DATATYPE '{datatype}' must have {keyword} = 64 ({keyword} = {PBsview[pos]})!")
            PBsview[pos] = PBsview[pos] // 8
            if ldview[pos] > 0.0 or lvview[pos] > 0.0:
                fclose(fi)
                raise ValueError(f"$DATATYPE '{datatype}' must have $P{i}E = 0,0 ($P{i}E = {PEsview[pos]})!")
        else:
            if datatype == b'I':
                if PBsview[pos] % 8 != 0:
                    fclose(fi)
                    raise NotImplementedError(f"Current implementation requires that {keyword} must be a multiple of 8 ({keyword} = {PBsview[pos]}) for $DATATYPE '{datatype}'!")
                PBsview[pos] = PBsview[pos] // 8
                if PBsview[pos] > 4:
                    fclose(fi)
                    raise NotImplementedError(f"Current implementation requires that {keyword} be no more than 4 bytes for $DATATYPE '{datatype}'!")
                bmview[pos] = 2 ** (PRsview[pos] - 1).bit_length() - 1
            else:
                # datatype must be 'A'
                if PBsview[pos] > 9:
                    fclose(fi)
                    raise NotImplementedError(f"Current implementation does not support {keyword} > 9 for $DATATYPE '{datatype}'!")

            if ldview[pos] > 0.0 or lvview[pos] > 0.0:
                if ldview[pos] == 0.0:
                    fclose(fi)
                    raise ValueError(f"We require f1 > 0 for $PnE/f1,f2/({PEsview[pos]})!")
                if lvview[pos] == 0.0:
                    lvview[pos] = 1.0
                    print(f"Warning: detected f2 = 0 for $P{i}E/{PEsview[pos]}/, which is not correct. Set f2 = 1 instead.")

        total_bytes_per_row += PBsview[pos]


    # SEEK data start
    data = np.zeros((N, M), dtype = np.float32)
    cdef float[:, :] data_view = data

    cdef size_t buffer_size = (<size_t>N) * (<size_t>total_bytes_per_row)
    cdef size_t data_size = data_end - data_start + 1
    if buffer_size != data_size and buffer_size != data_size - 1: # buffer_size == data_size - 1 because [data_start, data_end)
        print(f"Warning: Data size recorded in the DATA segment {data_end - data_start + 1} does not match the actual data size {buffer_size}.")

    cdef uchar* bytes_buffer = <uchar*>malloc(buffer_size + 1)
    assert bytes_buffer != NULL

    cdef uchar* buf = bytes_buffer

    from sys import byteorder
    cdef bint should_swap = endian != byteorder

    cdef int value_int
    cdef double value_double

    assert fseek(fi, data_start, SEEK_SET) == 0
    assert fread(<void*>bytes_buffer, 1, buffer_size, fi) == buffer_size

    for i in range(N):
        for j in range(M):
            if datatype != b'A' and should_swap:
                swapbytes(buf, PBsview[j])

            if datatype == b'D':
                memcpy(<void*>(&value_double), <void*>buf, PBsview[j])
                data_view[i, j] = <float>value_double
            elif datatype == b'F':
                memcpy(<void*>(&data_view[i, j]), <void*>buf, PBsview[j])
            else:
                if datatype == b'I':
                    memcpy(<void*>(&value_int), <void*>buf, PBsview[j])
                    value_int = value_int & bmview[j] # masking undefined bits
                else:
                    # datatype must be 'A'
                    value_int = 0
                    for k in range(PBsview[j]):
                        value_int = value_int * 10 + ((<char>buf[k]) - b'0')

                if ldview[j] == 0.0 and lvview[j] == 0.0:
                    data_view[i, j] = <float>value_int
                else:
                    data_view[i, j] = (<float>pow(10.0, ldview[j] * value_int / PRsview[j])) * lvview[j]

            buf += PBsview[j]

    free(bytes_buffer)
    fclose(fi)

    return feature_metadata, data, metadata
