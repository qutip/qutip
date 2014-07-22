
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

ITYPE = np.int32
ctypedef np.int32_t ITYPE_t

CTYPE = np.complex128
ctypedef np.complex128_t CTYPE_t

# Fused type for int32 and int64
ctypedef fused int32_or_int64:
    np.int32_t
    np.int64_t