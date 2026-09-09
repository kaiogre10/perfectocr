# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from cpython.unicode cimport PyUnicode_GET_LENGTH, PyUnicode_FromStringAndSize
from cpython.unicode cimport PyUnicode_GET_LENGTH, PyUnicode_1BYTE_DATA, PyUnicode_FromStringAndSize
from cpython.bytes cimport PyBytes_GET_SIZE, PyBytes_AS_STRING

cdef extern from "buffer_handler.h":
    void create_deque()
