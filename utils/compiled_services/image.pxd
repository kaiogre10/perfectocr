# image.pxd
from libc.stdint cimport uint8_t

cdef extern from "image.hpp":
    cppclass Image:
        Image(int width, int height, int channels) except +
        uint8_t* data()
        int width()
        int height()
        int channels()
        size_t size()