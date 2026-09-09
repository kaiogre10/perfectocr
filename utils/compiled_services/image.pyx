# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from libc.stdint cimport uint8_t, uint16_t

cdef extern from "image.hpp":
    cdef cppclass Image:
        Image(uint16_t width, uint16_t height, uint8_t channels) except *

        uint8_t* data()
        const uint8_t* data() const

        uint16_t width() const
        uint16_t height() const
        uint8_t channels() const

cdef class FullImg:

    cdef Image* _image

    def __cinit__(self, uint16_t width, uint16_t height, uint8_t channels):
        self._image = new Image(width, height, channels)

    def __dealloc__(self):
        if self._image != NULL:
            del self._image
            self._image = NULL

    @property
    def full_img(self):
        cdef uint8_t* ptr = self._image.data()

        cdef Py_ssize_t size = (
            <Py_ssize_t>self._image.width()
            * <Py_ssize_t>self._image.height()
            * <Py_ssize_t>self._image.channels()
        )

        return <uint8_t[:size]>ptr
    
    @property
    def data(self):
        return self._image.data()

    @property
    def width(self):
        return self._image.width()

    @property
    def height(self):
        return self._image.height()

    @property
    def channels(self):
        return self._image.channels()