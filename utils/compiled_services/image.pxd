# image.pxd
from libc.stdint cimport uint8_t

cdef extern from "image.hpp":
    cppclass Image:
        Image(int width, int height, int channels) except +
        uint8_t* data()
        const uint8_t* data() const
        int width()
        int height()
        int channels()
        size_t size()

cdef extern from *:
    """
    #ifndef DESTROY_IMAGE_DEFINED
    #define DESTROY_IMAGE_DEFINED
    #include "image.hpp"
    inline void destroy_image(Image* img) noexcept {
        delete img;
    }
    #endif
    """
    void destroy_image(Image* img) noexcept