#pragma once
#include <opencv2/imgcodecs.hpp>
#include <cstdint>
#include "../image_container/image.hpp"

extern "C" {
    void load_image(const char* filepath);
}
