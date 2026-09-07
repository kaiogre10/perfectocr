#pragma once
#include <opencv2/imgcodecs.hpp>
#include <cstdint>

extern "C" {
    void load_image(const char* filepath);
}
