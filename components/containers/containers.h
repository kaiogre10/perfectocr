#pragma once
#include <vector>
#include <cstdint>
#include <queue>
#include <image.hpp>

extern "C" {
    void container_create(int trigger);
    bool release_image(ImageContainer* image_ptr);
}

void push(std::vector<uint8_t>&& plain_payload);
std::queue<std::vector<uint8_t>> drain();
