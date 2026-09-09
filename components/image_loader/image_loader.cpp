#include "image_loader.h"
#include <cstdint>
#include <opencv2/core.hpp>
#include "../c_utils/c_utils.hpp"
#include "../file_handler/file_handler.hpp"
#include "../image_container/image.hpp"

extern "C" {
    void load_image(const char* filepath) {
        if (!filepath) {
            return;
        }
        // 1. Carga multiformato sin alterar canales originales (IMREAD_UNCHANGED)
        cv::Mat image_temp;
        files_handler::load_img(filepath, image_temp);
        if (image_temp.empty()) {
            return;
        }
        // 2. Normalización según espacio de color de entrada
        image_utils::normalize_image(image_temp);

        int channels = image_temp.channels();
        if (channels != 1) {
            return;
        }
        // En este punto la imagen ya está en escala de grises normalizada a uint8
        int width = image_temp.cols;    // Número de columnas = ancho
        int height = image_temp.rows;   // Número de filas = alto

        ImageContainer* image = create_img_buffer(width, height, channels);

        uint8_t* img_ptr = image_get_data(image);
        if (!img_ptr) {
            return;
        }

        size_t total_bytes = image_temp.total() * image_temp.elemSize();
        size_t image_size = image_get_size(image);

        if (total_bytes != image_size) {
            return;
        }

        memcpy(img_ptr, image_temp.data, total_bytes);
        image_temp.release();   // Liberar imagen original inmediatamente
    };
}
