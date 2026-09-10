# components/image_loader
add_library(image_loader STATIC
    "image_loader.cpp"
    "image_loader.hpp"
)

set_target_properties(image_loader PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../opencv_dep/build")
if(NOT EXISTS "${OpenCV_DIR}/OpenCVConfig.cmake")
    set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencv_dep/install/include/opencv2")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

target_include_directories(image_loader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(image_loader PRIVATE
	file_handler
    c_utils
    image_container
    ${OpenCV_LIBS}
    ${OS_FRONTIER_LIBS}
)
