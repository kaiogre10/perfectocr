add_library(file_handler STATIC
	"file_handler.cpp"
	"file_handler.hpp"
)

set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../opencv_dep/build")
if(NOT EXISTS "${OpenCV_DIR}/OpenCVConfig.cmake")
    set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencv_dep/install/include/opencv2")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

target_include_directories(file_handler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(file_handler PRIVATE
    ${OpenCV_LIBS}
    ${OS_FRONTIER_LIBS}
)
