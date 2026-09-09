add_library(c_utils STATIC
	"c_utils.cpp"
	"c_utils.hpp"
)

set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../opencv_dep/build")
if(NOT EXISTS "${OpenCV_DIR}/OpenCVConfig.cmake")
    set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencv_dep/install/include/opencv2")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

target_include_directories(c_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(c_utils PRIVATE
    ${OpenCV_LIBS}
    ${OS_FRONTIER_LIBS}
)
