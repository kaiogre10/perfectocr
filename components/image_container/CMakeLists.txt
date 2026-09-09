add_library(image_container STATIC
	"image.hpp"
	"image_container.c"
	"image_container.h"
)

target_include_directories(image_container 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(image_container
    PUBLIC
        ${OS_FRONTIER_LIBS}
)
