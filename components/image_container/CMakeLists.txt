add_library(image_container STATIC
	"image_container.c"
	"image_container.h"
    "image.hpp"
)

set_target_properties(image_container PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(image_container 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(image_container
    PUBLIC
        ${OS_FRONTIER_LIBS}
)
