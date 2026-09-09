add_library(payload_container STATIC
    "payload_container.hpp" 
    "payload_container.cpp"
)

target_include_directories(payload_container 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(payload_container 
    PUBLIC
        ${OS_FRONTIER_LIBS}
)