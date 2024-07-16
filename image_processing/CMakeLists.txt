cmake_minimum_required(VERSION 3.16.3)
project(ImageProcessing VERSION 0.9.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# For profiling performance
if(PROFILE)
    add_definitions(-DPROFILE)
endif()

add_definitions(-DCL_TARGET_OPENCL_VERSION=300)

# Check for CUDA
find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "CUDA found, setting custom paths for OpenCL.")
    set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} /usr/local/cuda-12.3/targets/x86_64-linux/include)
    set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /usr/local/cuda-12.3/targets/x86_64-linux/lib64)
else()
    message(STATUS "CUDA not found, using default OpenCL paths.")
endif()

# Build program
set(APPLICATION_SOURCE 
    src/main.cpp
    src/image.cpp
    src/opencl_image.cpp
)

set(APPLICATION_HEADERS 
    include/image.h
    include/stb_image_write.h
    include/stb_image.h
    include/opencl_image.h
    include/masks.h
)

add_executable(${PROJECT_NAME} ${APPLICATION_SOURCE} ${APPLICATION_HEADERS})

set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

set(INCLUDE_DIRS /user/include ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIRS})

target_link_directories(${PROJECT_NAME} PRIVATE /usr/lib/x86_64-linux-gnu)
find_package(OpenMP REQUIRED)
find_package(OpenCL REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCL_LIBRARIES} OpenMP::OpenMP_CXX)


set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Create output directories if they do not exist
add_custom_target(create_output_dir ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
)

add_dependencies(${PROJECT_NAME} create_output_dir)

# Custom commands for clean
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}
)

# Custom target for 'r'
add_custom_target(r
    COMMAND ${CMAKE_COMMAND} -E echo "Running ${PROJECT_NAME}"
    COMMAND ${PROJECT_NAME}
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)








