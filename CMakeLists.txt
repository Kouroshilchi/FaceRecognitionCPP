cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(FaceRecognitionCPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/std:c++17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
endif()

set(CMAKE_PREFIX_PATH "C:/libs/release/libtorch")
set(Torch_DIR "C:/libs/release/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(FaceRecognitionCPP "${CMAKE_CURRENT_SOURCE_DIR}/Project/model.cpp")

target_link_libraries(FaceRecognitionCPP ${TORCH_LIBRARIES})
if(MSVC)
    set_target_properties(FaceRecognitionCPP PROPERTIES 
        VS_DEBUGGER_ENVIRONMENT "PATH=%PATH%;${TORCH_INSTALL_PREFIX}/lib"
    )
endif()
target_compile_features(FaceRecognitionCPP PRIVATE cxx_std_17)
target_compile_options(FaceRecognitionCPP PRIVATE /std:c++17)

message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "C++ Flags: ${CMAKE_CXX_FLAGS}")