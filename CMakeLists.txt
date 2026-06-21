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

set(OpenCV_DIR "C:/libs/opencv/build")  
find_package(OpenCV REQUIRED)    

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB_RECURSE PROJECT_SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/Project/src/*.cpp"
)

add_executable(FaceRecognitionCPP 
    "${CMAKE_CURRENT_SOURCE_DIR}/Project/main.cpp"
    ${PROJECT_SOURCE_FILES}
)

if(MSVC)
    add_custom_command(TARGET FaceRecognitionCPP POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OpenCV_DIR}/x64/vc16/bin/opencv_world4120.dll"
            $<TARGET_FILE_DIR:FaceRecognitionCPP>)
    
    add_custom_command(TARGET FaceRecognitionCPP POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${TORCH_INSTALL_PREFIX}/lib/torch.dll"
            $<TARGET_FILE_DIR:FaceRecognitionCPP>)
endif()

target_include_directories(FaceRecognitionCPP PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/Project/include"
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}  
)

target_link_libraries(FaceRecognitionCPP PRIVATE 
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}          
)

if(MSVC)
    target_link_directories(FaceRecognitionCPP PRIVATE
        "${OpenCV_DIR}/x64/vc16/bin"
        "${TORCH_INSTALL_PREFIX}/lib"
    )
    
    set_target_properties(FaceRecognitionCPP PROPERTIES 
        VS_DEBUGGER_ENVIRONMENT "PATH=${OpenCV_DIR}/x64/vc16/bin;${TORCH_INSTALL_PREFIX}/lib;$ENV{PATH}"
    )
    
    set(ENV{PATH} "${OpenCV_DIR}/x64/vc16/bin;${TORCH_INSTALL_PREFIX}/lib;$ENV{PATH}")
    
    source_group("External Libraries" FILES 
        "${OpenCV_DIR}/x64/vc16/bin/opencv_world4120.dll"
        "${TORCH_INSTALL_PREFIX}/lib/torch.dll"
    )
endif()

target_compile_features(FaceRecognitionCPP PRIVATE cxx_std_17)
target_compile_options(FaceRecognitionCPP PRIVATE /std:c++17)

message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "C++ Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")       
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "TORCH_INSTALL_PREFIX: ${TORCH_INSTALL_PREFIX}")