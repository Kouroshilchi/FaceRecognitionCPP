cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(FaceRecognitionCPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/std:c++17)
endif()

set(CMAKE_PREFIX_PATH "C:/libs/release/libtorch")
set(Torch_DIR "C:/libs/release/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

set(OpenCV_DIR "C:/libs/opencv/build")
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB_RECURSE PROJECT_SOURCE_FILES    "${CMAKE_CURRENT_SOURCE_DIR}/Project/src/*.cpp")
file(GLOB_RECURSE PROJECTAPP_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/appCPP/src/*.cpp")

add_executable(FaceRecognitionCPP
    "${CMAKE_CURRENT_SOURCE_DIR}/Project/main.cpp"
    ${PROJECT_SOURCE_FILES}
)
target_include_directories(FaceRecognitionCPP PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/Project/include"
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(FaceRecognitionCPP PRIVATE
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)
target_compile_features(FaceRecognitionCPP PRIVATE cxx_std_17)

if(MSVC)
    target_link_directories(FaceRecognitionCPP PRIVATE
        "${OpenCV_DIR}/x64/vc16/bin"
        "${TORCH_INSTALL_PREFIX}/lib"
    )
    set_target_properties(FaceRecognitionCPP PROPERTIES
        VS_DEBUGGER_ENVIRONMENT "PATH=${OpenCV_DIR}/x64/vc16/bin;${TORCH_INSTALL_PREFIX}/lib;$ENV{PATH}"
    )
    add_custom_command(TARGET FaceRecognitionCPP POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OpenCV_DIR}/x64/vc16/bin/opencv_world4120.dll"
            $<TARGET_FILE_DIR:FaceRecognitionCPP>)
    add_custom_command(TARGET FaceRecognitionCPP POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${TORCH_INSTALL_PREFIX}/lib/torch.dll"
            $<TARGET_FILE_DIR:FaceRecognitionCPP>)
endif()

add_executable(FaceRecognitionAPP
    "${CMAKE_CURRENT_SOURCE_DIR}/appCPP/main.cpp"
    ${PROJECTAPP_SOURCE_FILES}
)
target_include_directories(FaceRecognitionAPP PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/appCPP/include"
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(FaceRecognitionAPP PRIVATE
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)
target_compile_features(FaceRecognitionAPP PRIVATE cxx_std_17)

if(MSVC)
    target_link_directories(FaceRecognitionAPP PRIVATE
        "${OpenCV_DIR}/x64/vc16/bin"
        "${TORCH_INSTALL_PREFIX}/lib"
    )
    set_target_properties(FaceRecognitionAPP PROPERTIES
        VS_DEBUGGER_ENVIRONMENT "PATH=${OpenCV_DIR}/x64/vc16/bin;${TORCH_INSTALL_PREFIX}/lib;$ENV{PATH}"
    )
    add_custom_command(TARGET FaceRecognitionAPP POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OpenCV_DIR}/x64/vc16/bin/opencv_world4120.dll"
            $<TARGET_FILE_DIR:FaceRecognitionAPP>)
    add_custom_command(TARGET FaceRecognitionAPP POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${TORCH_INSTALL_PREFIX}/lib/torch.dll"
            $<TARGET_FILE_DIR:FaceRecognitionAPP>)
endif()

message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
message(STATUS "TORCH_INSTALL_PREFIX: ${TORCH_INSTALL_PREFIX}")