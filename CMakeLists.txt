cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(FaceRecognitionCPP)
add_definitions(-DNOMINMAX)

add_definitions(-DWIN32_LEAN_AND_MEAN)
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

set(IMGUI_DIR "C:/libs/imgui/imgui")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB_RECURSE PROJECT_SOURCE_FILES    "${CMAKE_CURRENT_SOURCE_DIR}/Trainer/src/*.cpp")
file(GLOB_RECURSE PROJECTAPP_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/appCPP/src/*.cpp")

add_executable(FaceRecognitionCPP
    "${CMAKE_CURRENT_SOURCE_DIR}/Trainer/main.cpp"
    ${PROJECT_SOURCE_FILES}
)
 
target_include_directories(FaceRecognitionCPP PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/Trainer/include"
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(FaceRecognitionCPP PRIVATE
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS} 
)

target_compile_features(FaceRecognitionCPP PRIVATE cxx_std_17)

add_executable(ExportModel
    "${CMAKE_CURRENT_SOURCE_DIR}/utils/ModelExporter/export_model.cpp"
    ${PROJECT_SOURCE_FILES}
)

target_include_directories(ExportModel PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/Trainer/include"
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
 
target_link_libraries(ExportModel PRIVATE
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

target_compile_features(ExportModel PRIVATE cxx_std_17)

add_executable(FaceRecognitionAPP WIN32 
    "${CMAKE_CURRENT_SOURCE_DIR}/appCPP/main.cpp"
    ${IMGUI_DIR}/imgui.cpp  
    ${IMGUI_DIR}/imgui_draw.cpp   
    ${IMGUI_DIR}/imgui_widgets.cpp 
    ${IMGUI_DIR}/imgui_tables.cpp 
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/backends/imgui_impl_win32.cpp
    ${IMGUI_DIR}/backends/imgui_impl_dx11.cpp
    ${PROJECTAPP_SOURCE_FILES}   
    ${PROJECT_SOURCE_FILES}
)

target_include_directories(FaceRecognitionAPP PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/appCPP/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/Trainer/include"
    ${TORCH_INCLUDE_DIRS}       
    ${OpenCV_INCLUDE_DIRS}       
    ${IMGUI_DIR} 
    ${IMGUI_DIR}/backends
)

target_link_libraries(FaceRecognitionAPP PRIVATE
    ${TORCH_LIBRARIES}           
    ${OpenCV_LIBS}               
    d3d11
    d3dcompiler
    dxgi
    dwmapi
)

target_compile_features(FaceRecognitionAPP PRIVATE cxx_std_17)

if(MSVC)
    foreach(target FaceRecognitionCPP FaceRecognitionAPP ExportModel)
        target_link_directories(${target} PRIVATE
            "${OpenCV_DIR}/x64/vc16/bin"
            "${TORCH_INSTALL_PREFIX}/lib"
        )
        set_target_properties(${target} PROPERTIES
            VS_DEBUGGER_ENVIRONMENT "PATH=${OpenCV_DIR}/x64/vc16/bin;${TORCH_INSTALL_PREFIX}/lib;$ENV{PATH}"
        )
    endforeach() 
endif()

message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
message(STATUS "TORCH_INSTALL_PREFIX: ${TORCH_INSTALL_PREFIX}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")