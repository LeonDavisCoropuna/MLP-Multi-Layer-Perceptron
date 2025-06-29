cmake_minimum_required(VERSION 3.10)
project(mlp LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Estándar de C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Establecer build type si no se especifica
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
  message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
endif()

# Ruta de OpenCV (ajústala si es necesario)
set(OpenCV_DIR "/home/leon/Documentos/UNSA/TOPICOS IA/opencv/build")
find_package(OpenCV REQUIRED)

# Incluye tus directorios
include_directories(
  ${CMAKE_SOURCE_DIR}/models
  ${CMAKE_SOURCE_DIR}/utils
  ${OpenCV_INCLUDE_DIRS}
)

# Archivos fuente
add_executable(main 
  main.cpp  
  models/layers
  utils/tensor.hpp
  models/layers/layer.hpp
  models/layers/dense_layer.hpp 
  models/model.hpp
  utils/activations.hpp 
  utils/load_dataset.hpp
  utils/optimizer.hpp 
)

add_executable(test 
  test.cpp  
  utils/tensor.hpp
  models/model.hpp
  utils/activations.hpp 
)

# Flags para Release y Debug
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(main PRIVATE -O3 -march=native -DNDEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(main PRIVATE -O0 -g)
endif()

# OpenMP flags
if(OpenMP_FOUND)
  target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(test PRIVATE ${OpenCV_LIBS})

# Enlace de OpenCV
target_link_libraries(main PRIVATE ${OpenCV_LIBS})

# Warnings útiles
target_compile_options(main PRIVATE)
