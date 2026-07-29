cmake_minimum_required(VERSION 3.19)
project(CUDA_FP32_GEMM_KERNEL LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)

# ensure cuda is available
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CUDA_COMPUTE_CAPABILITY 120)

# Kernel 10 compile-time autotuning parameters.
option(SGEMM_K10_AUTOTUNE_MODE
       "Run the single SGEMM_K10_* candidate instead of phase-four dispatch"
       OFF)
set(SGEMM_K10_NUM_THREADS 128 CACHE STRING "Kernel 10 thread count")
set(SGEMM_K10_BN 128 CACHE STRING "Kernel 10 block tile N")
set(SGEMM_K10_BM 128 CACHE STRING "Kernel 10 block tile M")
set(SGEMM_K10_BK 16 CACHE STRING "Kernel 10 block tile K")
set(SGEMM_K10_WN 64 CACHE STRING "Kernel 10 warp tile N")
set(SGEMM_K10_WM 64 CACHE STRING "Kernel 10 warp tile M")
set(SGEMM_K10_WNITER 4 CACHE STRING "Kernel 10 warp N iterations")
set(SGEMM_K10_TN 4 CACHE STRING "Kernel 10 thread tile N")
set(SGEMM_K10_TM 8 CACHE STRING "Kernel 10 thread tile M")

# Kernel 11 compile-time parameters for controlled double-buffer comparison.
option(SGEMM_K11_DIAGNOSTIC_SEQUENTIAL
       "Run K11 with all threads loading and computing sequentially"
       OFF)
option(SGEMM_K11_SPLIT_THREE_TO_ONE
       "Reproduce the rejected K11 3:1 compute/load split experiment"
       OFF)
option(SGEMM_K12_PACED_ASYNC
       "Reproduce the rejected K12 paced async-copy experiment"
       OFF)
set(SGEMM_K11_NUM_THREADS 256 CACHE STRING "Kernel 11 thread count")
set(SGEMM_K11_BN 256 CACHE STRING "Kernel 11 block tile N")
set(SGEMM_K11_BM 128 CACHE STRING "Kernel 11 block tile M")
set(SGEMM_K11_BK 16 CACHE STRING "Kernel 11 block tile K")
set(SGEMM_K11_WN 32 CACHE STRING "Kernel 11 warp tile N")
set(SGEMM_K11_WM 128 CACHE STRING "Kernel 11 warp tile M")
set(SGEMM_K11_WNITER 1 CACHE STRING "Kernel 11 warp N iterations")
set(SGEMM_K11_TN 8 CACHE STRING "Kernel 11 thread tile N")
set(SGEMM_K11_TM 8 CACHE STRING "Kernel 11 thread tile M")

# in debug mode, add debug symbols to device code
# this disables most optimizations and kills performance
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
# add_compile_options("--ptxas-options=-v")

# Configure header file search paths
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
# Configure the source file path to be compiled
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

# generate executable
add_executable(sgemm sgemm.cu ${SRC})
set_target_properties(sgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_compile_definitions(
  sgemm PRIVATE
  SGEMM_K10_NUM_THREADS=${SGEMM_K10_NUM_THREADS}
  SGEMM_K10_BN=${SGEMM_K10_BN}
  SGEMM_K10_BM=${SGEMM_K10_BM}
  SGEMM_K10_BK=${SGEMM_K10_BK}
  SGEMM_K10_WN=${SGEMM_K10_WN}
  SGEMM_K10_WM=${SGEMM_K10_WM}
  SGEMM_K10_WNITER=${SGEMM_K10_WNITER}
  SGEMM_K10_TN=${SGEMM_K10_TN}
  SGEMM_K10_TM=${SGEMM_K10_TM}
  SGEMM_K11_NUM_THREADS=${SGEMM_K11_NUM_THREADS}
  SGEMM_K11_BN=${SGEMM_K11_BN}
  SGEMM_K11_BM=${SGEMM_K11_BM}
  SGEMM_K11_BK=${SGEMM_K11_BK}
  SGEMM_K11_WN=${SGEMM_K11_WN}
  SGEMM_K11_WM=${SGEMM_K11_WM}
  SGEMM_K11_WNITER=${SGEMM_K11_WNITER}
  SGEMM_K11_TN=${SGEMM_K11_TN}
  SGEMM_K11_TM=${SGEMM_K11_TM}
)
if(SGEMM_K10_AUTOTUNE_MODE)
  target_compile_definitions(sgemm PRIVATE SGEMM_K10_AUTOTUNE_MODE=1)
endif()
if(SGEMM_K11_DIAGNOSTIC_SEQUENTIAL)
  target_compile_definitions(sgemm PRIVATE SGEMM_K11_DIAGNOSTIC_SEQUENTIAL=1)
endif()
if(SGEMM_K11_SPLIT_THREE_TO_ONE)
  target_compile_definitions(sgemm PRIVATE SGEMM_K11_SPLIT_THREE_TO_ONE=1)
endif()
if(SGEMM_K12_PACED_ASYNC)
  target_compile_definitions(sgemm PRIVATE SGEMM_K12_PACED_ASYNC=1)
endif()
target_link_libraries(sgemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

add_executable(cuBLAS_sgemm cuBLAS_sgemm.cu )
set_target_properties(cuBLAS_sgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(cuBLAS_sgemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

add_executable(simplest_kernel simplest_kernel.cu)
set_target_properties(simplest_kernel PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(simplest_kernel ${CUDA_LIBRARIES})
