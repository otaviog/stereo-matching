include_directories(${PYTHON_INCLUDE_DIR})

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")

cuda_add_library(stereomatch
  cuda_utils.cpp
  # Costs
  cost.cpp
  ssd.cu  
  # Aggregations
  aggregation.cpp
  winners_take_all.cu
  dynamic_programming.cu
)


target_include_directories(stereomatch
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/stereomatch>
  $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIR}>
  )

target_link_libraries(stereomatch "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}" )

set_property(TARGET stereomatch PROPERTY CXX_STANDARD 17)

set_target_properties(stereomatch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(_cstereomatch SHARED _cstereomatch.cpp)
set_property(TARGET _cstereomatch PROPERTY CXX_STANDARD 17)
set_target_properties(_cstereomatch PROPERTIES PREFIX "")

if (SKBUILD)
  python_extension_module(_cstereomatch)
  install(TARGETS _cstereomatch LIBRARY DESTINATION stereomatch)
else (SKBUILD)
  set_target_properties(_cstereomatch PROPERTIES LIBRARY_OUTPUT_DIRECTORY
	${CMAKE_SOURCE_DIR}/stereomatch)
endif (SKBUILD)

target_compile_options(_cstereomatch PRIVATE -Wall -Wextra -pedantic)
target_link_libraries(_cstereomatch stereomatch)
