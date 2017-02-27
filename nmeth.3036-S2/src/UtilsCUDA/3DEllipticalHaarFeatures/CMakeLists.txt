#CMake file to build Tracking_GMM project

#define important variables for the rest of the project
set(SETBYUSER_CUDA_ARCH sm_20 CACHE STRING "CUDA architecture") #for Quadro FX4800 sm13;fir Tesla sm_20

# refer to the root source directory of the project as ${TGMM_SOURCE_DIR} and
# to the root binary directory of the project as ${TGMM_BINARY_DIR}. 
#The FindCUDA script is distributed since version 2.8

cmake_minimum_required (VERSION 2.8)
project (EllipticalHaarFeatures_project)



#
#Set important flags
#

# Call  "cmake-gui -DCMAKE_BUILD_TYPE=Release .." on the command line to specify a particular build, take their definition
# and dump it in the cache along with proper documentation, otherwise set CMAKE_BUILD_TYPE
# to Debug prior to calling PROJECT()
#
IF( CMAKE_BUILD_TYPE STREQUAL "")
	SET(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
ELSE() #Debug is the default
	SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF()

MESSAGE("CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

#to locate scripts
set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake") 

#
# Find necessary libraries
#
#Cuda
#Any device debugging symbols (-G) generate a kernel launch failure in thurst::sort_by_key
SET(CUDA_NVCC_FLAGS_DEBUG -DTHRUST_DEBUG;-G;-O0;-arch=${SETBYUSER_CUDA_ARCH} CACHE STRING "Semi-colon delimit multiple arguments") #set before FIND_PACKAGE(CUDA) in order to avoid FORCE to show them in GUI. So user can modify them
SET(CUDA_NVCC_FLAGS_RELEASE -O3;-arch=${SETBYUSER_CUDA_ARCH} CACHE STRING "Semi-colon delimit multiple arguments")
FIND_PACKAGE(CUDA REQUIRED)
SET(CUDA_VERBOSE_BUILD ON FORCE)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS}) 
#lcudart and lcuda  are already added as variables with find package
#SET(LIBS_CUDA_CUSPARSE "cusparse" CACHE STRING "")#add CUDA libraries ignored in CUDA.cmake



#mylib (it will comiple it on-the-fly)
if(NOT TARGET generator)

set(MYLIB_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/mylib)
add_subdirectory(${MYLIB_ROOT})
set(MYLIB_INCLUDE_DIR ${MYLIB_ROOT} CACHE STRING "")
SET(MYLIB_LIBRARIES_DIR ${CMAKE_CURRENT_BINARY_DIR}/mylib ${CMAKE_CURRENT_BINARY_DIR}/mylib/MY_TIFF)
SET(MYLIB_LIBRARIES mylib mytiff)

ENDIF()

MESSAGE("3D ELLIPTICASL HAAR MYLIB ${MYLIB_LIBRARIES_DIR}")

#it automatically adds for each configuration
include_directories(${MYLIB_INCLUDE_DIR})
LINK_DIRECTORIES(${MYLIB_LIBRARIES_DIR})
LINK_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})


#
# Add subdirectories
#
add_subdirectory (gentleBoost)
# Make sure the compiler can find include files from our Hello library. 
include_directories (${GENTLEBOOST_SOURCE_DIR}/gentleBoost)
# Make sure the linker can find the Hello library once it is built.
link_directories (${GENTLEBOOST_BINARY_DIR}/gentleBoost)  

#
#OpenMP
#
FIND_PACKAGE (OpenMP)

if (OPENMP_FOUND)
   #SET (CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS}")   #this will use the flags for all projects (check http://www.cmake.org/pipermail/cmake/2011-March/043534.html to see how to do it for a specific target)
   set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
   SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


#
# submodules: add source and header files from subfolders
#
file(GLOB CUDA_HDRS *.h)
file(GLOB CUDA_CUDA_SRCS *.cu)

file(GLOB TICTOC_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/external/Nathan/*.h)
file(GLOB TICTOC_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/external/Nathan/*.c)
file(GLOB XML_PARSER_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/external/xmlParser/*.h)
file(GLOB XML_PARSER_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/external/xmlParser/*.cpp)
file(GLOB MY_TIFF_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/external/MY_TIFF/*.h)
file(GLOB MY_TIFF_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/external/MY_TIFF/*.c)
file(GLOB External_Mylib_HDRS ${MYLIB_INCLUDE_DIR}/*.h)

#TODO SET THE FLAGS FOR DEBUG VERSION
CUDA_COMPILE(CUDA_CUDA_OBJ STATIC ${CUDA_CUDA_SRCS})


#compile executable to train a classifier for cell division 
IF(NOT TARGET trainCellDivision)
add_executable(trainCellDivision mainCellDivisionClassifier.cpp EllipticalHaarFeatures.cpp AnnotationEllipsoid.cpp ${CUDA_SRCS} ${CUDA_CUDA_OBJ} ${CUDA_HDRS} ${CUDA_CUDA_SRCS} ${TICTOC_HDRS} ${TICTOC_SRCS} ${XML_PARSER_HDRS} ${XML_PARSER_SRCS} ${MY_TIFF_HDRS} ${MY_TIFF_SRCS})
target_link_libraries(trainCellDivision ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
TARGET_LINK_LIBRARIES(trainCellDivision ${MYLIB_LIBRARIES})

# Link the executable to the Gentle boost library.
target_link_libraries (trainCellDivision GentleBoost) 

ENDIF()


#create library so I can use this functionality from a different code
IF( NOT TARGET ellipticalHaarFeatures)
add_library (ellipticalHaarFeatures EllipticalHaarFeatures.cpp AnnotationEllipsoid.cpp ${CUDA_SRCS} ${CUDA_CUDA_OBJ} ${CUDA_HDRS} ${CUDA_CUDA_SRCS} ${TICTOC_HDRS} ${TICTOC_SRCS} ${XML_PARSER_HDRS} ${XML_PARSER_SRCS})
target_link_libraries(ellipticalHaarFeatures ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
ENDIF()

source_group("Header Files"  FILES ${CUDA_CUDA_HDRS})
source_group("Header Files\\External\\Nathan"  FILES TICTOC_HDRS)
source_group("Header Files\\External\\xmlParser"  FILES XML_PARSER_HDRS)
source_group("Header Files\\External\\MY_TIFF"  FILES MY_TIFF_HDRS)
source_group("Header Files\\mylib" FILES ${External_Mylib_HDRS})

source_group("Source Files" FILES mainTest.cpp EllipticalHaarFeatures.cpp mainCellDivisionClassifier.cpp AnnotationEllipsoid.cpp ${CUDA_CUDA_SRCS})
source_group("Source Files\\External\\Nathan"  FILES TICTOC_SRCS)
source_group("Source Files\\External\\xmlParser"  FILES XML_PARSER_SRCS)
source_group("Source Files\\External\\MY_TIFF"  FILES MY_TIFF_SRCS)
