#CMake file to build Tracking_GMM project


set(SETBYUSER_CUDA_ARCH sm_20 CACHE STRING "CUDA architecture") #for Quadro FX4800 sm13;fir Tesla sm_20

# refer to the root source directory of the project as ${TGMM_SOURCE_DIR} and
# to the root binary directory of the project as ${TGMM_BINARY_DIR}. 
#The FindCUDA script is distributed since version 2.8

cmake_minimum_required (VERSION 2.8)
project (TemporalLogicalRules_project)



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
#mylib (it will comiple it on-the-fly)

if(NOT DEFINED MYLIB_ROOT)

#I need to set it in this scope and in th eparent scope
set(MYLIB_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/mylib)
add_subdirectory(${MYLIB_ROOT})
set(MYLIB_INCLUDE_DIR ${MYLIB_ROOT})
SET(MYLIB_LIBRARIES_DIR ${CMAKE_CURRENT_BINARY_DIR}/mylib ${CMAKE_CURRENT_BINARY_DIR}/mylib/MY_TIFF)
SET(MYLIB_LIBRARIES mylib mytiff)


set(MYLIB_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/mylib PARENT_SCOPE)
set(MYLIB_INCLUDE_DIR ${MYLIB_ROOT} PARENT_SCOPE)
SET(MYLIB_LIBRARIES_DIR ${CMAKE_CURRENT_BINARY_DIR}/mylib ${CMAKE_CURRENT_BINARY_DIR}/mylib/MY_TIFF PARENT_SCOPE)
SET(MYLIB_LIBRARIES mylib mytiff PARENT_SCOPE)

ENDIF()


#it automatically adds for each configuration
include_directories(${MYLIB_INCLUDE_DIR})
LINK_DIRECTORIES(${MYLIB_LIBRARIES_DIR})
LINK_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})



#
#Cuda
#
SET(CUDA_NVCC_FLAGS_DEBUG -DTHRUST_DEBUG;-g;-G;-O0;-arch=${SETBYUSER_CUDA_ARCH} CACHE STRING "Semi-colon delimit multiple arguments") #set before FIND_PACKAGE(CUDA) in order to avoid FORCE to show them in GUI. So user can modify them
SET(CUDA_NVCC_FLAGS_RELEASE -O3;-arch=${SETBYUSER_CUDA_ARCH} CACHE STRING "Semi-colon delimit multiple arguments")
SET(CUDA_NVCC_FLAGS_RELWITHDEBINFO -O3;-g;-arch=${SETBYUSER_CUDA_ARCH} CACHE STRING "Semi-colon delimit multiple arguments")
FIND_PACKAGE(CUDA REQUIRED)
SET(CUDA_VERBOSE_BUILD ON FORCE)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS}) 
#lcudart and lcuda  are already added as variables with find package
#SET(LIBS_CUDA_CUSPARSE "cusparse" CACHE STRING "")#in case we need to add CUDA libraries ignored in CUDA.cmake 


IF(WIN32)

IF (${CMAKE_SIZEOF_VOID_P} MATCHES 4) #win32 machine
	link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32") # needed to add cusparse
ELSE (${CMAKE_SIZEOF_VOID_P} MATCHES 4) #win64 machine
	link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib/x64") # needed to add cusparse
ENDIF (${CMAKE_SIZEOF_VOID_P} MATCHES 4)

ELSEIF()
	link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib") # needed to add cusparse
ENDIF()


#Sparse Hungarian algorithm (our own library)
include_directories ("${PROJECT_SOURCE_DIR}/sparseHungarianAlgorithm")
add_subdirectory (sparseHungarianAlgorithm)



#
# submodules: add source and header files from subfolders
#
file(GLOB SRC_HDRS *.h)

file(GLOB XML_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/external/xmlParser/*.h)
file(GLOB XML_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/external/xmlParser/*.cpp)
file(GLOB External_Mylib_HDRS ${MYLIB_INCLUDE_DIR}/*.h)

file(GLOB UtilsCUDA_SRCS knnCUDA/*.cpp)
file(GLOB UtilsCUDA_HDRS knnCUDA/*.h)
file(GLOB UtilsCUDA_CUDA_SRCS knnCUDA/*.cu)

SET(SRC_CPP temporalLogicalRules.cpp GaussianMixtureModel_Redux.cpp nuclei.cpp supervoxel.cpp lineage.cpp lineageHyperTree.cpp trackletCalculation.cpp lineageWindowFeatures.cpp localGeometricDescriptor.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../constants.cpp)



#TODO SET THE FLAGS FOR DEBUG VERSION
CUDA_COMPILE(UtilsCUDA_CUDA_OBJ STATIC ${UtilsCUDA_CUDA_SRCS})


#create a library so we can add this code to other larger CMAKE projects
IF(NOT TARGET TemporalLogicalRules) #it only access here if this particular target has not been created yet

add_library(TemporalLogicalRules ${SRC_CPP} ${SRC_HDRS} ${XML_SRCS} ${XML_HDRS} ${UtilsCUDA_SRCS} ${UtilsCUDA_HDRS} ${UtilsCUDA_CUDA_SRCS} ${UtilsExe_SRCS} ${UtilsCUDA_CUDA_OBJ})
TARGET_LINK_LIBRARIES(TemporalLogicalRules ${MYLIB_LIBRARIES})
TARGET_LINK_LIBRARIES(TemporalLogicalRules SparseHungarianAlgorithm)
target_link_libraries(TemporalLogicalRules  ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})

ENDIF()

#organize IDE
source_group("Header Files"  FILES ${SRC_HDRS})
source_group("Header Files\\knnCUDA"  FILES ${UtilsCUDA_HDRS})
source_group("Header Files\\external\\xmlParser"  FILES ${XML_HDRS})
source_group("Source Files\\external\\xmlParser"  FILES ${XML_SRCS})
source_group("Source Files\\knnCUDA"  FILES ${UtilsCUDA_SRCS})
source_group("Source Files\\knnCUDA"  FILES ${UtilsCUDA_CUDA_SRCS})
source_group("Source Files"  FILES ${SRC_CPP})
source_group("Header Files\\mylib" FILES ${External_Mylib_HDRS})

