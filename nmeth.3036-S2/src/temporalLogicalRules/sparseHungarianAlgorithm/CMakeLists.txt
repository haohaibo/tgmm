#CMake file to build Tracking_GMM project

# refer to the root source directory of the project as ${TGMM_SOURCE_DIR} and
# to the root binary directory of the project as ${TGMM_BINARY_DIR}. 
#The FindCUDA script is distributed since version 2.8

cmake_minimum_required (VERSION 2.8)
project (SparseHungarianAlgorithm_project)



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



#
# submodules: add source and header files from subfolders
#
file(GLOB SRC_HDRS *.h)
file(GLOB SRC_C *.c)
file(GLOB SRC_CPP *.cpp)



#Add external code (Hungarian algorithm
file(GLOB_RECURSE ExternalC_SRCS external/*.c)
file(GLOB_RECURSE ExternalCXX_SRCS external/*.cpp)
file(GLOB_RECURSE External_HDRS external/*.h)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/external)

#create a library so we can add this code to other larger CMAKE projects
IF(NOT TARGET SparseHungarianAlgorithm)

add_library(SparseHungarianAlgorithm ${SRC_CPP}  ${SRC_C} ${SRC_HDRS} ${ExternalC_SRCS} ${ExternalCXX_SRCS} ${ExternalC_HDRS})
set_property(TARGET SparseHungarianAlgorithm PROPERTY COMPILE_DEFINITIONS IL_STD)

ENDIF()


#organize IDE
source_group("Header Files"  FILES ${SRC_HDRS})
source_group("Source Files"  FILES ${SRC_CPP})
source_group("Source Files"  FILES ${SRC_C})



