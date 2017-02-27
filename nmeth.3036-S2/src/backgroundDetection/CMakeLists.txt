#CMake file to build PostGRESQL library based on libsq
#Code from http://www.codeproject.com/Articles/35138/Accessing-PostgreSql-data-base-using-libpq-C-Libra

# refer to the root source directory of the project as ${TGMM_SOURCE_DIR} and
# to the root binary directory of the project as ${TGMM_BINARY_DIR}. 
#The FindCUDA script is distributed since version 2.8

cmake_minimum_required (VERSION 2.8)
project (BackgroundDetectionLib_Project)



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

#temporal logical rules for supervoxel and lineage classes
include_directories ("${PROJECT_SOURCE_DIR}/../temporalLogicalRules")
add_subdirectory ("${PROJECT_SOURCE_DIR}/../temporalLogicalRules" "${CMAKE_CURRENT_BINARY_DIR}/tempLogRules")

#ellipsoid annotations
include_directories("${PROJECT_SOURCE_DIR}/../UtilsCUDA/3DEllipticalHaarFeatures")


#
# submodules: add source and header files from subfolders
#
file(GLOB SRC_HDRS *.h)
file(GLOB SRC_CPP *.cpp)


#create a library so we can add this code to other larger CMAKE projects
add_library(BackgroundDetectionLib ${SRC_CPP} ${SRC_HDRS} )
TARGET_LINK_LIBRARIES(BackgroundDetectionLib TemporalLogicalRules)
TARGET_LINK_LIBRARIES(BackgroundDetectionLib ellipticalHaarFeatures)
