#CMake file to build Tracking_GMM project

# refer to the root source directory of the project as ${TGMM_SOURCE_DIR} and
# to the root binary directory of the project as ${TGMM_BINARY_DIR}. 
#The FindCUDA script is distributed since version 2.8

cmake_minimum_required (VERSION 2.8)
project (WatershedWithPersistanceAgglomeration_project)



#
#Set important flags
#

# Call  "cmake-gui -DCMAKE_BUILD_TYPE=Release .." on the command line to specify a 
#particular build, take their definition
# and dump it in the cache along with proper documentation, otherwise set CMAKE_BUILD_TYPE
# to Debug prior to calling PROJECT()
#
IF( CMAKE_BUILD_TYPE STREQUAL "")
	SET(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
ELSE() #Debug is the default
	SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF()

MESSAGE("CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")


MESSAGE("PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}")
MESSAGE("CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")

#to locate scripts
set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake") 

#
# Find necessary libraries
#


#for class supervoxel
include_directories ("${PROJECT_SOURCE_DIR}/../temporalLogicalRules")
add_subdirectory ("${PROJECT_SOURCE_DIR}/../temporalLogicalRules" "${CMAKE_CURRENT_BINARY_DIR}/tempLogRules")


#for parse config file
include_directories ("${PROJECT_SOURCE_DIR}/../Utils")



#for median filter
include_directories("${PROJECT_SOURCE_DIR}/CUDAmedianFilter2D")
add_subdirectory("${PROJECT_SOURCE_DIR}/CUDAmedianFilter2D")



#for JP2000 format using PicTools
IF( IS_DIRECTORY "${PROJECT_SOURCE_DIR}/../../PICTools" )
	include_directories("${PROJECT_SOURCE_DIR}/../../PICTools")
	add_subdirectory("${PROJECT_SOURCE_DIR}/../../PICTools" "${CMAKE_CURRENT_BINARY_DIR}/PICTools")
	INCLUDE_DIRECTORIES( ${PICTOOLS_INCLUDE_DIR} )
	LINK_DIRECTORIES(${PICTOOLS_LIBRARY_DIR})
	MESSAGE("Compiling code with PicTools for JP2 support. Include dirs ${PICTOOLS_INCLUDE_DIR}")
ELSE()
	MESSAGE("Compiling code WITHOUT PicTools for JP2 support. Folder ${PROJECT_SOURCE_DIR}/PICTools not found")
ENDIF()


#
# submodules: add source and header files from subfolders
#
file(GLOB SRC_HDRS *.h)
file(GLOB SRC_C *.c)

file(GLOB MEDIAN_SRC_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/external/medianFilter2D/*.h)
file(GLOB MEDIAN_SRC_C ${CMAKE_CURRENT_SOURCE_DIR}/external/medianFilter2D/*.c)


SET(SRC_CPP watershedPersistanceAgglomeration.cpp hierarchicalSegmentation.cpp)



#compile executable
IF( NOT TARGET WatershedWithPersistanceAgglomeration_Test)
add_executable(WatershedWithPersistanceAgglomeration_Test mainTest.cpp  ${SRC_CPP} ${SRC_HDRS} ${SRC_C} ${MEDIAN_SRC_HDRS} ${MEDIAN_SRC_C})
TARGET_LINK_LIBRARIES(WatershedWithPersistanceAgglomeration_Test TemporalLogicalRules)
TARGET_LINK_LIBRARIES(WatershedWithPersistanceAgglomeration_Test MedianFilter2D_CUDAlib)
ENDIF()


#compile executable
IF( NOT TARGET ProcessStack)
add_executable(ProcessStack mainProcessStack.cpp  ${SRC_CPP} ${SRC_HDRS} ${SRC_C} ${MEDIAN_SRC_HDRS} ${MEDIAN_SRC_C} "${PROJECT_SOURCE_DIR}/../Utils/parseConfigFile.cpp")
TARGET_LINK_LIBRARIES(ProcessStack TemporalLogicalRules)
TARGET_LINK_LIBRARIES(ProcessStack MedianFilter2D_CUDAlib)

 IF( IS_DIRECTORY "${PROJECT_SOURCE_DIR}/../../PICTools" )
	TARGET_LINK_LIBRARIES(ProcessStack PicToolsJ2K)
	add_definitions(-DPICTOOLS_JP2K)

 ENDIF()

ENDIF()

#compile small main program to excute ProcessStack in Batch mode multicore for Windows
#TODO: generate a similar script for Unix 
IF(WIN32)
IF( NOT TARGET ProcessStackBatchMulticore)
add_executable(ProcessStackBatchMulticore  mainProcessStackBatchMulticore.cxx "${PROJECT_SOURCE_DIR}/../Utils/parseConfigFile.cpp")
ENDIF()
ENDIF()


#compile library
IF( NOT TARGET HierarchicalSegmentationPBC)
add_library(HierarchicalSegmentationPBC ${SRC_CPP} ${SRC_HDRS} ${SRC_C} ${MEDIAN_SRC_HDRS} ${MEDIAN_SRC_C})
TARGET_LINK_LIBRARIES(HierarchicalSegmentationPBC TemporalLogicalRules)
TARGET_LINK_LIBRARIES(HierarchicalSegmentationPBC MedianFilter2D_CUDAlib)

IF( IS_DIRECTORY "${PROJECT_SOURCE_DIR}/../../PICTools" )
	TARGET_LINK_LIBRARIES(ProcessStack PicToolsJ2K)
	add_definitions(-DPICTOOLS_JP2K)
ENDIF()

ENDIF()


IF(WIN32)
set_target_properties(ProcessStack PROPERTIES COMPILE_FLAGS "-W4")
set_target_properties(WatershedWithPersistanceAgglomeration_Test PROPERTIES COMPILE_FLAGS "-W4")
set_target_properties(HierarchicalSegmentationPBC PROPERTIES COMPILE_FLAGS "-W4")
ELSE()
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x") #to allow C+11 features in older GCC compilers
ENDIF()

#organize IDE
source_group("Header Files"  FILES ${SRC_HDRS})
source_group("Header Files\\external\\medianFilter2D"  FILES ${MEDIAN_SRC_HDRS})
source_group("Source Files"  FILES mainTest.cpp mainProcessStack.cpp)
source_group("Source Files"  FILES ${SRC_CPP})
source_group("Source Files"  FILES ${SRC_C})
source_group("Source Files\\external\\medianFilter2D"  FILES ${MEDIAN_SRC_C})

