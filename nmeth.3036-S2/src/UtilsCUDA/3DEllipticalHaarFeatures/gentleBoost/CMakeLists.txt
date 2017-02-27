#CMake file to build Tracking_GMM project

#define important variables for the rest of the project
set(SETBYUSER_CUDA_ARCH sm_20 CACHE STRING "CUDA architecture") #for Quadro FX4800 sm13;fir Tesla sm_20

# refer to the root source directory of the project as ${TGMM_SOURCE_DIR} and
# to the root binary directory of the project as ${TGMM_BINARY_DIR}. 
#The FindCUDA script is distributed since version 2.8

cmake_minimum_required (VERSION 2.8)
project (GentleBoostProject)


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


#
#OpenMP
#
FIND_PACKAGE (OpenMP)

if (OPENMP_FOUND)
   #SET (CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS}")   #this will use the flags for all projects (check http://www.cmake.org/pipermail/cmake/2011-March/043534.html to see how to do it for a specific target)
   set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
   SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


file(GLOB GENTLE_BOOST_CPP_SRCS *.cpp)
file(GLOB GENTLE_BOOST_HDRS *.h)


IF(NOT TARGET GentleBoost)
	add_library (GentleBoost ${GENTLE_BOOST_CPP_SRCS} ${GENTLE_BOOST_HDRS}) 
ENDIF()

IF(NOT TARGET GentleBoost_MainTrain)
	add_executable (GentleBoost_MainTrain mainTrainSet.cxx ${GENTLE_BOOST_CPP_SRCS} ${GENTLE_BOOST_HDRS}) 
ENDIF()
