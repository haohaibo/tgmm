#CMake file to build Tracking_GMM project

#define important variables for the rest of the project
set(SETBYUSER_CUDA_ARCH sm_20 CACHE STRING "CUDA architecture") #for Quadro FX4800 sm13;fir Tesla sm_20

# refer to the root source directory of the project as ${TGMM_SOURCE_DIR} and
# to the root binary directory of the project as ${TGMM_BINARY_DIR}. 
#The FindCUDA script is distributed since version 2.8

cmake_minimum_required (VERSION 2.8)
project (Tracking_GMM_project)


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
set(MYLIB_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/mylib)
add_subdirectory(${MYLIB_ROOT})
set(MYLIB_INCLUDE_DIR ${MYLIB_ROOT})
include_directories(${MYLIB_INCLUDE_DIR})

#it automatically adds for each configuration
SET(MYLIB_LIBRARIES_DIR ${CMAKE_CURRENT_BINARY_DIR}/mylib ${CMAKE_CURRENT_BINARY_DIR}/mylib/MY_TIFF)
SET(MYLIB_LIBRARIES mylib mytiff)
LINK_DIRECTORIES(${MYLIB_LIBRARIES_DIR})
LINK_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})


#Cuda
#Any device debugging symbols (-G) generate a kernel launch failure in thurst::sort_by_key
SET(CUDA_NVCC_FLAGS_DEBUG -DTHRUST_DEBUG;-g;-O0;-arch=${SETBYUSER_CUDA_ARCH} CACHE STRING "Semi-colon delimit multiple arguments") #set before FIND_PACKAGE(CUDA) in order to avoid FORCE to show them in GUI. So user can modify them
SET(CUDA_NVCC_FLAGS_RELEASE -O3;-arch=${SETBYUSER_CUDA_ARCH} CACHE STRING "Semi-colon delimit multiple arguments")
SET(CUDA_NVCC_FLAGS_RELWITHDEBINFO -O3;-g;-arch=${SETBYUSER_CUDA_ARCH} CACHE STRING "Semi-colon delimit multiple arguments")
FIND_PACKAGE(CUDA REQUIRED)
SET(CUDA_VERBOSE_BUILD ON FORCE)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS}) 
#lcudart and lcuda  are already added as variables with find package
SET(LIBS_CUDA_CUSPARSE "cusparse" CACHE STRING "")#add CUDA libraries ignored in CUDA.cmake




#Sparse Hungarian algorithm (our own library with external BSD code)
include_directories ("${PROJECT_SOURCE_DIR}/temporalLogicalRules")
add_subdirectory (temporalLogicalRules)
LINK_DIRECTORIES( "${CMAKE_CURRENT_BINARY_DIR}/temporalLogicalRules" )
LINK_DIRECTORIES( "${CMAKE_CURRENT_BINARY_DIR}/temporalLogicalRules/sparseHungarianAlgorithm" )



#include watershed + persistance based clustering + hierarchical segmentation
include_directories("${PROJECT_SOURCE_DIR}/nucleiChSvWshedPBC")
add_subdirectory("${PROJECT_SOURCE_DIR}/nucleiChSvWshedPBC")


IF(WIN32)

IF (${CMAKE_SIZEOF_VOID_P} MATCHES 4) #win32 machine
	link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32") # needed to add cusparse
ELSE (${CMAKE_SIZEOF_VOID_P} MATCHES 4) #win64 machine
	link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib/x64") # needed to add cusparse
ENDIF (${CMAKE_SIZEOF_VOID_P} MATCHES 4)

ELSE()
	IF (${CMAKE_SIZEOF_VOID_P} MATCHES 4) #32 machine
	link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib") # needed to add cusparse
	UNSET(LIBS_CUDA_CUSPARSE CACHE)
	SET(LIBS_CUDA_CUSPARSE "${CUDA_TOOLKIT_ROOT_DIR}/lib/libcusparse.so" CACHE STRING "")#add CUDA libraries ignored in CUDA.cmake
ELSE (${CMAKE_SIZEOF_VOID_P} MATCHES 4) #64-bit machine
	UNSET(LIBS_CUDA_CUSPARSE CACHE)
	SET(LIBS_CUDA_CUSPARSE "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcusparse.so" CACHE STRING "")#add CUDA libraries
ENDIF (${CMAKE_SIZEOF_VOID_P} MATCHES 4)

	
ENDIF()


#for JP2000 format using PicTools
IF( IS_DIRECTORY "${PROJECT_SOURCE_DIR}/../PICTools" )
	include_directories("${PROJECT_SOURCE_DIR}/../PICTools")
	add_subdirectory("${PROJECT_SOURCE_DIR}/../PICTools" "${CMAKE_CURRENT_BINARY_DIR}/PICTools")
	INCLUDE_DIRECTORIES( ${PICTOOLS_INCLUDE_DIR} )
	LINK_DIRECTORIES(${PICTOOLS_LIBRARY_DIR})
	MESSAGE("Compiling TGMM code with PicTools for JP2 support. Include dirs ${PICTOOLS_INCLUDE_DIR}")
ELSE()
	MESSAGE("Compiling TGMM code WITHOUT PicTools for JP2 support. Folder ${PROJECT_SOURCE_DIR}/PICTools not found")
ENDIF()


#background detection
include_directories("${PROJECT_SOURCE_DIR}/backgroundDetection")
add_subdirectory("${PROJECT_SOURCE_DIR}/backgroundDetection")
LINK_DIRECTORIES("${PROJECT_SOURCE_DIR}/backgroundDetection")

#3D elliptical Haar features detection
include_directories ("${PROJECT_SOURCE_DIR}/UtilsCUDA/3DEllipticalHaarFeatures")
add_subdirectory ("${PROJECT_SOURCE_DIR}/UtilsCUDA/3DEllipticalHaarFeatures")
include_directories (${ELLIPTICALHAARFEATURES_SOURCE_DIR}/ellipticalHaarFeatures)
link_directories (${ELLIPTICALHAARFEATURES_BINARY_DIR}/ellipticalHaarFeatures)  




include_directories("${PROJECT_SOURCE_DIR}/UtilsCUDA/3DEllipticalHaarFeatures")


#
# submodules: add source and header files from subfolders
#
file(GLOB UtilsExe_SRCS UtilsExe/*.cpp)
file(GLOB UtilsExe_HDRS UtilsExe/*.h)

file(GLOB Utils_SRCS Utils/*.cpp)
file(GLOB UtilsC_SRCS Utils/*.c)
file(GLOB Utils_HDRS Utils/*.h)


file(GLOB UtilsCUDA_SRCS UtilsCUDA/*.cpp)
file(GLOB UtilsCUDA_HDRS UtilsCUDA/*.h)
file(GLOB UtilsCUDA_CUDA_SRCS UtilsCUDA/*.cu)


file(GLOB_RECURSE ExternalC_SRCS external/*.c)
file(GLOB_RECURSE ExternalCXX_SRCS external/*.cpp)
file(GLOB_RECURSE External_HDRS external/*.h)


#TODO SET THE FLAGS FOR DEBUG VERSION
CUDA_COMPILE(UtilsCUDA_CUDA_OBJ STATIC ${UtilsCUDA_CUDA_SRCS} "${PROJECT_SOURCE_DIR}/constants.cpp")

#build main executables
set(MAIN_CLASSES_SRCS GaussianMixtureModel.cpp variationalInference.cpp kdtree.cpp responsibilities.cpp cellDivision.cpp supportFunctionsMain.cpp  backgroundDetectionInterface.cpp constants.cpp)
file(GLOB MAIN_CLASSES_HDRS *.h)

add_executable(TGMM mainTrackingGaussianMixtureModel.cpp ${MAIN_CLASSES_SRCS} ${MAIN_CLASSES_HDRS} ${UtilsCUDA_SRCS} ${UtilsCUDA_CUDA_OBJ} ${UtilsCUDA_HDRS} ${UtilsCUDA_CUDA_SRCS} ${UtilsExe_SRCS} ${UtilsExe_HDRS} ${Utils_SRCS} ${UtilsC_SRCS} ${Utils_HDRS} ${OpticalFlow_SRCS} ${OpticalFlow_HDRS} ${UtilsMatlabSharedLibrary_SRCS} ${UtilsMatlabSharedLibrary_HDRS} ${ExternalC_SRCS} ${ExternalCXX_SRCS} ${External_HDRS})

IF(WIN32)
set_target_properties(TGMM PROPERTIES COMPILE_FLAGS "-W4")
ELSE()
	set_target_properties(TGMM PROPERTIES COMPILE_FLAGS "-Wall")
	find_package(OpenMP)
	if (OPENMP_FOUND)
	    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	endif()
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x") #to allow C+11 features in older GCC compilers
ENDIF()

target_link_libraries(TGMM ${LIBS_CUDA_CUSPARSE} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
CUDA_ADD_CUBLAS_TO_TARGET(TGMM)


TARGET_LINK_LIBRARIES(TGMM TemporalLogicalRules)
TARGET_LINK_LIBRARIES( TGMM ${MYLIB_LIBRARIES})
TARGET_LINK_LIBRARIES( TGMM ellipticalHaarFeatures)
TARGET_LINK_LIBRARIES( TGMM BackgroundDetectionLib)
target_link_libraries( TGMM GentleBoost) 
target_link_libraries( TGMM HierarchicalSegmentationPBC)

IF( IS_DIRECTORY "${PROJECT_SOURCE_DIR}/../PICTools" )
	TARGET_LINK_LIBRARIES(TGMM PicToolsJ2K)
	add_definitions(-DPICTOOLS_JP2K)
ENDIF()

#copy calssifiers (txt files needed by the code)
add_custom_command(TARGET TGMM POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/filesToCopyToExeFolder $<TARGET_FILE_DIR:TGMM>)

#copy dll for CUDA and MSVC2010 C-compiler
file(GLOB ConfigFiles2 ${CMAKE_SOURCE_DIR}/../lib/*.dll)                   
foreach(ConfigFile2 ${ConfigFiles2})
	add_custom_command(TARGET TGMM POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${ConfigFile2} $<TARGET_FILE_DIR:TGMM>)
	add_custom_command(TARGET TGMM POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${ConfigFile2} $<TARGET_FILE_DIR:ProcessStack>)
endforeach()

#copy DLLs for PICTOOLS
IF( IS_DIRECTORY "${PROJECT_SOURCE_DIR}/../PICTools" )
	

	file(GLOB ConfigFiles ${CMAKE_SOURCE_DIR}/../PICTools/lib/picx*.dll)                   
	foreach(ConfigFile ${ConfigFiles})
		add_custom_command(TARGET TGMM POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${ConfigFile} $<TARGET_FILE_DIR:TGMM>)
		add_custom_command(TARGET TGMM POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${ConfigFile} $<TARGET_FILE_DIR:ProcessStack>)
	endforeach()
ENDIF()

#
# Source groups for IDE
#

#We need each external folder separately
file(GLOB External_gsl_SRCS external/gsl/*.c)
file(GLOB External_Nathan_SRCS external/Nathan/*.c)
file(GLOB External_xmlParser_SRCS external/xmlParser2/*.cpp)

file(GLOB External_Mylib_HDRS mylib/*.h)
file(GLOB_RECURSE External_Eigen_HDRS external/Eigen/*)

file(GLOB External_gsl_HDRS external/gsl/*.h)
file(GLOB External_Nathan_HDRS external/Nathan/*.h)
file(GLOB External_xmlParser_HDRS external/xmlParser2/*.h)


source_group("Header Files"  FILES ${MAIN_CLASSES_HDRS})
source_group("Header Files\\UtilsCUDA"  FILES ${UtilsCUDA_HDRS})
source_group("Header Files\\Utils"  FILES ${Utils_HDRS})
source_group("Header Files\\External\\gsl" FILES ${External_gsl_HDRS})
source_group("Header Files\\External\\Nathan" FILES ${External_Nathan_HDRS})
source_group("Header Files\\External\\xmlParser" FILES ${External_xmlParses_HDRS})
source_group("Header Files\\mylib" FILES ${External_Mylib_HDRS})
source_group("Header Files\\External\\Eigen" FILES ${External_Eigen_HDRS})

source_group("Source Files" FILES mainTrackingGaussianMixtureModel.cpp)
source_group("Source Files\\UtilsCUDA"  FILES ${UtilsCUDA_SRCS})
source_group("Source Files\\UtilsCUDA"  FILES ${UtilsCUDA_CUDA_SRCS})
source_group("Source Files\\Utils"  FILES ${Utils_SRCS})
source_group("Source Files\\Utils"  FILES ${UtilsC_SRCS})
source_group("Source Files\\External\\gsl" FILES ${External_gsl_SRCS})
source_group("Source Files\\External\\Nathan" FILES ${External_Nathan_SRCS})
source_group("Source Files\\External\\xmlParser" FILES ${External_xmlParses_SRCS})

