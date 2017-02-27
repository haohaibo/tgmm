This file contains basic instructions for installing and running the cell lineaging software package “Tracking with Gaussian Mixture Models” (TGMM). The code provided here has been tested with the 64-bit version of Windows 7 and with the 64-bit version of Ubuntu Linux 12.04 LTS, using a variety of CUDA-compatible NVIDIA GPUs.


1.-CONTENTS OF THE SOFTWARE ARCHIVE
-----------------------------------
We assume that the user has uncompressed the file “Supplementary_Software_1.zip” in a folder of their choice, referred to here as $ROOT_TGMM. The subfolders in $ROOT_TGMM contain the following components:

-"src": All source code files. This folder also includes a CMakeList.txt file that can be used to generate a Visual Studio solution (using CMake) and compile the source code.
-"doc": Documentation of the TGMM software.
-"build": A Visual Studio C++ 2010 project generated from src using CMake. This subfolder also contains precompiled binaries, suitable for running the code without the need for re-compiling the source code.
-"data": Contains a three-dimensional time-lapse data set with 31 time points (corresponding to a cropped sub-region of the Drosophila SiMView recording presented in the main text), for testing the TGMM code and ensuring that the software is running as expected.

Note: The Visual Studio project will not compile unless the folder "build" is copied to the same absolute path as that used to generate the project. We provide the full project folder primarily as a reference for the final structure of a successful Visual Studio solution.


2.-INSTALLATION AND SOFTWARE REQUIREMENTS 
-----------------------------------------

In order to run the precompiled binaries, the following auxiliary software package must be installed as well:

-CUDA Toolkit 5.5: required to run algorithms on an NVIDIA GPU
 Download: https://developer.nvidia.com/cuda-toolkit-archive

We provide precompiled binaries for the 64-bit version of Windows 7. The folder with the precompiled binaries also contains all required DLLs, and thus no external software packages other than the CUDA drivers mentioned above need to be installed. The software can effectively be run out-of-the-box, as detailed below in section 3.

For Linux, compilation of the source code is required (see detailed instructions in section 2.1). For compiling the source code, any software version equal to or above the CUDA Toolkit software version listed above should suffice.

For possible common runtime errors and solutions see section 5.


2.1-Source code compilation in Linux

A.-Make sure CMake is installed (http://www.cmake.org/). For Ubuntu distributions, you can simply use the following command:
	sudo apt-get install cmake cmake-gui

B.-Go to the folder $ROOT_TGMM and create a new folder called "build", where the binaries will subsequently be generated:
	cd $ROOT_TGMM
	mkdir build
	cd build

C.-In the build folder, execute the following commands:
	cmake -D CMAKE_BUILD_TYPE=RELEASE ../src/
	make

The first command locates all libraries (for example, from the CUDA Toolkit) and generates all necessary makefiles to compile the code. The second command calls these makefiles. After executing the second command, you should see messages in the terminal commenting on the compilation progress. If the progress report reaches 100%, the program has compiled successfully. After successful compilation, the executables $ROOT_TGMM/build/nucleiChSvWshedPBC/ProcessStack and $ROOT_TGMM/build/TGMM should be present. You can use cmake-gui or cmake options to change different parameters in the makefiles (for example, final destination folder or CUDA architecture level).


3.-RUNNING THE TGMM SOFTWARE
----------------------------
We provide a test data set that allows the user to test the code and familiarize themselves with software configuration before applying the code to their own data sets. Currently, 2D+time and 3D+time datasets with 8-bit or 16-bit unsigned integer TIFF stacks are supported as the input data format. The two-dimensional or three-dimensional image data recorded for each time point should be provided as a single TIFF file.


3.1-Configuration file

The file “$ROOT_TGMM\data\TGMM_configFile.txt” serves as a template for the configuration file and contains all parameters required to run the TGMM code. In principle (and for all results presented in this study), only parameters listed under “main parameters” need to be modified for each new experiment. Access to parameters listed under “advanced parameters” is provided as well and is intended for experienced users who wish to experiment further with the code.

Each parameter is accompanied by a description of its functionality (see section "Overview of advanced framework parameters" in the Supplementary Materials for more details). In order to process a new data set, simply copy the configuration text file and adjust parameters as needed.

IMPORTANT NOTE: Before applying the TGMM software to the test data set, the variables debugPathPrefix and imgFilePattern in the configuration file need to be adjusted, so the software can locate the image stacks (imgFilePattern) and save the results (debugPathPrefix).


3.2-Watershed segmentation with persistence-based agglomeration

WINDOWS
-------
In order to generate the hierarchical segmentation for each time point, follow these three steps:
A.-Open a Windows command line terminal (run “cmd.exe”).
B.-Go to the folder “$ROOT_TGMM\build\nucleiChSvWshedPBC\Release”.
C.-Execute the command “ProcessStackBatchMulticore.exe $ROOT_TGMM\data\TGMM_configFile.txt 0 30”.

The program automatically detects how many processing cores are present in the workstation and parallelizes the image segmentation task accordingly. The last two arguments are the first time point and the last time point of the time-lapse image data set.

Once processing is complete, new files “$ROOT_TGMM\data\TM?????_timeFused_blending\ SPC0_CM0_CM1_CHN00_CHN01.fusedStack_?????_hierarchicalSegmentation_conn3D74_medFilRad2.bin“ should have been generated (one for each time point). These binary files store all information required to restore the hierarchical segmentation for each time point. If the binary files were not created, an error occurred during execution of “ProcessStackBatchMulticore.exe” and a corresponding error message is displayed in the terminal.

LINUX
-----
In order to generate the hierarchical segmentation for each time point, follow these three steps:
A.-Open a terminal.
B.-Go to the folder “$ROOT_TGMM/build/nucleiChSvWshedPBC”.
C.-Execute the following command: parallel -j8 ./ProcessStack $ROOT_TGMM\data\TGMM_configFile_linux.txt -- {0..30}

The option -j8 indicates how many cores should be used in parallel (in this case 8). The last option, {0..30}, indicates that the program ProcessStack should be executed for time points 0 to 30.

IMPORTANT NOTE: The command parallel is part of the GNU software (http://www.gnu.org/software/parallel/). The program presents an easy interface to call programs in parallel. If this software is not already installed, it can be downloaded from the GNU website or installed from official repositories. For example, in Ubuntu you can simply use the following command: sudo apt-get install moreutils

IMPORTANT NOTE: Make sure to use TGMM_configFile_linux.txt instead of TGMM_configFile.txt, since the latter contains Windows end-of-line symbols that will lead to a failure during code parsing in Linux. You can also use the tool dos2unix to ensure that any given text file can be used as a config file.


3.3-Bayesian sequential tracking with Gaussian Mixture Models

In order to track cell nuclei and reconstruct cell lineages, follow these three steps (the same instructions are valid for Windows and Linux):
A.-Open a Windows command line terminal (run “cmd.exe” in Windows).
B.-Go to the folder “$ROOT_TGMM\build\Release”.
C.-Execute the command: “TGMM.exe $ROOT_TGMM\data\TGMM_configFile.txt 0 30”

The command line will display notifications about the progress of the tracking and segmentation algorithm. Since the hierarchical segmentation results from step 3.2 are saved separately in the “.bin” files, different tracking parameter settings can be tested without the need for recalculating or changing the segmentation data. The output data format of the tracking module is explained in section 4.


3.4-Verifying successful program execution

In order to simplify the verification of successful TGMM software execution, we provide the output for the test data set in “$ROOT_TGMM\data\TGMMruns_testRunToCheckOutput”. The output generated by your execution of the program should be very similar to the contents of this folder.


4.-TRACKING AND SEGMENTATION OUTPUT DATA FORMAT
-----------------------------------------------
The folder “debugPathPrefix\GMEMtracking3D_%date“ contains the output of the TGMM run.

The final result can be found in the subfolder “$debugPathPrefix\GMEMtracking3D_%date\ XML_finalResult_lht” or “$debugPathPrefix\GMEMtracking3D_%date\XML_finalResult_lht_ bckgRm”. The latter directory is used if the user applied the background classifier. The output subfolder contains one XML file and one “.svb” file per time point.

The XML file contains the main tracking and segmentation information. Each object is stored under the tag <GaussianMixtureModel> with the following attributes:

-id [integer]: unique id of the object in this particular time point.
-lineage [integer]: unique id of the cell lineage the object belongs to.
-parent [integer]: id of the linked object at the previous time point. Following the chain of “parent” objects reconstructs the track. A value of -1 indicates the birth of a track.
-splitScore [float]: confidence level for the correct tracking of this particular object. A value of 0 indicates very low confidence and a value of 5 indicates very high confidence. Sorting elements by confidence level can guide the user in the data curation process and facilitate more effective editing of the TGMM results (see main text and Fig. 4).
-scale [float[3]]: voxel scaling factors along the x-, y- and z-axis.
-nu, beta, alpha [float]: value of the hyper-parameters for the Bayesian GMM.
-m [float[3]]: mean of the Gaussian Mixture (object centroid, in pixels).
-W [float[3][3]]: precision matrix of the Gaussian Mixture (object shape).
-*Prior: same as before, but for prior values obtained from the previous time point. These values are used during the inference procedure.
-svIdx [integer[]]: list of indices of the super-voxels clustered by this Gaussian. Together with the “.svb” file, this information can be used to obtain precise segmentation regions for each object.

The “.svb” file is a binary file in a custom format that can be read with the constructor “supervoxel::supervoxel(istream& is)”. Briefly, it contains information about all super-voxels generated at a particular time point. Thus, using the “svIdx” attribute, the precise segmentation mask for each object can be recovered.


5.-TROUBLESHOOTING COMMON RUNTIME ERRORS
----------------------------------------

5.1-Program execution starts and one of the following error messages is displayed in the terminal: "no CUDA-capable device is detected" or "CUDA driver version is insufficient for CUDA runtime version".

First, confirm that the workstation is equipped with an NVIDIA CUDA-capable graphics card. This is a hardware requirement for running the software. If such a card is installed, you most likely need to update the driver in order to be compatible with CUDA Toolkit 5.5. Go to https://developer.nvidia.com/cuda-downloads and download the current toolkit. The toolkit will also install an updated NVIDIA driver.

5.2-When you try to run the program from the terminal, a Windows dialog pops up with the following message "The program can't start because msvcp100.dll is missing from your computer”. 

For some reason, the provided DLL from the Microsoft Visual C++ 2010 SP1 Redistributable Package (x64) is not compatible with your windows version. Delete the DLL from the TGMM software folder and go to http://www.microsoft.com/en-us/download/confirmation.aspx?id=13523 to download and install the appropriate version of the Microsoft Visual C++ 2010 SP1 Redistributable Package.

5.3-Note that the program needs to be called from a “cmd.exe” terminal in Windows. Cygwin or MinGw terminals cause the program to fail.

5.4-“ProcessStackBatchMulticore.exe” requires paths to be provided using absolute path names. The use of relative path names also causes the program to fail.

5.5-Note that the parameter “imgFilePattern” in the configuration file “TGMM_configFile.txt” requires the use of forward slashes in path names (since the image library used to read TIFF files follows the Unix convention), whereas the parameter “debugPathPrefix” in the same file requires the use of double backslashes in path names (since backslashes are special characters that are interpreted by the operating system). On Linux systems, always use forward slashes in both parameters.

5.6-Program execution starts and one of the following error messages is displayed on the terminal: "invalid device symbol in C:/ROOT_TGMM/src/nucle/iChSvWshedPBC/CUDAmedianFilter2D/medianFilter2D.cu at line 230"

The provided binaries were compiled for CUDA compute capability 2.0 or higher. If your NVIDIA GPU card has a lower CUDA compute capability (this information is available from https://developer.nvidia.com/cuda-gpus), the provided binaries will not work. However, you can recompile the source code, which should allow you to run the software. Before compiling, you need to edit the CMakeLists.txt file and modify the line at the top: set(SETBYUSER_CUDA_ARCH sm_20 CACHE STRING "CUDA architecture"). Adjust the flag sm_20 to the appropriate CUDA compute capability of your NVIDIA GPU (for example, sm_13 for CUDA compute capability 1.3).