/*
* Copyright (C) 2011-2013 by  Fernando Amat
* See license.txt for full license and copyright notice.
*
* Authors: Fernando Amat
*
* mainProcessStackBatchMulticore.cpp
*
*  Created on: December 17th, 2013
*      Author: Fernando Amat
*
* \brief Generates a hierachical segmentation of a 3D stack and saves teh result
* in binary format
*
*/

#include <fstream>
#include <iostream>
#include <string>

#include "parseConfigFile.h"

#if defined(_WIN32) || defined(_WIN64)
#define NOMINMAX
#include <Windows.h>
#include "Shlwapi.h"
#endif

#pragma comment(lib, "Shlwapi.lib")

using namespace std;

int generateBatchScriptFile(string scriptFilename, string ProcessStackPath,
                            string configFilename, int iniFrame, int endFrame,
                            int numCores);
void parseImageFilePattern(string& imgRawPath, int frame);

int main(int argc, const char** argv) {
  if (argc != 4 && argc != 5) {
    cout << "ERROR: wrong number of parameters. Call executable with "
            "<configFile> <iniFrame> <endFrame> <numCores-optional>"
         << endl;
    return 2;
  }

  string configFilename(argv[1]);
  int iniFrame = atoi(argv[2]);
  int endFrame = atoi(argv[3]);

  // find out number of cores in the machine
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);

  int numCores = sysinfo.dwNumberOfProcessors;
  if (argc == 5) numCores = atoi(argv[4]);

  // find out folder of the executable
  HMODULE hModule = GetModuleHandleW(NULL);
  char path[MAX_PATH];
  GetModuleFileName(hModule, path, MAX_PATH);

  // change to process stack
  PathRemoveFileSpec(path);
  string ProcessStackPath(string(path) + "\\ProcessStack.exe");

  // generate script file
  //  Gets the temp path env string (no guarantee it's a valid path).
  TCHAR szTempFileName[MAX_PATH];
  TCHAR lpTempPathBuffer[MAX_PATH];
  DWORD dwRetVal = GetTempPath(MAX_PATH, lpTempPathBuffer);  // buffer for path
  //  Generates a temporary file name.
  UINT uRetVal =
      GetTempFileName(lpTempPathBuffer,               // directory for tmp files
                      TEXT("TGMMProcessStackBatch"),  // temp file name prefix
                      0,                              // create unique name
                      szTempFileName);                // buffer for name
  // change extension
  PathRenameExtension(szTempFileName, ".cmd");
  string scriptFilename(szTempFileName);

  int err =
      generateBatchScriptFile(scriptFilename, ProcessStackPath, configFilename,
                              iniFrame, endFrame, numCores);
  if (err > 0) return err;

  // call script
  system(scriptFilename.c_str());

  // check if files have geen generated (it is equivalent to assume successful
  // execution)
  cout << "======================================================" << endl;
  cout << "======================================================" << endl;
  cout << "Checking if hierarchical segmentation files were properly created"
       << endl;
  configOptionsTrackingGaussianMixture configOptions;
  bool flagError = false;
  for (int frame = iniFrame; frame <= endFrame; frame++) {
    int err =
        configOptions.parseConfigFileTrackingGaussianMixture(configFilename);
    if (err != 0) return err;

    string basename(configOptions.imgFilePattern);
    parseImageFilePattern(basename, frame);

    char buffer[256];
    int radiusMedianFilter = configOptions.radiusMedianFilter;
    int conn3D = configOptions.conn3D;
    sprintf(buffer, "_hierarchicalSegmentation_conn3D%d_medFilRad%d", conn3D,
            radiusMedianFilter);
    string suffix(buffer);
    string fileOutHS(basename + suffix + ".bin");

    // check if file exists
    ifstream fin(fileOutHS.c_str());
    if (fin.is_open() == false) {
      cout << "====ERROR: hierarchical segmentation file " << fileOutHS
           << " was not created.====" << endl;
      cout << "Run the following command to analyze the error: " << endl;
      cout << ProcessStackPath << " \"" << configFilename << "\" " << frame
           << endl;
      flagError = true;
    } else {
      cout << "OK: Hierarchical segmentation file " << fileOutHS
           << " was successfully created" << endl;
    }
    fin.close();
  }

  if (flagError == false)
    cout << "Hierarchical segmentation ran successfully" << endl;
  else
    cout << "ERRORS ocurred while running hierarchical segmentation" << endl;

  return 0;
}

//============================================================
int generateBatchScriptFile(string scriptFilename, string ProcessStackPath,
                            string configFilename, int iniFrame, int endFrame,
                            int numCores) {
  ofstream fout(scriptFilename.c_str());

  if (fout.is_open() == false) {
    fout << "ERROR: generateBatchScriptFile: file " << scriptFilename
         << " could not be generated to run batch script" << endl;
    return 2;
  }

  fout << "@echo off" << endl;
  fout << "setlocal enableDelayedExpansion" << endl;

  fout << ":: Display the output of each process if the /O option is used"
       << endl;
  fout << ":: else ignore the output of each process" << endl;
  fout << "if /i \"%~1\" equ \"/O\" (" << endl;
  fout << "  set \"lockHandle=1\"" << endl;
  fout << "set \"showOutput=1\"" << endl;
  fout << ") else (" << endl;
  fout << "set \"lockHandle=1^>nul 9\"" << endl;
  fout << "set \"showOutput=\"" << endl;
  fout << ")" << endl;

  fout << ":: List of commands goes here. Each command is prefixed with :::"
       << endl;

  for (int ii = iniFrame; ii <= endFrame; ii++) {
    fout << "::: " << ProcessStackPath << " \"" << configFilename << "\" " << ii
         << endl;
  }

  fout << ":: Define the maximum number of parallel processes to run." << endl;
  fout << ":: Each process number can optionally be assigned to a particular "
          "server"
       << endl;
  fout << ":: and/or cpu via psexec specs (untested)." << endl;

  fout << "set \"maxProc=" << numCores << "\"" << endl;

  fout << ":: Optional - Define CPU targets in terms of PSEXEC specs" << endl;
  fout << "::           (everything but the command)" << endl;
  fout << "::" << endl;
  fout << ":: If a cpu is not defined for a proc, then it will be run on the "
          "local machine."
       << endl;
  fout << ":: I haven't tested this feature, but it seems like it should work."
       << endl;
  fout << "::" << endl;
  fout << ":: set cpu1=psexec \\server1 ..." << endl;
  fout << ":: set cpu2=psexec \\server1 ..." << endl;
  fout << ":: set cpu3=psexec \\server2 ..." << endl;
  fout << ":: etc." << endl;

  fout << ":: For this demo force all cpu specs to undefined (local machine)"
       << endl;
  fout << "for /l %%N in (1 1 %maxProc%) do set \"cpu%%N=\"" << endl;

  fout << ":: Get a unique base lock name for this particular instantiation."
       << endl;
  fout << ":: Incorporate a timestamp from WMIC if possible, but don't fail if"
       << endl;
  fout << ":: WMIC not available. Also incorporate a random number." << endl;
  fout << "set \"lock=\"" << endl;
  fout << "for /f \"skip=1 delims=-+ \" %%T in ('2^>nul wmic os get "
          "localdatetime') do ("
       << endl;
  fout << "set \"lock=%%T\"" << endl;
  fout << "goto :break" << endl;
  fout << ")" << endl;
  fout << ":break" << endl;
  fout << "set \"lock=%temp%lock%lock%_%random%_\"" << endl;
  fout << "" << endl;
  fout << ":: Initialize the counters" << endl;
  fout << "set /a \"startCount=0, endCount=0\"" << endl;
  fout << "" << endl;
  fout << ":: Clear any existing end flags" << endl;
  fout << "for /l %%N in (1 1 %maxProc%) do set \"endProc%%N=\"" << endl;
  fout << "" << endl;
  fout << ":: Launch the commands in a loop" << endl;
  fout << "set launch=1" << endl;
  fout << "for /f \"tokens=* delims=:\" %%A in ('findstr /b \":::\" \"%~f0\"') "
          "do ("
       << endl;
  fout << "if !startCount! lss %maxProc% (" << endl;
  fout << "set /a \"startCount+=1, nextProc=startCount\"" << endl;
  fout << ") else (" << endl;
  fout << "call :wait" << endl;
  fout << ")" << endl;
  fout << "set cmd!nextProc!=%%A" << endl;
  fout << "if defined showOutput echo "
          "--------------------------------------------------------------------"
          "-----------"
       << endl;
  fout << "echo !time! - proc!nextProc!: starting %%A" << endl;
  fout << "2>nul del %lock%!nextProc!" << endl;
  fout << "%= Redirect the lock handle to the lock file. The CMD process will  "
          "   =%"
       << endl;
  fout << "%= maintain an exclusive lock on the lock file until the process "
          "ends. =%"
       << endl;
  fout << "start /b "
          " cmd /c %lockHandle%^>\"%lock%!nextProc!\" 2^>^&1 !cpu%%N! %%A"
       << endl;
  fout << ")" << endl;
  fout << "set \"launch=\"" << endl;
  fout << "" << endl;
  fout << ":wait" << endl;
  fout << ":: Wait for procs to finish in a loop" << endl;
  fout << ":: If still launching then return as soon as a proc ends" << endl;
  fout << ":: else wait for all procs to finish" << endl;
  fout << ":: redirect stderr to null to suppress any error message if "
          "redirection"
       << endl;
  fout << ":: within the loop fails." << endl;
  fout << "for /l %%N in (1 1 %startCount%) do (" << endl;
  fout << "%= Redirect an unused file handle to the lock file. If the process "
          "is    =%"
       << endl;
  fout << "%= still running then redirection will fail and the IF body will "
          "not run =%"
       << endl;
  fout << "if not defined endProc%%N if exist \"%lock%%%N\" (" << endl;
  fout << "%= Made it inside the IF body so the process must have finished =%"
       << endl;
  fout << "if defined showOutput echo "
          "===================================================================="
          "==========="
       << endl;
  fout << "echo !time! - proc%%N: finished !cmd%%N!" << endl;
  fout << "if defined showOutput type \"%lock%%%N\"" << endl;
  fout << "if defined launch (" << endl;
  fout << "set nextProc=%%N" << endl;
  fout << "  exit /b" << endl;
  fout << ")" << endl;
  fout << "set /a \"endCount+=1, endProc%%N=1\"" << endl;
  fout << ") 9>>\"%lock%%%N\"" << endl;
  fout << ") 2>nul" << endl;
  fout << "if %endCount% lss %startCount% (" << endl;
  fout << "1>nul 2>nul ping /n 2 ::1" << endl;
  fout << "goto :wait" << endl;
  fout << ")" << endl;
  fout << "" << endl;
  fout << "2>nul del %lock%*" << endl;
  fout << "if defined showOutput echo "
          "===================================================================="
          "==========="
       << endl;
  // fout<<"echo Thats all folks!"<<endl;
  fout << "" << endl;

  fout.close();
  return 0;
}

//==================================================================================
//=======================================================================
void parseImageFilePattern(string& imgRawPath, int frame) {
  size_t found = imgRawPath.find_first_of("?");
  while (found != string::npos) {
    int intPrecision = 0;
    while ((imgRawPath[found] == '?') && found != string::npos) {
      intPrecision++;
      found++;
      if (found >= imgRawPath.size()) break;
    }

    char bufferTM[16];
    switch (intPrecision) {
      case 2:
        sprintf(bufferTM, "%.2d", frame);
        break;
      case 3:
        sprintf(bufferTM, "%.3d", frame);
        break;
      case 4:
        sprintf(bufferTM, "%.4d", frame);
        break;
      case 5:
        sprintf(bufferTM, "%.5d", frame);
        break;
      case 6:
        sprintf(bufferTM, "%.6d", frame);
        break;
    }
    string itoaTM(bufferTM);

    found = imgRawPath.find_first_of("?");
    imgRawPath.replace(found, intPrecision, itoaTM);

    // find next ???
    found = imgRawPath.find_first_of("?");
  }
}
