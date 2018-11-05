/*
* mainTrainSet.cxx
*
*  Created on: May 12, 2013
*      Author: amatf
*
* \brief Given a binary file containing features and {+1,-1} reposnses (training
* set) it computes the classifier using Gentle boost code. It saves teh
* classifier so it can be reused later.
* the format of teh binary file is as follows:
* numberOfFeaturesPerSamples[int32], numberOfSamples[int32], xTrain[float *
* numFeatures * numSamples], yTrain [float * numSamples].
*
* ordering of xTrain = [all values for sample 0, all values for sample 1, all
* values for sample 2, etc] -> the fastest running index is over the features
* (not over teh samples)
*/

#if defined(_WIN32) || defined(_WIN64)
#include <io.h>
#include "Shlwapi.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <queue>
#include <string>
#include <vector>
#include "gentleBoost.h"

using namespace std;

//============================================================================
// forward declaration of some auxiliary functions
int readBinaryTrainingFile(string filename, vector<float> &xTrain,
                           vector<float> &yTrain, long long int &numFeatures,
                           long long int &numSamples);

//=============================================================================

int main(int argc, const char **argv) {
  if (argc != 5) {
    std::cout << "ERROR: input arguments are <trainingFilesPath> "
                 "<numWeakClassifiers> <J> <crossValidationPercentage>"
              << endl;
    return 2;
  }

  string trainingFilesPath(argv[1]);
  int numWeakClassifiers = atoi(argv[2]);
  int J = atoi(argv[3]);  // J = 1 -> stumps
  float crossValidationPercentage =
      atof(argv[4]);  // percentage of sample used for training. Set it > 1.0 to
                      // use all the samples

  // variables ot store training data
  long long int numFeatures = -1, numSamples = 0;
  vector<float> xTrain, yTrain;

  // read all the training data from possible binary files in the folder path(
  // recursively)
  long long int hFile;
  int numFiles = 0;

  queue<string> qPath;  // accumulates all possible subfolders
  qPath.push(trainingFilesPath);

  while (qPath.empty() == false) {
    string pathF = qPath.front();
    qPath.pop();
#if defined(_WIN32) || defined(_WIN64)
    _finddata_t c_file;
    if ((hFile = _findfirst((pathF + "/*").c_str(), &c_file)) != -1L) {
      do {
        if (strcmp(".", c_file.name) == 0) continue;
        if (strcmp("..", c_file.name) == 0) continue;

        if ((c_file.attrib & (0x10)) == 0x10)  // file is a subfolder
          qPath.push(pathF + "/" + c_file.name);
        else {
          // check if file is a training file
          string filename(c_file.name);
          if (filename.find("_trainingData.bin") !=
              string::npos)  // match found
          {
            filename = pathF + "/" + string(c_file.name);
            long long int numFeaturesAux, numSamplesAux;
            readBinaryTrainingFile(filename, xTrain, yTrain, numFeaturesAux,
                                   numSamplesAux);

            if (numFeatures < 0)
              numFeatures = numFeaturesAux;
            else if (numFeatures != numFeaturesAux) {
              cout << "ERROR: numFeatures between two trainign data files do "
                      "not match"
                   << endl;
            }

            numSamples += numSamplesAux;
            numFiles++;
          }
        }
      } while (_findnext(hFile, &c_file) == 0);

      _findclose(hFile);
    }
#else
    cout << "ERROR:CODE ONLY READY FOR WINDOWS. TODO: PORT SYSTEM CALLS "
            "_finddata, _findnext TO LINUX"
         << endl;
    exit(3);
// check the web http://anynowhere.com/bb/posts.php?t=275
#endif
  }
  cout << "Read a total of " << numSamples << " samples from " << numFiles
       << " binary files" << endl;

  //====================================================================================
  /*
  cout<<"----------------------------DEBUGGING: import classifier from
MAtlab--------------------"<<endl;
  {
          string classifierFilename("E:/temp/testMatlab_classifier.txt");
          char buffer[256];
          sprintf(buffer,"_J%d_numWaeakClassifiers%d_numSamples%d",J,numWeakClassifiers,
numSamples);
          string itoa(buffer);
          string classifierOutputFileBasename(trainingFilesPath +
"/gentleBoostClassifier");
#if defined(_WIN32) || defined(_WIN64)
          SYSTEMTIME str_t;
          GetSystemTime(&str_t);
          char extra[256];
          sprintf(extra,"_%d_%d_%d_%d_%d_%d",str_t.wYear,str_t.wMonth,str_t.wDay,str_t.wHour,str_t.wMinute,str_t.wSecond);
          string itoaDate(extra);
#else
          sprintf(extra,"_%ld",time(NULL));
#endif

          float *Fx = new float[ numSamples ];
          vector<vector< treeStump > > classifier;
          int err = loadClassifier(classifier, classifierFilename);
          if( err > 0 )
                  cout<<"ERROR: loading classifier "<<classifierFilename<<endl;
          transposeXtrainOutOfPlace(&(xTrain[0]), numSamples, numFeatures);//we
need to perform transposition from GPU features to gentleBoost classifier
          boostingTreeClassifier(&(xTrain[0]),Fx ,classifier , numSamples,
numFeatures);

          string ROCfilename( classifierOutputFileBasename + "_ROCtraining" +
itoa + itoaDate + ".txt");
          ofstream out( ROCfilename.c_str() );
          if( !out.is_open())
          {
                  cout<<"ERROR: opening file "<<ROCfilename<<" to save ROC
curve"<<endl;
          }
          precisionRecallAccuracyCurve(&(yTrain[0]), Fx, numSamples, out, 0.01);
          out.close();
          calibrateBoostingScoreToProbabilityPlattMethod(&(yTrain[0]), Fx,
numSamples);


          delete[] Fx;
  }
  exit(3);
  */
  //======================================================================================

  //---------------perform cross-validation if indicated by
  //user-------------------
  cout << "Separating data into training and testing for cross-validation with "
          "% "
       << crossValidationPercentage << endl;

// initialize random seed for random shuffle of data
#if defined(_WIN32) || defined(_WIN64)
  srand(unsigned(GetTickCount()));
#else
  srand(unsigned(time(NULL)));
#endif

  long long int nPos = 0, nNeg = 0;
  for (vector<float>::const_iterator iter = yTrain.begin();
       iter != yTrain.end(); ++iter) {
    if ((*iter) > 0)
      nPos++;
    else
      nNeg++;
  }

  float *xTrain_train = NULL;
  float *xTrain_test = NULL;
  float *yTrain_train = NULL;
  float *yTrain_test = NULL;
  long long int numSamples_train, numSamples_test;
  long long int nPos_train, nNeg_train;
  if (crossValidationPercentage < 1.0f) {
    long long int nPos_test, nNeg_test;

    nPos_train = (long long int)(crossValidationPercentage * nPos);
    nNeg_train = (long long int)(crossValidationPercentage * nNeg);
    nPos_test = nPos - nPos_train;
    nNeg_test = nNeg - nNeg_train;
    numSamples_train = nNeg_train + nPos_train;
    numSamples_test = nNeg_test + nPos_test;

    vector<long long int> nPosIdx,
        nNegIdx;  // store indexes for positive and negative
    nPosIdx.reserve(nPos);
    nNegIdx.reserve(nNeg);
    for (long long int ii = 0; ii < numSamples; ii++) {
      if (yTrain[ii] < 0.0f)
        nNegIdx.push_back(ii);
      else
        nPosIdx.push_back(ii);
    }

    // shuffle indexes at random to split data
    random_shuffle(nNegIdx.begin(), nNegIdx.end());
    random_shuffle(nPosIdx.begin(), nPosIdx.end());

    // separate both indexes so we can sort indexes for fast memory access
    vector<long long int> nNegIdx_test(nNegIdx.begin(),
                                       nNegIdx.begin() + nNeg_test);
    vector<long long int> nPosIdx_test(nPosIdx.begin(),
                                       nPosIdx.begin() + nPos_test);
    nNegIdx.erase(nNegIdx.begin(), nNegIdx.begin() + nNeg_test);
    nPosIdx.erase(nPosIdx.begin(), nPosIdx.begin() + nPos_test);

    vector<long long int> nIdx_train(nNegIdx);
    nIdx_train.insert(nIdx_train.begin(), nPosIdx.begin(), nPosIdx.end());
    vector<long long int> nIdx_test(nNegIdx_test);
    nIdx_test.insert(nIdx_test.begin(), nPosIdx_test.begin(),
                     nPosIdx_test.end());

    // sort indexes for cache friendly access
    sort(nIdx_test.begin(), nIdx_test.end());
    sort(nIdx_train.begin(), nIdx_train.end());

    xTrain_test = new float[numSamples_test * ((long long int)numFeatures)];
    yTrain_test = new float[numSamples_test * ((long long int)numFeatures)];

    // generate test and training data
    long long int offset = 0;
    long long int auxIdx;
    for (size_t ii = 0; ii < nIdx_test.size(); ii++) {
      auxIdx = nIdx_test[ii];
      yTrain_test[ii] = yTrain[auxIdx];
      memcpy(&(xTrain_test[offset]), &(xTrain[auxIdx * numFeatures]),
             sizeof(float) * numFeatures);
      offset += numFeatures;
    }
    // training data just reorders xTrain (this is why we wanted sort also)
    offset = 0;
    xTrain_train = &(xTrain[0]);
    yTrain_train = &(yTrain[0]);
    for (size_t ii = 0; ii < nIdx_train.size(); ii++) {
      auxIdx = nIdx_train[ii];
      if (ii != auxIdx)  // we need to copy memory
      {
        yTrain_train[ii] = yTrain[auxIdx];
        memcpy(&(xTrain_train[offset]),
               &(xTrain[auxIdx * ((long long int)numFeatures)]),
               sizeof(float) * numFeatures);
      }
      offset += numFeatures;
    }

    cout << "Number of training samples =" << nIdx_train.size()
         << "; number of test samples=" << nIdx_test.size() << endl;

  } else {
    xTrain_train = &(xTrain[0]);
    yTrain_train = &(yTrain[0]);
    numSamples_train = numSamples;
    numSamples_test = 0;
    nNeg_train = nNeg;
    nPos_train = nPos;
  }

  cout << "Finished separating data for cross-validation " << endl;
  //----------------train classifier------------------------------------
  cout << "Starting training for " << numWeakClassifiers
       << " weak classifiers and J=" << J
       << ";number of training samples =" << numSamples_train
       << ";number of features per sample =" << numFeatures << endl;
  transposeXtrainOutOfPlace(xTrain_train, numSamples_train,
                            numFeatures);  // we need to perform transposition
                                           // from GPU features to gentleBoost
                                           // classifier
  vector<vector<treeStump> > classifier;

  string classifierOutputFileBasename(trainingFilesPath +
                                      "/gentleBoostClassifier");

  // add weights (they are normalized inside teh training code)
  float *wTrain = new float[numSamples_train];
  float negPosRatio = ((float)nNeg_train) / ((float)nPos_train);
  for (long long int ii = 0; ii < numSamples_train; ii++) {
    if (yTrain[ii] < 0)
      wTrain[ii] = 1.0;
    else
      wTrain[ii] = negPosRatio;
  }
  gentleTreeBoost(xTrain_train, yTrain_train, wTrain, numWeakClassifiers,
                  classifier, numSamples_train, numFeatures, J);

  delete[] wTrain;
  cout << "Finished training" << endl;

  //----------------test cross-validation if indicated by
  //user--------------------

  char buffer[256];
  sprintf(buffer, "_J%d_numWaeakClassifiers%d_numSamples%lld", J,
          numWeakClassifiers, numSamples_train);
  string itoa(buffer);
#if defined(_WIN32) || defined(_WIN64)
  SYSTEMTIME str_t;
  GetSystemTime(&str_t);
  char extra[256];
  sprintf(extra, "_%d_%d_%d_%d_%d_%d", str_t.wYear, str_t.wMonth, str_t.wDay,
          str_t.wHour, str_t.wMinute, str_t.wSecond);
#else
  char extra[256];
  sprintf(extra, "_%ld", time(NULL));
#endif
  string itoaDate(extra);

  cout << "Starting testing for " << numWeakClassifiers
       << " weak classifiers and J=" << J
       << ";number of testing samples =" << numSamples_test
       << ";number of features per sample =" << numFeatures << endl;
  float *Fx = NULL;
  if (numSamples_test > 0) {
    Fx = new float[numSamples_test];
    // transposeXtrainOutOfPlace(xTrain_test, numSamples_test, numFeatures);//we
    // need to perform transposition from GPU features to gentleBoost classifier
    boostingTreeClassifierTranspose(xTrain_test, Fx, classifier,
                                    numSamples_test, numFeatures);

    string ROCfilename(classifierOutputFileBasename + "_ROC" + itoa + itoaDate +
                       ".txt");
    ofstream out(ROCfilename.c_str());
    if (!out.is_open()) {
      cout << "ERROR: opening file " << ROCfilename << " to save ROC curve"
           << endl;
    }
    precisionRecallAccuracyCurve(yTrain_test, Fx, numSamples_test, out);
    out.close();
    calibrateBoostingScoreToProbabilityPlattMethod(yTrain_test, Fx,
                                                   numSamples_test);
  } else {  // test on training: training has been transposed already!!!
    Fx = new float[numSamples_train];
    // transposeXtrainOutOfPlace(xTrain_train, numSamples_train,
    // numFeatures);//we need to perform transposition from GPU features to
    // gentleBoost classifier
    boostingTreeClassifier(xTrain_train, Fx, classifier, numSamples_train,
                           numFeatures);

    string ROCfilename(classifierOutputFileBasename + "_ROCtraining" + itoa +
                       itoaDate + ".txt");
    ofstream out(ROCfilename.c_str());
    if (!out.is_open()) {
      cout << "ERROR: opening file " << ROCfilename << " to save ROC curve"
           << endl;
    }
    precisionRecallAccuracyCurve(yTrain_train, Fx, numSamples_train, out);
    out.close();
    calibrateBoostingScoreToProbabilityPlattMethod(yTrain_train, Fx,
                                                   numSamples_train);
  }
  cout << "Finished testing" << endl;
  //----------------save classifier-------------------------------------
  string classifierFilename(classifierOutputFileBasename + itoa + itoaDate +
                            ".txt");
  cout << "Saving classifier in file " << classifierFilename << endl;

  int err = saveClassifier(classifier, classifierFilename);
  if (err > 0) return err;

  if (numSamples_test > 0) {
    float *FxTest = new float[numSamples_test];
    cout << "Testing that classifier can be load correctly...";
    vector<vector<treeStump> > classifierTest;
    err = loadClassifier(classifierTest, classifierFilename);
    if (err > 0) return err;

    float norm2 = -1e32f;
    /* If stumpthr == xVal it can trigger errors just because of float precision
    when using the < operator in the weak classifier
    boostingTreeClassifierTranspose(xTrain_test,FxTest ,classifierTest ,
    numSamples_test, numFeatures);
    for(long long int ii = 0 ; ii<numSamples_test; ii++)
    {
            float auxFx = sqrt((Fx[ii]-FxTest[ii])*(Fx[ii]-FxTest[ii]));
            norm2 = max(norm2 ,  auxFx );
            if(auxFx > 0.1)
            {
                    cout<<"I am here. ii =
    "<<ii<<";"<<Fx[ii]<<";"<<FxTest[ii]<<endl;
            }
    }
    */

    float norm = 0.0f;
    if (classifierTest.size() != classifier.size()) norm = 1e32;
    for (unsigned int ii = 0; norm < 0.1 && ii < classifierTest.size(); ii++) {
      if (classifierTest[ii].size() != classifier[ii].size()) {
        norm = 1e32;
        break;
      }
      for (unsigned int jj = 0; jj < classifierTest[ii].size(); jj++) {
        if (classifierTest[ii][jj] != classifier[ii][jj]) {
          norm = 1e32;
          break;
        }
      }
    }

    if (norm > 0.1)
      if (norm2 > 0.1)  // both errors
        cout << " ERROR!!!ERROR!!! INCORRECT equality comparison!!! INCORRECT "
                "FX comparison. Inf-Norm = "
             << norm2 << endl;
      else  // only equality error
        cout << " ERROR!!!ERROR!!! INCORRECT equality comparison!!!" << endl;
    else if (norm2 > 0.1)  // only Fx error
      cout << " ERROR!!!ERROR!!! INCORRECT FX comparison. Inf-Norm = " << norm2
           << endl;
    else
      cout << "OK" << endl;
    delete[] FxTest;
  }

  //-----------release memory

  if (xTrain_train != (&(xTrain[0])) && xTrain_train != NULL)
    delete[] xTrain_train;
  if (yTrain_train != (&(yTrain[0])) && yTrain_train != NULL)
    delete[] yTrain_train;
  if (xTrain_test != NULL) delete[] xTrain_test;
  if (yTrain_test != NULL) delete[] yTrain_test;

  xTrain.clear();
  yTrain.clear();
  if (Fx != NULL) delete[] Fx;
}

//===============================================================================
int readBinaryTrainingFile(string filename, vector<float> &xTrain,
                           vector<float> &yTrain, long long int &numFeatures,
                           long long int &numSamples) {
  ifstream fin(filename.c_str(), ios::binary | ios::in);

  if (fin.is_open() == false) {
    cout << "ERROR: readBinaryTrainingFile: could not open file " << filename
         << " to read training data" << endl;
    return 2;
  }

  int aux;
  fin.read((char *)(&aux), sizeof(int));
  numFeatures = aux;
  fin.read((char *)(&aux), sizeof(int));
  numSamples = aux;

  cout << "Reading " << numFeatures << " features per sample for " << numSamples
       << " samples from file " << filename << endl;

  // resize vectors (adding to existing elements)
  size_t xTrainOffset = xTrain.size();
  size_t yTrainOffset = yTrain.size();
  xTrain.resize(xTrainOffset + numSamples * numFeatures);
  yTrain.resize(yTrainOffset + numSamples);

  // read from file
  fin.read((char *)(&(xTrain[xTrainOffset])),
           sizeof(float) * numFeatures * numSamples);
  fin.read((char *)(&(yTrain[yTrainOffset])), sizeof(float) * numSamples);

  fin.close();

  return 0;
}
