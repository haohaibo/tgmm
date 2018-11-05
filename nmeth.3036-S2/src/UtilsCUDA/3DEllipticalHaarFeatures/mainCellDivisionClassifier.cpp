#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include "AnnotationEllipsoid.h"
#include "EllipticalHaarFeatures.h"
#include "external/Nathan/tictoc.h"
#include "external/xmlParser/xmlParser.h"
#include "gentleBoost/gentleBoost.h"

namespace mylib {
#include "mylib/image.h"
}

#if defined(_WIN32) || defined(_WIN64)

#include <Windows.h>

#endif

using namespace std;
int main(int argc, const char **argv) {
  // for cell division
  // cout<<"Training classifier for cell division"<<endl;
  // string
  // annotationsFilenameXML("C:/Users/Fernando/TrackingNuclei/matlabCode/visualization/annotations/finalsCellDivisionClassifierOriginalImageBased/classifierAnnotations_2011_2013_cat.xml2");
  // string positiveClassLabel("twocell");//the class in the annotations that is
  // considered a positive sample

  // for cell background (based only on image features)
  cout << "Training classifier for cell versus background" << endl;
  string annotationsFilenameXML(
      "C:/Users/Fernando/TrackingNuclei/matlabCode/visualization/annotations/"
      "finalsCellVersusBackgroundClassifier/"
      "classifierAnnotationsBackground_cat.xml2");
  string positiveClassLabel("nocell");  // the class in the annotations that is
                                        // considered a positive sample

  string classifierOutputFileBasename(
      "C:/Users/Fernando/TrackingNuclei/matlabCode/cellDivisionModel/"
      "classifiers/classifier");
  int devCUDA = 0;
  // int symmetryN = 1;//calculate features for this image using all possible 8
  // combinations of eigenvectors directions to artifically increment the
  // traning data
  // int numWeakClassifiers = 50;
  // int J = 1;
  // float crossValidationPercentage = 0.6f;//percentage of sample used for
  // training. Set it > 1.0 to use all the samples

  // basicEllipticalHaarFeatureVector::useDoGfeatures = false;//uncomment this
  // if you want to test results without DoG features
  if (basicEllipticalHaarFeatureVector::useDoGfeatures == false)
    std::cout << "======================WARNING: TRAINING WITHOUT DoG FEATURES "
                 "ENHANCEMENT========================="
              << endl;

  if (argc != 5) {
    std::cout << "ERROR: input arguments are <numWeakClassifiers> <J> "
                 "<symmetryN> <crossValidationPercentage>"
              << endl;
    return 2;
  }

  int numWeakClassifiers = atoi(argv[1]);
  int J = atoi(argv[2]);
  int symmetryN = atoi(argv[3]);  // calculate features for this image using all
                                  // possible 8 combinations of eigenvectors
                                  // directions to artifically increment the
                                  // traning data
  float crossValidationPercentage =
      atof(argv[4]);  // percentage of sample used for training. Set it > 1.0 to
                      // use all the samples

  //=====================================================================================

  TicTocTimer tt = tic();

  symmetryN = max(symmetryN, 1);  // minimum value
  symmetryN = min(symmetryN, 8);  // maximum value

  //----------------read annotations-----------------------------------------

  XMLNode xMainNode =
      XMLNode::openFileHelper(annotationsFilenameXML.c_str(), "document");
  int n = xMainNode.nChildNode("Surface");

  vector<AnnotationEllipsoid> annotationsVec(n);
  long long int nPos = 0, nNeg = 0;
  for (int ii = 0; ii < n; ii++) {
    annotationsVec[ii] = AnnotationEllipsoid(xMainNode, ii);
    // parse class name to labels
    if (annotationsVec[ii].className.compare(positiveClassLabel) ==
        0)  // cell division
    {
      annotationsVec[ii].classVal = 1;
      nPos++;
    } else {
      annotationsVec[ii].classVal = 0;
      nNeg++;
    }
  }

  cout << "Read " << n << " annotations in " << toc(&tt)
       << " secs. Positives = " << nPos << "; negatives=" << nNeg
       << "; symmetry = " << symmetryN << endl;

  //----------------calculate features-------------------------------------

  // sort vector of features by image filename
  sort(annotationsVec.begin(), annotationsVec.end());
  int sizeW = dimsImage * (1 + dimsImage) / 2;

  // preallocate memory for training
  int numFeatures = getNumberOfHaarFeaturesPerllipsoid();
  long long int numSamples = ((long long int)symmetryN) * ((long long int)n);
  nPos *= ((long long int)symmetryN);
  nNeg *= ((long long int)symmetryN);
  feature *xTrain = new feature[numSamples * ((long long int)numFeatures)];
  feature *yTrain = new feature[numSamples];

  long long int offsetXtrain = 0;

  string imgFilenameOld = annotationsVec[0].imgFilename;
  for (int ii = 0; ii < n; ii++) {
    // check filename exists
    ifstream checkFile(imgFilenameOld.c_str());
    if (checkFile.is_open() == false) {
      cout << "ERROR: image " << imgFilenameOld
           << " cannot be found to calculate features on annotation" << endl;
      return 3;
    }

    int numEllipsoids = 0;
    int offset = ii;
    feature yTrainAux;
    int yTrainOffset = ii * symmetryN;
    int yTrainOffsetAnchor = yTrainOffset;
    while (ii < n &&
           annotationsVec[ii].imgFilename.compare(imgFilenameOld) == 0) {
      yTrainAux = 2.0 * (((feature)annotationsVec[ii].classVal) - 0.5);
      // for(int aa = 0; aa<symmetryN; aa++)
      //{
      yTrain[yTrainOffset++] = yTrainAux;
      //}
      numEllipsoids++;
      ii++;
    }

    // duplicate labels for all symmetries
    feature *yTrainSrc = &(yTrain[yTrainOffsetAnchor]);
    for (int aa = 1; aa < symmetryN; aa++) {
      yTrainOffsetAnchor += numEllipsoids;
      memcpy(&(yTrain[yTrainOffsetAnchor]), yTrainSrc,
             sizeof(feature) * numEllipsoids);
    }

    cout << "Calculating features for " << numEllipsoids
         << " annotations in image " << imgFilenameOld << endl;

    // allocate memory
    double *m = new double[dimsImage * numEllipsoids];
    double *W = new double[sizeW * numEllipsoids];
    int pos = 0;
    for (int jj = offset; jj < ii; jj++, pos++) {
      for (int aa = 0; aa < dimsImage; aa++) {
        m[pos + aa * numEllipsoids] = annotationsVec[jj].mu[aa];
      }
      for (int aa = 0; aa < sizeW; aa++) {
        W[pos + aa * numEllipsoids] = annotationsVec[jj].W[aa];
      }
    }

    // read image
    mylib::Array *img = mylib::Read_Image(((char *)imgFilenameOld.c_str()), 0);

    if (img == NULL) {
      cout << "ERROR: at mainCellDivisionClassifier: problem reading file "
           << imgFilenameOld << endl;
      return 5;
    }
    // hack to make the code work for uin8 without changing everything to
    // templates
    // basically, parse the image to uint16, since the code was designed for
    // uint16
    if (img->type == mylib::UINT8_TYPE) {
      img = mylib::Convert_Array_Inplace(img, img->kind, mylib::UINT16_TYPE, 16,
                                         0);
    }
    // hack to make the code work for 2D without changing everything to
    // templates
    // basically, add one black slice to the image (you should select conn3D = 4
    // or 8)
    if (img->ndims == 2) {
      mylib::Dimn_Type dimsAux[dimsImage];
      for (int ii = 0; ii < img->ndims; ii++) dimsAux[ii] = img->dims[ii];
      for (int ii = img->ndims; ii < dimsImage; ii++) dimsAux[ii] = 2;

      mylib::Array *imgAux =
          mylib::Make_Array(img->kind, img->type, dimsImage, dimsAux);
      memset(imgAux->data, 0, (imgAux->size) * sizeof(mylib::uint16));
      memcpy(imgAux->data, img->data, img->size * sizeof(mylib::uint16));

      mylib::Array *imgSwap = imgAux;
      img = imgAux;
      mylib::Free_Array(imgSwap);
    }
    if (img->type != mylib::UINT16_TYPE) {
      cout << "ERROR: at mainCellDivisionClassifier: code only takes UINT16 "
              "images for training so far"
           << endl;
      return 4;
    }

    mylib::uint16 *imgPtr = (mylib::uint16 *)(img->data);
    long long int *dimsVec = new long long int[img->ndims];
    for (int aa = 0; aa < img->ndims; aa++) dimsVec[aa] = img->dims[aa];

    // calculate features for this image using all possible 8 combinations of
    // eigenvectors directions to artifically increment the traning data
    long long int offsetAux =
        (((long long int)numFeatures) * ((long long int)numEllipsoids));
    for (int symmetry = 0; symmetry < symmetryN; symmetry++) {
      basicEllipticalHaarFeatureVector **fBasic =
          calculateEllipticalHaarFeatures(m, W, numEllipsoids, imgPtr, dimsVec,
                                          devCUDA, symmetry);
      if (fBasic == NULL) return 1;  // some error happened

      // extend Haar features
      int numHaarFeaturesPerEllipsoid = 0;
      float *fVec = &(xTrain[offsetXtrain]);
      calculateCombinationsOfBasicHaarFeatures(
          fBasic, numEllipsoids, &numHaarFeaturesPerEllipsoid, &fVec);
      if (fVec == NULL) return 2;  // some error happened
      if (numHaarFeaturesPerEllipsoid != numFeatures) {
        cout << "ERROR: numFeatures " << numFeatures
             << " is different than numHaarFeaturesPerEllipsoid "
             << numHaarFeaturesPerEllipsoid << endl;
        return 3;
      }

      offsetXtrain += offsetAux;
      // release mmemory
      for (int ii = 0; ii < numEllipsoids; ii++) delete fBasic[ii];
      delete[] fBasic;
    }

    // release memory
    delete[] W;
    delete[] m;
    mylib::Free_Array(img);
    delete[] dimsVec;

    // start new image
    if (ii >= n) break;
    imgFilenameOld = annotationsVec[ii].imgFilename;
    ii--;
  }

  cout << "Finished calculating features for " << numSamples
       << " annotations in " << toc(&tt) << " secs" << endl;

  //---------------clean possible outliers in the training
  // set----------------------
  int numOuliers = cleanTrainingSetTranspose(
      xTrain, yTrain, &numSamples, &nPos, &nNeg, (long long int)(numFeatures));
  cout << "Finished cleaning possible outliers in " << toc(&tt)
       << " secs.Detected and removed " << numOuliers << " outliers" << endl;

  //---------------perform cross-validation if indicated by
  // user-------------------
  cout << "Separating data into training and testing for cross-validation with "
          "% "
       << crossValidationPercentage << endl;

// initialize random seed for random shuffle of data
#if defined(_WIN32) || defined(_WIN64)
  srand(unsigned(GetTickCount()));
#else
  srand(unsigned(time(NULL)));
#endif

  if (nPos + nNeg != numSamples) {
    cout << "ERROR: at cross-validation data generation: nPos + nNeg != "
            "numSamples"
         << endl;
    return 7;
  }

  feature *xTrain_train = NULL;
  feature *xTrain_test = NULL;
  feature *yTrain_train = NULL;
  feature *yTrain_test = NULL;
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

    xTrain_test = new feature[numSamples_test * ((long long int)numFeatures)];
    yTrain_test = new feature[numSamples_test * ((long long int)numFeatures)];

    // generate test and training data
    long long int offset = 0;
    long long int auxIdx;
    for (size_t ii = 0; ii < nIdx_test.size(); ii++) {
      auxIdx = nIdx_test[ii];
      yTrain_test[ii] = yTrain[auxIdx];
      memcpy(&(xTrain_test[offset]),
             &(xTrain[auxIdx * ((long long int)numFeatures)]),
             sizeof(feature) * numFeatures);
      offset += ((long long int)numFeatures);
    }
    // training data kust reorders xTrain (this is why we wanted sort also)
    offset = 0;
    xTrain_train = xTrain;
    yTrain_train = yTrain;
    for (size_t ii = 0; ii < nIdx_train.size(); ii++) {
      auxIdx = nIdx_train[ii];
      if (ii != auxIdx)  // we need to copy memory
      {
        yTrain_train[ii] = yTrain[auxIdx];
        memcpy(&(xTrain_train[offset]),
               &(xTrain[auxIdx * ((long long int)numFeatures)]),
               sizeof(feature) * numFeatures);
      }
      offset += ((long long int)numFeatures);
    }

    cout << "Number of training samples =" << nIdx_train.size()
         << "; number of test samples=" << nIdx_test.size() << endl;

  } else {
    xTrain_train = xTrain;
    yTrain_train = yTrain;
    numSamples_train = numSamples;
    numSamples_test = 0;
    nNeg_train = nNeg;
    nPos_train = nPos;
  }

  cout << "Finished separating data for cross-validation in " << toc(&tt)
       << " secs." << endl;
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

  // add weights (they are normalized inside teh training code)
  feature *wTrain = new feature[numSamples_train];
  feature negPosRatio = ((float)nNeg_train) / ((float)nPos_train);
  for (long long int ii = 0; ii < numSamples_train; ii++) {
    if (yTrain[ii] < 0)
      wTrain[ii] = 1.0;
    else
      wTrain[ii] = negPosRatio;
  }
  gentleTreeBoost(xTrain_train, yTrain_train, wTrain, numWeakClassifiers,
                  classifier, numSamples_train, numFeatures, J);

  delete[] wTrain;
  cout << "Finished training in " << toc(&tt) << " secs" << endl;

  //----------------test cross-validation if indicated by
  // user--------------------

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
  feature *Fx = NULL;
  if (numSamples_test > 0) {
    Fx = new feature[numSamples_test];
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
    Fx = new feature[numSamples_train];
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
  cout << "Finished testing in " << toc(&tt) << " secs" << endl;
  //----------------save classifier-------------------------------------
  string classifierFilename(classifierOutputFileBasename + itoa + itoaDate +
                            ".txt");
  cout << "Saving classifier in file " << classifierFilename << endl;

  int err = saveClassifier(classifier, classifierFilename);
  if (err > 0) return err;

  if (numSamples_test > 0) {
    feature *FxTest = new feature[numSamples_test];
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
  annotationsVec.clear();

  if (xTrain_train != xTrain && xTrain_train != NULL) delete[] xTrain_train;
  if (yTrain_train != yTrain && yTrain_train != NULL) delete[] yTrain_train;
  if (xTrain_test != NULL) delete[] xTrain_test;
  if (yTrain_test != NULL) delete[] yTrain_test;

  delete[] xTrain;
  delete[] yTrain;
  if (Fx != NULL) delete[] Fx;

  return 0;
}
