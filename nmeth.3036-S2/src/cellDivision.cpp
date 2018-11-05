/*
 *
 * \brief Contains routines dealing with cell division for lineage hypertree
 *
 *
 */

#include "cellDivision.h"
#include <iostream>
#include "UtilsCUDA/3DEllipticalHaarFeatures/AnnotationEllipsoid.h"
#include "UtilsCUDA/3DEllipticalHaarFeatures/gentleBoost/gentleBoost.h"
#include "external/Nathan/tictoc.h"
#include "temporalLogicalRules/utilsAmatf.h"
#include "variationalInference.h"

using namespace std;

//=============================================================================================================
int cellDivisionMinFlowTouchingSupervoxels(lineageHyperTree& lht, int TM,
                                           int conn3D,
                                           size_t minNeighboringVoxels,
                                           int& numCellDivisions,
                                           int& numBirths) {
  numCellDivisions = 0;
  numBirths = 0;

  if (TM >= lht.getMaxTM()) return 0;

  int64 boundarySize[dimsImage];
  int64* neighOffset =
      supervoxel::buildNeighboorhoodConnectivity(conn3D, boundarySize);

  vector<uint64> PixelIdxListD, PixelIdxListM;
  PixelIdxListD.reserve(300);
  PixelIdxListM.reserve(300);
  // check all the nuclei in this time point
  for (list<nucleus>::iterator iterN = lht.nucleiList[TM].begin();
       iterN != lht.nucleiList[TM].end(); ++iterN) {
    if (iterN->treeNode.getNumChildren() < 2)
      continue;  // no possibility of splitting because we only hav eone
                 // supervoxel in this nucleus
    if (iterN->treeNodePtr->parent != NULL &&
        iterN->treeNodePtr->parent->getNumChildren() >
            1)  // this is already part of a division
      continue;

    vector<ChildrenTypeNucleus> vecS_M = iterN->treeNode.getChildren();

    // determine connectivity graph between supervoxels
    vector<vector<size_t> > graphNodes(
        vecS_M.size());  // it will store indexes to other nodes
    for (size_t ii = 0; ii < vecS_M.size(); ii++) {
      graphNodes[ii].push_back(ii);  // every node is connected ot itself
      for (size_t jj = ii + 1; jj < vecS_M.size(); jj++) {
        vecS_M[ii]->neighboringVoxels(*(vecS_M[jj]), conn3D, neighOffset,
                                      PixelIdxListD, PixelIdxListM);
        if (PixelIdxListD.size() >= minNeighboringVoxels)  // we consider these
                                                           // two supervoxels
                                                           // connected
        {
          graphNodes[ii].push_back(jj);
          graphNodes[jj].push_back(ii);
        }
      }
    }

    // find connected components within graph
    vector<vector<ChildrenTypeNucleus> >
        vecS_D;  // vecS_D[i] contains the i-th group of connected supervoxels
    vecS_D.reserve(vecS_M.size());
    vector<bool> visited(vecS_M.size(), false);
    int numCC = 0;
    for (size_t ii = 0; ii < vecS_M.size(); ii++) {
      if (visited[ii] ==
          false)  // this node has not been visited->new connected component
      {
        vecS_D.resize(numCC + 1);
        queue<int> q;
        q.push(ii);
        while (q.empty() == false)  // recursive
        {
          int kk = q.front();
          q.pop();
          for (size_t jj = 0; jj < graphNodes[kk].size(); jj++) {
            int idx = graphNodes[kk][jj];
            if (visited[idx] == false) {
              vecS_D[numCC].push_back(vecS_M[idx]);
              visited[idx] = true;
              q.push(idx);
            }
          }
        }
        numCC++;
      }
    }

    // decide what to do depending on the number of disconnected groups
    // if numCC == 1->nothing to do
    // if numCC == 2->easy: straight forward cell division into two groups
    // if numCC > 2 ->not so easy

    vector<int> birthTrackId;  // when numCC > 2->contains index of group in
                               // vecS_D that will start a new track
    int daughter2Id = -1;      // when numCC > 1 contains the id of the group in
                               // vecS_D that will contain the second daughter
    // by omision themissing id is the one untocuhed that continues teh lineage
    if (numCC == 2) {
      daughter2Id =
          1;  // omitted group id is 0 and obvious 2nd daughter is index 1;
    } else if (numCC > 2) {
      int daughter1Id = -1;
      if (iterN->treeNodePtr->parent == NULL)  // we cannnot calculate
                                               // probability of divisions->all
                                               // elements should start a new
                                               // track
      {
        daughter1Id = 0;  // so the current lineage continues
      } else {
        // recalculate centroid for each disconnected group of supervoxels
        vector<float> centroidArray(dimsImage * numCC, 0);
        vector<float> weightsArray(numCC, 0);
        float wAux;
        assert(supervoxel::getDataType() == 8);
        float* imgPtr = (float*)(vecS_D[0][0]->dataPtr);
        for (int ii = 0; ii < numCC; ii++) {
          for (size_t jj = 0; jj < vecS_D[ii].size(); jj++) {
            wAux = 0.0f;
            for (vector<uint64>::const_iterator iter =
                     vecS_D[ii][jj]->PixelIdxList.begin();
                 iter != vecS_D[ii][jj]->PixelIdxList.end(); ++iter) {
              wAux += imgPtr[*iter];
            }
            weightsArray[ii] += wAux;
            for (int kk = 0; kk < dimsImage; kk++) {
              centroidArray[ii * dimsImage + kk] +=
                  wAux * vecS_D[ii][jj]->centroid[kk];
            }
          }
          // calculate average centroid
          for (int kk = 0; kk < dimsImage; kk++) {
            centroidArray[ii * dimsImage + kk] /= weightsArray[ii];
          }
        }

        // decide which two gropups perform a cell division and which are left
        // as new tracks
        float maxProb = -1.0f;  // so at least one is assigned (even if it is 0)

        for (int ii = 0; ii < numCC; ii++) {
          float* centroidCh1 = &(centroidArray[dimsImage * ii]);
          for (int jj = ii + 1; jj < numCC; jj++) {
            // float prob =
            // MahalanobisDistanceMotherAlongDaughtersAxis(iterN->treeNodePtr->parent->data->centroid,
            // centroidCh1, &(centroidArray[dimsImage * jj]),
            // supervoxel::getScale());
            float prob =
                1.0 /
                (DistanceCellDivisionPlane(
                     iterN->treeNodePtr->parent->data->centroid, centroidCh1,
                     &(centroidArray[dimsImage * jj]), supervoxel::getScale()) +
                 1e-4);
            if (prob > maxProb) {
              daughter1Id = ii;
              daughter2Id = jj;
              maxProb = prob;
            }
          }
        }
      }
      // all the groups that are not daughters generate new tracks
      for (int ii = 0; ii < (int)(vecS_D.size()); ii++) {
        if (ii != daughter1Id && ii != daughter2Id) birthTrackId.push_back(ii);
      }
    }

    if (daughter2Id >= 0)  // there is a cell division
    {
      numCellDivisions++;

      // remove link from supervoxel to nuclei
      iterN->treeNode.deleteChild(vecS_D[daughter2Id][0]);
      vecS_D[daughter2Id][0]->treeNode.deleteParent();

      // create new nuclei
      ParentTypeSupervoxel iterN_D2 = lht.addNucleusFromSupervoxel(
          TM, vecS_D[daughter2Id]
                    [0]);  // returns pointer to the newly created nucleus

      if (iterN->treeNodePtr->parent !=
          NULL)  // it is not the beginning of a lineage
      {
        iterN_D2->treeNode.setParent(iterN->treeNode.getParent());
        iterN->treeNode.getParent()->bt.SetCurrent(iterN->treeNodePtr->parent);
        iterN_D2->treeNodePtr =
            iterN->treeNode.getParent()->bt.insert(iterN_D2);
        if (iterN_D2->treeNodePtr == NULL) return 3;
      } else {  // we have to create a new lineage
        lht.lineagesList.push_back(lineage());
        list<lineage>::iterator listLineageIter =
            ((++(lht.lineagesList.rbegin()))
                 .base());  // iterator for the last element in the list

        iterN_D2->treeNode.setParent(listLineageIter);
        iterN_D2->treeNodePtr = listLineageIter->bt.insert(iterN_D2);
        if (iterN_D2->treeNodePtr == NULL) return 3;
      }
      // add the rest of supervoxels to the new nucleus
      for (size_t ii = 1; ii < vecS_D[daughter2Id].size(); ii++) {
        iterN->treeNode.deleteChild(vecS_D[daughter2Id][ii]);
        iterN_D2->treeNode.addChild(vecS_D[daughter2Id][ii]);
        vecS_D[daughter2Id][ii]->treeNode.setParent(iterN_D2);
      }

      // recalculate centroid for nucleis
      lht.calculateNucleiIntensityCentroid<float>(iterN);
      lht.calculateNucleiIntensityCentroid<float>(iterN_D2);
    }
    // generate birth for all the other tracks
    for (size_t kk = 0; kk < birthTrackId.size(); kk++) {
      daughter2Id = birthTrackId[kk];  // to reuse code

      // For parhyale 13_12_30, sometimes a birth wants to take all the sv left
      // from a main nucleus. This is a patch to fix that issue.
      // TODO: find bug. Why do we want to take all
      if (iterN->treeNode.getNumChildren() == vecS_D[daughter2Id].size()) {
        continue;
      }

      numBirths++;
      // remove link from supervoxel to nuclei
      iterN->treeNode.deleteChild(vecS_D[daughter2Id][0]);
      vecS_D[daughter2Id][0]->treeNode.deleteParent();

      // create new nuclei
      ParentTypeSupervoxel iterN_D2 = lht.addNucleusFromSupervoxel(
          TM, vecS_D[daughter2Id]
                    [0]);  // returns pointer to the newly created nucleus

      // we have to create a new lineage
      lht.lineagesList.push_back(lineage());
      list<lineage>::iterator listLineageIter =
          ((++(lht.lineagesList.rbegin()))
               .base());  // iterator for the last element in the list

      iterN_D2->treeNode.setParent(listLineageIter);
      iterN_D2->treeNodePtr = listLineageIter->bt.insert(iterN_D2);
      if (iterN_D2->treeNodePtr == NULL) return 3;

      // add the rest of supervoxels to the new nucleus
      for (size_t ii = 1; ii < vecS_D[daughter2Id].size(); ii++) {
        iterN->treeNode.deleteChild(vecS_D[daughter2Id][ii]);
        iterN_D2->treeNode.addChild(vecS_D[daughter2Id][ii]);
        vecS_D[daughter2Id][ii]->treeNode.setParent(iterN_D2);
      }

      // recalculate centroid for nucleis
      lht.calculateNucleiIntensityCentroid<float>(iterN);
      lht.calculateNucleiIntensityCentroid<float>(iterN_D2);
    }
  }
  return 0;
}

//=============================================================================================================
int cellDivisionMinFlowTouchingSupervoxels(lineageHyperTree& lht, int TM,
                                           int conn3D,
                                           size_t minNeighboringVoxels,
                                           int& numCellDivisions) {
  numCellDivisions = 0;

  if (TM >= lht.getMaxTM()) return 0;

  int64 boundarySize[dimsImage];
  int64* neighOffset =
      supervoxel::buildNeighboorhoodConnectivity(conn3D, boundarySize);

  // very simple right now: just check if super voxels belonging to the same
  // nucleus are "touching" or not
  vector<uint64> PixelIdxListD, PixelIdxListM;
  PixelIdxListD.reserve(300);
  PixelIdxListM.reserve(300);
  for (list<nucleus>::iterator iterN = lht.nucleiList[TM].begin();
       iterN != lht.nucleiList[TM].end(); ++iterN) {
    if (iterN->treeNode.getNumChildren() < 2)
      continue;  // no possibility of splitting
    if (iterN->treeNodePtr->parent != NULL &&
        iterN->treeNodePtr->parent->getNumChildren() >
            1)  // this is already coming from a division
      continue;

    vector<ChildrenTypeNucleus> vecS_M = iterN->treeNode.getChildren();
    vector<ChildrenTypeNucleus> vecS_D1(
        1, vecS_M[0]);  // contains all the supervoxels for first daughter
    vector<ChildrenTypeNucleus>
        vecS_D2;  // contains all the supervoxels for second daughter
    // we scan all the elements one by one and assign them to the best group. If
    // at the end vecS_D2.empty() == true -> no division
    for (size_t ii = 1; ii < vecS_M.size(); ii++) {
      size_t numNeigh = 0;
      for (size_t jj = 0; jj < vecS_D1.size(); jj++) {
        vecS_D1[jj]->neighboringVoxels(*(vecS_M[ii]), conn3D, neighOffset,
                                       PixelIdxListD, PixelIdxListM);
        numNeigh += PixelIdxListD.size();
      }
      // check to which daughter it belongs
      if (numNeigh >= minNeighboringVoxels)  // attach to daughter 1
      {
        vecS_D1.push_back(vecS_M[ii]);
      } else {  // check if it is better to attach to daughter 2
        if (vecS_D2.empty() == true)
          vecS_D2.push_back(vecS_M[ii]);
        else {
          size_t numNeigh2 = 0;
          for (size_t jj = 0; jj < vecS_D2.size(); jj++) {
            vecS_D2[jj]->neighboringVoxels(*(vecS_M[ii]), conn3D, neighOffset,
                                           PixelIdxListD, PixelIdxListM);
            numNeigh2 += PixelIdxListD.size();
          }

          if (numNeigh2 > numNeigh)
            vecS_D2.push_back(vecS_M[ii]);
          else  // in case of tie it goes to the daughter 1 (arbitrary decision)
            vecS_D1.push_back(vecS_M[ii]);
        }
      }
    }

    // apply cell division to nuclei if necessary
    if (vecS_D2.empty() == false) {
      numCellDivisions++;

      // remove link from supervoxel to nuclei
      iterN->treeNode.deleteChild(vecS_D2[0]);
      vecS_D2[0]->treeNode.deleteParent();

      // create new nuclei
      ParentTypeSupervoxel iterN_D2 = lht.addNucleusFromSupervoxel(
          TM, vecS_D2[0]);  // returns pointer to the newly created nucleus

      if (iterN->treeNodePtr->parent !=
          NULL)  // it is not the beginning of a lineage
      {
        iterN_D2->treeNode.setParent(iterN->treeNode.getParent());
        iterN->treeNode.getParent()->bt.SetCurrent(iterN->treeNodePtr->parent);
        iterN_D2->treeNodePtr =
            iterN->treeNode.getParent()->bt.insert(iterN_D2);
        if (iterN_D2->treeNodePtr == NULL) return 3;
      } else {  // we have to create a new lineage
        lht.lineagesList.push_back(lineage());
        list<lineage>::iterator listLineageIter =
            ((++(lht.lineagesList.rbegin()))
                 .base());  // iterator for the last element in the list

        iterN_D2->treeNode.setParent(listLineageIter);
        iterN_D2->treeNodePtr = listLineageIter->bt.insert(iterN_D2);
        if (iterN_D2->treeNodePtr == NULL) return 3;
      }
      // add the rest of supervoxels to the new nucleus
      for (size_t ii = 1; ii < vecS_D2.size(); ii++) {
        iterN->treeNode.deleteChild(vecS_D2[ii]);
        iterN_D2->treeNode.addChild(vecS_D2[ii]);
        vecS_D2[ii]->treeNode.setParent(iterN_D2);
      }

      // recalculate centroid for nucleis
      lht.calculateNucleiIntensityCentroid<float>(iterN);
      lht.calculateNucleiIntensityCentroid<float>(iterN_D2);
    }
  }
  return 0;
}

//======================================================================
int testKullbackLeiblerMethod() {
  string annotationsFilenameXML(
      "C:/Users/Fernando/TrackingNuclei/matlabCode/visualization/annotations/"
      "finalsCellDivisionClassifierOriginalImageBased/"
      "classifierAnnotations_2011_2013_cat.xml2");
  string classifierOutputFileBasename(
      "C:/Users/Fernando/TrackingNuclei/matlabCode/cellDivisionModel/"
      "classifiers/KLmethod");
  float scale[dimsImage] = {1.0f, 1.0f, 5.0f};
  //----------------read annotations-----------------------------------------

  XMLNode xMainNode =
      XMLNode::openFileHelper(annotationsFilenameXML.c_str(), "document");
  int n = xMainNode.nChildNode("Surface");

  vector<AnnotationEllipsoid> annotationsVec(n);
  long long int nPos = 0, nNeg = 0;
  for (int ii = 0; ii < n; ii++) {
    annotationsVec[ii] = AnnotationEllipsoid(xMainNode, ii);
    // parse class name to labels
    if (annotationsVec[ii].className.compare("twocell") == 0)  // cell division
    {
      annotationsVec[ii].classVal = 1;
      nPos++;
    } else {
      annotationsVec[ii].classVal = 0;
      nNeg++;
    }
  }

  cout << "Read " << n << " annotations. Positives = " << nPos
       << "; negatives=" << nNeg << endl;

  //----------------calculate features-------------------------------------

  // sort vector of features by image filename
  sort(annotationsVec.begin(), annotationsVec.end());
  int sizeW = dimsImage * (1 + dimsImage) / 2;

  // preallocate memory for training
  long long int numSamples = ((long long int)n);
  feature* yTrain = new feature[numSamples];
  feature* Fx = new feature[numSamples];  // keeps score for KL method

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
    int yTrainOffset = ii;
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

    cout << "Calculating features for " << numEllipsoids
         << " annotations in image " << imgFilenameOld << endl;

    // allocate memory and parse annotations into GaussianMixxtureModelFormat
    vector<GaussianMixtureModel*> vecGM(numEllipsoids);
    int pos = 0;
    for (int jj = offset; jj < ii; jj++, pos++) {
      GaussianMixtureModel* auxGM = new GaussianMixtureModel(pos, scale);

      auxGM->nu_k = 1.0;
      for (int aa = 0; aa < dimsImage; aa++) {
        auxGM->m_k(aa) = annotationsVec[jj].mu[aa];
      }
      int countW = 0;
      for (int aa = 0; aa < dimsImage; aa++) {
        for (int bb = aa; bb < dimsImage; bb++) {
          auxGM->W_k(aa, bb) = annotationsVec[jj].W[countW];
          auxGM->W_k(bb, aa) = annotationsVec[jj].W[countW];
          countW++;
        }
      }

      vecGM[pos] = auxGM;
    }

    // read image
    mylib::Array* img = mylib::Read_Image(((char*)imgFilenameOld.c_str()), 0);

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

      mylib::Array* imgAux =
          mylib::Make_Array(img->kind, img->type, dimsImage, dimsAux);
      memset(imgAux->data, 0, (imgAux->size) * sizeof(mylib::uint16));
      memcpy(imgAux->data, img->data, img->size * sizeof(mylib::uint16));

      mylib::Array* imgSwap = imgAux;
      img = imgAux;
      mylib::Free_Array(imgSwap);
    }
    if (img->type != mylib::UINT16_TYPE) {
      cout << "ERROR: at mainCellDivisionClassifier: code only takes UINT16 "
              "images for training so far"
           << endl;
      return 4;
    }

    mylib::Convert_Array_Inplace(img, mylib::PLAIN_KIND, mylib::FLOAT32_TYPE, 1,
                                 1.0);
    mylib::Value v1a, v0a;
    v1a.fval = 1.0;
    v0a.fval = 0.0;
    mylib::Scale_Array_To_Range(img, v0a, v1a);

    mylib::float32* imgPtr = (mylib::float32*)(img->data);

    // setup responsibilities (in this case we use binary assignment: each voxel
    // belongs to the most likely Gaussian)
    responsibilities r(img->size, vecGM.size(), img->size);
    r.R_nk->nz = 0;

    size_t posIm = 0;
    double Zp;
    Matrix<double, dimsImage, 1> s;
    for (size_t zz = 0; zz < img->dims[2]; zz++) {
      s(2) = zz;
      for (size_t yy = 0; yy < img->dims[1]; yy++) {
        s(1) = yy;
        for (size_t xx = 0; xx < img->dims[0]; xx++) {
          s(0) = xx;
          // calculate to which Gaussian we assign this voxel
          double prob = 0, probAux;
          int k = -1;
          for (size_t aa = 0; aa < vecGM.size(); aa++) {
            Zp = pow(6.28318530717958623200, dimsImage / 2.0) /
                 pow(vecGM[aa]->W_k.determinant(), 0.5);
            probAux = exp(-0.5 * (((s - vecGM[aa]->m_k).transpose() *
                                   vecGM[aa]->W_k * (s - vecGM[aa]->m_k))(0))) /
                      Zp;
            if (probAux > prob) k = aa;
          }

          if (k >= 0) {
            r.R_nk->i[r.R_nk->nz] = posIm;
            r.R_nk->p[r.R_nk->nz] = k;
            r.R_nk->x[r.R_nk->nz] = 1.0;
            r.R_nk->nz++;
          }
          posIm++;
        }
      }
    }

    calculateLocalKullbackDiversity(img, vecGM, r);

    // copy results
    for (int aa = 0; aa < numEllipsoids; aa++) {
      Fx[yTrainOffsetAnchor + aa] = vecGM[aa]->splitScore;
    }

    // release memory
    mylib::Free_Array(img);
    for (size_t jj = 0; jj < vecGM.size(); jj++) delete vecGM[jj];
    vecGM.clear();

    // start new image
    if (ii >= n) break;
    imgFilenameOld = annotationsVec[ii].imgFilename;
    ii--;
  }

  cout << "Finished calculating features " << endl;

  // calculate precision-recall
  char buffer[256];
  sprintf(buffer, "_numSamples%lld", numSamples);
  string itoa(buffer);
#if defined(_WIN32) || defined(_WIN64)
  SYSTEMTIME str_t;
  GetSystemTime(&str_t);
  char extra[256];
  sprintf(extra, "_%d_%d_%d_%d_%d_%d", str_t.wYear, str_t.wMonth, str_t.wDay,
          str_t.wHour, str_t.wMinute, str_t.wSecond);
  string itoaDate(extra);
#else
  char extra[256];
  sprintf(extra, "_%ld", time(NULL));
  string itoaDate(extra);
#endif
  string ROCfilename(classifierOutputFileBasename + "_ROC" + itoa + itoaDate +
                     ".txt");
  ofstream out(ROCfilename.c_str());
  if (!out.is_open()) {
    cout << "ERROR: opening file " << ROCfilename << " to save ROC curve"
         << endl;
  }
  precisionRecallAccuracyCurve(yTrain, Fx, numSamples, out);
  out.close();

  // release memory
  delete[] yTrain;
  delete[] Fx;

  return 0;
}

//=========================================================================================================================
int backgroundCheckWithClassifierShortLineages(lineageHyperTree& lht, int TM,
                                               int maxLengthBackgroundCheck,
                                               const imageType* imgPtr,
                                               const long long int* imgDims) {
  if (TM >= lht.getMaxTM()) return 0;

  TicTocTimer tt = tic();

  vector<list<supervoxel>::iterator>
      vecSiter;  // to store which one we are calculating features from
  vecSiter.reserve(lht.supervoxelsList[TM].size() / 2);

  TreeNode<ChildrenTypeLineage>* aux;
  for (list<supervoxel>::iterator iterS = lht.supervoxelsList[TM].begin();
       iterS != lht.supervoxelsList[TM].end(); ++iterS) {
    // check if it is a short lineage
    if (iterS->treeNode.hasParent() == true) {
      aux = iterS->treeNode.getParent()->treeNodePtr->parent;
      int ll = 1;
      while (aux != NULL && ll <= maxLengthBackgroundCheck) {
        aux = aux->parent;
        ll++;
      }

      if (ll <= maxLengthBackgroundCheck)
        vecSiter.push_back(iterS);
      else
        iterS->probClassifier = -1e32f;  // we set as if it was not background
                                         // (we'll have to refine this later)
    }
  }

  if (vecSiter.empty()) return 0;  // nothing to do

  // calculate features for the select items
  // load cell background classifier
  string classifierCellBackgroundFilename(
      "C:/Users/Fernando/cppProjects/TrackingGaussianMixtures/trunk/build/"
      "Release/classifierCellBackground.txt");
  vector<vector<treeStump> > classifierCellBackground;
  int errC = loadClassifier(classifierCellBackground,
                            classifierCellBackgroundFilename);
  if (errC > 0) return errC;

  long long int numFeatures = getNumberOfHaarFeaturesPerllipsoid();

  // parse multivaroate Gaussian to calculate features
  int numEllipsoids = vecSiter.size();
  int sizeW = dimsImage * (1 + dimsImage) / 2;
  double* m = new double[dimsImage * numEllipsoids];
  double* W = new double[sizeW * numEllipsoids];
  double Waux[dimsImage * (1 + dimsImage) / 2];
  double mAux[dimsImage];
  float I;
  for (int jj = 0; jj < numEllipsoids; jj++) {
    vecSiter[jj]->weightedGaussianStatistics<float>(
        mAux, Waux, &I, true);  // calculate equivalence

    int countW = jj;
    int countM = jj;
    int countWaux = 0;
    for (int aa = 0; aa < dimsImage; aa++) {
      m[countM] = mAux[aa];
      for (int bb = aa; bb < dimsImage; bb++) {
        W[countW] = Waux[countWaux];
        countW += numEllipsoids;
        countWaux++;
      }
      countM += numEllipsoids;
    }
  }

  // calculate features
  basicEllipticalHaarFeatureVector** fBasic = calculateEllipticalHaarFeatures(
      m, W, vecSiter.size(), imgPtr, imgDims, 0, 0);
  if (fBasic == NULL) return 1;  // some error happened

  // extend Haar features
  int numHaarFeaturesPerEllipsoid = 0;
  float* xTest = NULL;
  calculateCombinationsOfBasicHaarFeatures(
      fBasic, vecSiter.size(), &numHaarFeaturesPerEllipsoid, &xTest);
  if (xTest == NULL) return 2;  // some error happened
  if (numHaarFeaturesPerEllipsoid != numFeatures) {
    cout << "ERROR: numFeatures " << numFeatures
         << " is different than numHaarFeaturesPerEllipsoid "
         << numHaarFeaturesPerEllipsoid << endl;
    return 3;
  }

  // calculate classifier results
  vector<feature> Fx(vecSiter.size());
  boostingTreeClassifierTranspose(xTest, &(Fx[0]), classifierCellBackground,
                                  Fx.size(), numFeatures);

  // parse classifier results
  for (size_t ii = 0; ii < vecSiter.size(); ii++)
    vecSiter[ii]->probClassifier = Fx[ii];

  // release memory
  delete[] xTest;
  Fx.clear();
  for (int ii = 0; ii < numEllipsoids; ii++) {
    delete fBasic[ii];
  }
  delete[] fBasic;
  delete[] W;
  delete[] m;

  cout << "INFO: backgroundCheckWithClassifierShortLineages: it took "
       << toc(&tt) << " secs to calculate " << vecSiter.size()
       << " supervoxel background classifier out of "
       << lht.supervoxelsList[TM].size() << " total number of supervoxels"
       << endl;

  return 0;
}

float DistanceCellDivisionPlane(float centroidPar[dimsImage],
                                float centroidCh1[dimsImage],
                                float centroidCh2[dimsImage],
                                float scale[dimsImage]) {
  // calculate midplane feature
  float norm = 0.0f;
  float d = 0.0f;
  float p0, n, m;
  for (int ii = 0; ii < dimsImage; ii++) {
    p0 = 0.5 * (centroidCh1[ii] + centroidCh2[ii]);  // midpoint
    n = (centroidCh1[ii] - p0) * scale[ii];          // normal
    norm += (n * n);
    // calculate distance of mother cell to division plane
    m = (centroidPar[ii] - p0) * scale[ii];

    d += (n * m);
  }

  d = fabs(d) / sqrt(norm);  // midplane distance

  return d;
}
//======================================================================================================
// copy and paste from float
// temporalCost::MahalanobisDistanceMotherAlongDaughtersAxis(supervoxel* svPar,
// supervoxel* svCh1, supervoxel* svCh2)
float MahalanobisDistanceMotherAlongDaughtersAxis(float centroidPar[dimsImage],
                                                  float centroidCh1[dimsImage],
                                                  float centroidCh2[dimsImage],
                                                  float scale[dimsImage]) {
  // generate a Gaussian along the axis formed by the two daughters
  float v1[3], v3[3], v2[3], mu[3], d[3];
  float norm = 0.0f;
  for (int ii = 0; ii < 3; ii++) {
    v1[ii] = (centroidCh1[ii] - centroidCh2[ii]) * scale[ii];
    norm += v1[ii] * v1[ii];
    mu[ii] = 0.5f * (centroidCh1[ii] + centroidCh2[ii]) * scale[ii];
  }
  // define covariance along the principal axis
  const float K = 16.0f;
  d[0] = norm / K;  // if svPar->centroid is equal to one of the
                    // daughters->probability of split is exp(-K/ 8) [it should
                    // be very low]
  d[1] = d[0] / 2;  // along the perpendicular plane with respecty to division
                    // axis the Guassian is even tighter
  d[2] = d[1];

  // convert vector to norm 1
  norm = sqrt(norm);
  for (int ii = 0; ii < 3; ii++) v1[ii] /= norm;

  // find 2 orthonal vectors to the division axis
  norm = sqrt(v1[0] * v1[0] + v1[1] * v1[1]);
  if (norm > 1e-3)  // to avoid singular case
  {
    v2[0] = -v1[1] / norm;
    v2[1] = v1[0] / norm;
    v2[2] = 0.0f;
    v3[0] = -v1[0] * v1[2] / norm;
    v3[1] = -v1[1] * v1[2] / norm;
    v3[2] = norm;
  } else {  // we can assume v1 is very close to [0 0 1]
    v1[0] = 0.0f;
    v1[1] = 0.0f;
    v1[2] = 1.0f;
    v2[0] = 0.0f;
    v2[1] = 1.0;
    v2[2] = 0.0f;
    v3[0] = -1.0f;
    v3[1] = 0.0f;
    v3[2] = 0.0f;
  }

  // calculate inmverse covariance matrix directly
  for (int ii = 0; ii < 3; ii++) {
    d[ii] = 1.0f / d[ii];
    mu[ii] -= (centroidPar[ii] * scale[ii]);  // to compute probability later
  }
  float W[6];

  W[0] = v1[0] * v1[0] * d[0] + v2[0] * v2[0] * d[1] + v3[0] * v3[0] * d[2];
  W[1] = v1[0] * v1[1] * d[0] + v2[0] * v2[1] * d[1] + v3[0] * v3[1] * d[2];
  W[2] = v1[0] * v1[2] * d[0] + v2[0] * v2[2] * d[1] + v3[0] * v3[2] * d[2];
  W[3] = v1[1] * v1[1] * d[0] + v2[1] * v2[1] * d[1] + v3[1] * v3[1] * d[2];
  W[4] = v1[1] * v1[2] * d[0] + v2[1] * v2[2] * d[1] + v3[1] * v3[2] * d[2];
  W[5] = v1[2] * v1[2] * d[0] + v2[2] * v2[2] * d[1] + v3[2] * v3[2] * d[2];

  // calculate probability [in our case, between [0,1], so it is not exactly a
  // perfect Gaussian PDF, just an exponential decay]
  float prob1 = utilsAmatf_MahalanobisDistance_3D(mu, W);
  if (prob1 > 12)
    prob1 = 0.0f;  // exp(-0.5 * 12 ) = 0.0025
  else
    prob1 = exp(-0.5 * prob1);

  //=============================================================
  // feature based on distance between mother and daughters
  float norm1 = 0.0f, norm2 = 0.0f;
  float aux;
  for (int ii = 0; ii < 3; ii++) {
    aux = (centroidCh1[ii] - centroidPar[ii]) * scale[ii];
    norm1 += aux * aux;
    aux = (centroidCh2[ii] - centroidPar[ii]) * scale[ii];
    norm2 += aux * aux;
  }
  float sigmaDist = 15 * 15;  // in pixels
  float prob2 = exp(-0.5 * norm1 / sigmaDist) + exp(-0.5 * norm2 / sigmaDist);

  //==============================================================
  // final probabilitiy: weighted sum
  float prob = 0.5 * prob1 + 0.5 * prob2;
  return prob;
}
