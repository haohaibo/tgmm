/*
 * Copyright (C) 2011-2013 by  Fernando Amat
 * See license.txt for full license and copyright notice.
 *
 * Authors: Fernando Amat
 *  backgroundClassifier.cpp
 *
 *  Created on: January 21st, 2013
 *      Author: Fernando Amat
 *
 * \brief features (temporal and geometrical and spatial) to distinguish
 * background lineages
 *
 *
 */

#include "backgroundClassifier.h"
#include <assert.h>
#include <iostream>
#include "utilsAmatf.h"

using namespace std;

int backgroundDetectionFeatures::teporalWindowSize;

backgroundDetectionFeatures::backgroundDetectionFeatures(){
    // teporalWindowSize = 0;//you have to setup temporal windiw size yourself
    // since it is a static variable
};

backgroundDetectionFeatures::backgroundDetectionFeatures(
    int teporalWindowSize_) {
  teporalWindowSize = teporalWindowSize_;
};

backgroundDetectionFeatures::~backgroundDetectionFeatures(){};

//========================================================
int backgroundDetectionFeatures::calculateFeatures(
    TreeNode<ChildrenTypeLineage>* root, int devCUDA) {
  const int numKNNdist = 6;  // to calculate ratios between nearest neighbors

  f.resize(getNumFeatures());

  if (root == NULL ||
      root->getNumChildren() !=
          1)  // cell divisions or dying cells are excluded
  {
    memset(&(f[0]), 0, sizeof(float) * f.size());
    return 2;  // to indicate this should not be added to the set of features
  }

  int numFeatures = 0;

  queue<TreeNode<ChildrenTypeLineage>*> q;
  q.push(root);
  TreeNode<ChildrenTypeLineage>* auxNode;

  int iniTM = root->data->TM;
  // find longest branch (it is were we will calculate SPATIAO-GEOMETRIC stats)
  int lengthBranchMax = 0;
  TreeNode<ChildrenTypeLineage>* rootLongestBranch = NULL;
  while (q.empty() == false) {
    auxNode = q.front();
    q.pop();
    TreeNode<ChildrenTypeLineage>* rootBranch = auxNode;
    int lengthBranch = 1;

    while (auxNode != NULL) {
      if (auxNode->data->TM - iniTM + 1 >
          teporalWindowSize)  // we reached the end of the window size
      {
        break;
      }

      switch (auxNode->getNumChildren()) {
        case 0:  // cell death
          auxNode = NULL;
          break;

        case 1:            // cell displacement
          lengthBranch++;  // continue branch
          if (auxNode->left != NULL)
            auxNode = auxNode->left;
          else
            auxNode = auxNode->right;
          break;

        case 2:                   // cell division
          q.push(auxNode->left);  // start new branches
          q.push(auxNode->right);

          auxNode = NULL;  // branch termination
          break;

        default:
          cout << "ERROR: we should never be here!" << endl;
          exit(3);
      }
    }
    if (lengthBranch > lengthBranchMax) {
      lengthBranchMax = lengthBranch;
      rootLongestBranch = rootBranch;
    }
  }

  // check if we have enough length
  if (lengthBranchMax < 3) return 3;  // not possible to calculate stats

  q.push(rootLongestBranch);  // initial point

  // prellocate memory for different stats
  vector<float> nucSize;  // difference in size
  nucSize.reserve(teporalWindowSize);
  vector<float> nucOffsetJaccard;  // Jaccard index after subtracting
                                   // translation between nuclei
  nucOffsetJaccard.reserve(teporalWindowSize);
  vector<float> numEmptyVoxels;  // number of empty voxels inide nuclei
  numEmptyVoxels.reserve(teporalWindowSize);
  vector<float> XYZ[dimsImage];  // to fit straihgt line
  for (int ii = 0; ii < dimsImage; ii++) XYZ[ii].reserve(teporalWindowSize);

  vector<float> KNNdist[numKNNdist];
  for (int ii = 0; ii < numKNNdist; ii++)
    KNNdist[ii].reserve(teporalWindowSize);
  vector<ChildrenTypeLineage> iterNucleusNNvec(numKNNdist);
  vector<float> distVec(numKNNdist);

  while (q.empty() == false) {
    auxNode = q.front();
    q.pop();

    if (auxNode->getNumChildren() == 1 &&
        (auxNode->data->TM - iniTM < teporalWindowSize))  // we only keep going
                                                          // if there are no
                                                          // cell divisions
    {
      if (auxNode->left != NULL)
        q.push(auxNode->left);
      else
        q.push(auxNode->right);
    }

    // location
    for (int ii = 0; ii < dimsImage; ii++) {
      XYZ[ii].push_back(auxNode->data->centroid[ii]);
    }

    // size
    size_t ss = 0;
    for (vector<ChildrenTypeNucleus>::iterator iterS =
             auxNode->data->treeNode.getChildren().begin();
         iterS != auxNode->data->treeNode.getChildren().end(); ++iterS)
      ss += (*iterS)->PixelIdxList.size();

    nucSize.push_back(ss);

    // shape
    if (auxNode->parent != NULL)
      nucOffsetJaccard.push_back(
          lineageHyperTree::offsetJaccardDistance(auxNode));

    // number of empty voxels
    ss = 0;
    for (vector<ChildrenTypeNucleus>::iterator iterS =
             auxNode->data->treeNode.getChildren().begin();
         iterS != auxNode->data->treeNode.getChildren().end(); ++iterS)
      ss += (*iterS)->numHoles();
    numEmptyVoxels.push_back(ss);

    // nearest neighbors distance ratio
    int err2 = lineageHyperTree::findKNearestNucleiNeighborInSpaceEuclideanL2(
        auxNode->data, iterNucleusNNvec,
        distVec);  // distVec is sorted in ascending order
    if (err2 > 0) return err2;
    for (int ii = 0; ii < numKNNdist; ii++) {
      KNNdist[ii].push_back(distVec[ii]);  // ratio between nearest neghbors
    }
  }

  // we need at least 3 elements to calculate std of delta measurements
  if (nucSize.size() < 3) {
    memset(
        &(f[0]), 0,
        sizeof(float) *
            f.size());  // all features except for temporal ones would be zero
    numFeatures += 20;
  } else {
    // subtract mean to all points
    double mm, std;
    for (int ii = 0; ii < dimsImage; ii++) {
      utilsAmatf_meanAndStd<float>(XYZ[ii], &mm, &std);
      for (size_t jj = 0; jj < XYZ[ii].size(); jj++) XYZ[ii][jj] -= mm;
    }

    // fit a line to all the points and calculate deviation
    double v[dimsImage];
    double lambda;
    vector<double> U(XYZ[0].size());
    utilsAmatf_firstPC<float>(XYZ, v, &lambda, &(U[0]));

    float p;
    float lambda2 = lambda * lambda;
    float maxVal =
        -1e32f;  // to remove largest difference (kind of robust metric)
    int maxPos = -1;
    for (size_t jj = 0; jj < U.size(); jj++) {
      p = 0;
      for (int ii = 0; ii < dimsImage; ii++) {
        p += XYZ[ii][jj] * XYZ[ii][jj];
      }

      U[jj] = max(0.0, p - lambda2 * U[jj] * U[jj]);  // distance from point ot
                                                      // straight line. max() to
                                                      // avoid rounding errors
                                                      // for singular cases
                                                      // (perfect line)

      if (U[jj] > maxVal) {
        maxVal = U[jj];
        maxPos = jj;
      }
    }

    utilsAmatf_meanAndStd<double>(U, &mm, &std);
    f[numFeatures++] = mm;   // 0 (usign 0-indexing)
    f[numFeatures++] = std;  // I could look at other features

    // repeat stats removing worse outlier
    U.erase(U.begin() + maxPos);
    utilsAmatf_meanAndStd<double>(U, &mm, &std);
    f[numFeatures++] = mm;   // 2
    f[numFeatures++] = std;  // I could look at other features

    // stdev of the difference between displacements
    f[numFeatures] = 0;
    for (int ii = 0; ii < dimsImage; ii++) {
      for (size_t jj = 0; jj < XYZ[ii].size() - 1; jj++) {
        XYZ[ii][jj] -= XYZ[ii][jj + 1];
      }
      XYZ[ii].pop_back();  // remove last element
      utilsAmatf_meanAndStd<float>(XYZ[ii], &mm, &std);

      f[numFeatures] += std;  // 4
    }
    numFeatures++;

    // relative difference in size
    for (size_t jj = 0; jj < nucSize.size() - 1; jj++) {
      nucSize[jj] =
          (nucSize[jj] - nucSize[jj + 1]) / nucSize[jj];  // relative nucSize
    }
    nucSize.pop_back();
    utilsAmatf_meanAndStd<float>(nucSize, &mm, &std);
    f[numFeatures++] = std;  // 5

    // number of empty voxels
    utilsAmatf_meanAndStd<float>(numEmptyVoxels, &mm, &std);
    f[numFeatures++] = mm;  // 6

    // nearest neighbors absolute and ratio
    double mmKNN;
    utilsAmatf_mean<float>(KNNdist[0], &mmKNN);
    f[numFeatures++] = mmKNN;  // 7
    for (int ii = 1; ii < numKNNdist; ii++) {
      utilsAmatf_mean<float>(KNNdist[ii], &mm);
      f[numFeatures++] = mm;          // absolut 8, 10, 12, 14, 16
      f[numFeatures++] = mm / mmKNN;  // ratio 9, 11, 13, 15, 17
    }

    // offset Jaccard distance (shape )
    utilsAmatf_meanAndStd<float>(nucOffsetJaccard, &mm, &std);
    f[numFeatures++] = mm;   // 18
    f[numFeatures++] = std;  // 19
  }

  // temporal features: based on analyzing branches (so they include cell
  // division)
  // inspired by [1]V. Bettadapura, G. Schindler, T. Plötz, and I. Essa,
  // “Augmenting Bag-of-Words: Data-Driven Discovery of Temporal and Structural
  // Information for Activity Recognition,” in IEEE Conference on Computer
  // Vision and Pattern Recognition (CVPR), 2013.
  int histEvents[3] = {0, 0, 0};  // displacement, death, cell division
                                  // (histogram counting frquency of events)
  int histBranchLengthDeath[3] = {0, 0, 0};  // accumulates stats of length for
                                             // branches ended in cell death.
                                             // The bins are [0,3],[4,6],[7,Inf)
  int histBranchLengthWindow[3] = {0, 0, 0};  // accumulates stats of length for
                                              // branches ended with the
                                              // temporal window.  The bins are
                                              // [0,3],[4,6],[7,Inf)
  int histBranchLengthDiv[3] = {0, 0, 0};     // accumulates stats of length for
  // branches ended in cell divisions.
  // The bins are [0,3],[4,6],[7,Inf)
  int histBin[3] = {3, 6, numeric_limits<int>::max()};
  int numBranches = 0;

  while (q.empty() == false) q.pop();
  q.push(root);

  while (q.empty() == false) {
    numBranches++;
    auxNode = q.front();
    q.pop();
    int lengthBranch = 1;

    while (auxNode != NULL) {
      if (auxNode->data->TM - iniTM + 1 >=
          teporalWindowSize)  // we reached the end of the window size
      {
        // insert branch into histogram
        for (int ii = 0; ii < 3; ii++) {
          if (lengthBranch <= histBin[ii]) {
            histBranchLengthWindow[ii]++;
            break;
          }
        }
        break;
      }

      switch (auxNode->getNumChildren()) {
        case 0:  // cell death
          histEvents[0]++;
          // insert branch into histogram
          for (int ii = 0; ii < 3; ii++) {
            if (lengthBranch <= histBin[ii]) {
              histBranchLengthDeath[ii]++;
              break;
            }
          }
          auxNode = NULL;  // branch termination
          break;

        case 1:  // cell displacement
          histEvents[1]++;
          lengthBranch++;  // continue branch
          if (auxNode->left != NULL)
            auxNode = auxNode->left;
          else
            auxNode = auxNode->right;
          break;

        case 2:  // cell division
          histEvents[2]++;
          q.push(auxNode->left);  // start new branches
          q.push(auxNode->right);
          // insert branch into histogram
          for (int ii = 0; ii < 3; ii++) {
            if (lengthBranch <= histBin[ii]) {
              histBranchLengthDiv[ii]++;
              break;
            }
          }
          auxNode = NULL;  // branch termination
          break;

        default:
          cout << "ERROR: we should never be here!" << endl;
          exit(3);
      }
    }
  }

  // save all the bins and features from histograms
  int histCum = 0;
  for (int ii = 0; ii < 3; ii++) {
    f[numFeatures++] = histEvents[ii];  // absolute value //20, 21, 22
    histCum += histEvents[ii];
  }
  for (int ii = 0; ii < 3; ii++)  // relative value
    f[numFeatures++] = ((float)histEvents[ii]) / (float)histCum;  // 23,24,25
  for (int ii = 0; ii < 3; ii++)
    f[numFeatures++] = histBranchLengthDeath[ii];  // 26,27,28
  for (int ii = 0; ii < 3; ii++)
    f[numFeatures++] = histBranchLengthDiv[ii];  // 29,30,31
  for (int ii = 0; ii < 3; ii++)
    f[numFeatures++] = histBranchLengthWindow[ii];  // 32,33,34

  assert(getNumFeatures() == numFeatures);

  return 0;
}

//=======================================================================================
bool backgroundDetectionFeatures::isCorrectConfidenceLevel(
    TreeNode<ChildrenTypeLineage>* root, int confidenceLevel) {
  if (((int)root->data->confidence) != confidenceLevel) return false;

  queue<TreeNode<ChildrenTypeLineage>*> q;
  q.push(root);
  TreeNode<ChildrenTypeLineage>* auxNode;

  int iniTM = root->data->TM;
  while (q.empty() == false) {
    auxNode = q.front();
    q.pop();

    if (((int)root->data->confidence) != confidenceLevel) return false;

    if (auxNode->data->TM - iniTM < teporalWindowSize) {
      if (auxNode->left != NULL) q.push(auxNode->left);
      if (auxNode->right != NULL) q.push(auxNode->right);
    }
  }

  return true;
}
