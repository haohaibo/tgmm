/*
 * Copyright (C) 2011-2012 by  Fernando Amat
 * See license.txt for full license and copyright notice.
 *
 * Authors: Fernando Amat
 *  lineageHyperTree.cpp
 *
 *  Created on: August 17th, 2012
 *      Author: Fernando Amat
 *
 * \brief Holds together the hierarchical tree formed by supervoxels, nuclei and
 * lineages. It stores lists with all the nodes in the hypertree to make sure
 * addition and deletions are propagated properly
 *
 */

#include <assert.h>
#include <math.h>
#include <time.h>
#include <algorithm>
#include <fstream>
#include <map>

#include "knnCUDA/knnCuda.h"
#include "lineageHyperTree.h"
#include "trackletCalculation.h"
namespace mylib {
#include "histogram.h"  //be careful: BIN_COUNT has been defined by windows.h
#include "image.h"
};

void lineageHyperTree::clear() {
  for (unsigned int ii = 0; ii < maxTM; ii++) {
    supervoxelsList[ii].clear();
    nucleiList[ii].clear();
  }
  lineagesList.clear();
}

lineageHyperTree::lineageHyperTree() : maxTM(0) {
  supervoxelsList = NULL;
  nucleiList = NULL;
  isSublineage = false;
};

lineageHyperTree::lineageHyperTree(unsigned int maxTM_) : maxTM(maxTM_) {
  supervoxelsList = new list<supervoxel>[maxTM_];
  nucleiList = new list<nucleus>[maxTM_];
  isSublineage = false;
};
lineageHyperTree::~lineageHyperTree() {
  clear();
  delete[] supervoxelsList;
  delete[] nucleiList;
};

void lineageHyperTree::deleteLineage(
    list<lineage>::iterator& iterL)  // deletes all the nuclei associated with a
                                     // lineage and the lineage ityself. iterL
                                     // is updated with the next element in
                                     // lineagesList
{
  iterL->bt.reset();

  // delete all nuclei associated with this lineage
  TreeNode<ChildrenTypeLineage>* aux;
  queue<TreeNode<ChildrenTypeLineage>*> q;
  q.push(iterL->bt.pointer_mainRoot());
  while (q.empty() == false) {
    aux = q.front();
    q.pop();

    if (aux->left != NULL) q.push(aux->left);
    if (aux->right != NULL) q.push(aux->right);

    nucleiList[aux->data->TM].erase(aux->data);
  }

  // erase lineage itself
  iterL = lineagesList.erase(iterL);
}
//=============================================================================
lineageHyperTree::lineageHyperTree(const lineageHyperTree& p) : maxTM(p.maxTM) {
  supervoxelsList = new list<supervoxel>[p.maxTM];
  nucleiList = new list<nucleus>[p.maxTM];

  for (unsigned int ii = 0; ii < maxTM; ii++) {
    supervoxelsList[ii] = p.supervoxelsList[ii];
    nucleiList[ii] = p.nucleiList[ii];
  }
  lineagesList = p.lineagesList;
  vecRefLastTreeNode = p.vecRefLastTreeNode;
  isSublineage = p.isSublineage;
  sublineageRootVec = p.sublineageRootVec;
}
lineageHyperTree& lineageHyperTree::operator=(const lineageHyperTree& p) {
  if (maxTM != p.maxTM) {
    cout << "ERROR: at lineageHyperTree::operator= : trying to assign "
            "constructor of different maxTM size"
         << endl;
  }

  if (this != &p) {
    for (unsigned int ii = 0; ii < maxTM; ii++) {
      supervoxelsList[ii] = p.supervoxelsList[ii];
      nucleiList[ii] = p.nucleiList[ii];
    }
    lineagesList = p.lineagesList;
    vecRefLastTreeNode = p.vecRefLastTreeNode;
    isSublineage = p.isSublineage;
    sublineageRootVec = p.sublineageRootVec;
  }
  return *this;
}
//===========================================================================
int lineageHyperTree::readListSupervoxelsFromTif(string filename,
                                                 int vecPosTM) {
  // open image
  mylib::Array* img = mylib::Read_Image((char*)(filename.c_str()), 0);

  if (img == NULL) {
    cout << "ERROR: at lineageHyperTree::readListSupervoxelsFromTif: file "
         << filename << " could not be opened" << endl;
    return 1;
  }

  // hack to make the code work for uin8 without changing everything to
  // templates
  // basically, parse the image to uint16, since the code was designed for
  // uint16
  if (img->type == mylib::UINT8_TYPE) {
    img =
        mylib::Convert_Array_Inplace(img, img->kind, mylib::UINT16_TYPE, 16, 0);
  }
  // hack to make the code work for 2D without changing everything to templates
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
    cout << "ERROR: at lineageHyperTree::readListSupervoxelsFromTif: file "
         << filename << " is not UINT16" << endl;
    return 1;
  }

  // set supervoxel dimsImage
  for (int ii = 0; ii < dimsImage; ii++)
    supervoxel::dataDims[ii] = img->dims[ii];

  // check supervoxelsList
  if (vecPosTM >= (int)maxTM) {
    cout << "ERROR: at lineageHyperTree::readListSupervoxelsFromTif: time "
            "point requested is larger than maxTM. Change that constant and "
            "recompile the code."
         << endl;
    return 1;
  } else {
    supervoxelsList[vecPosTM].clear();
  }

  list<supervoxel>* listPtr =
      (&(supervoxelsList[vecPosTM]));  // to avoid refering to it all the time

  mylib::uint16* imgPtr = (mylib::uint16*)(img->data);
  mylib::uint16 maxLabel = 0, auxLabel = 0;
  vector<list<supervoxel>::iterator>
      vecIter;  // to store location of each label so I can insert efficiently
  vecIter.reserve(65000);
  for (mylib::Indx_Type ii = 0; ii < img->size; ii++) {
    auxLabel = imgPtr[ii];
    if (auxLabel > 0)  // non-background
    {
      // check if this is teh first time we see a label
      if (auxLabel > maxLabel) {
        for (size_t ii = listPtr->size(); ii < auxLabel;
             ii++)  // add new supervoxel to teh list
        {
          listPtr->push_back(supervoxel(vecPosTM));
          vecIter.push_back(
              (++(listPtr->rbegin()))
                  .base());  // iterator for the last element in the list
          vecIter.back()->PixelIdxList.reserve(
              1000);  // to avoid constant dynamic allocation
        }
        maxLabel = auxLabel;
      }

      // add information to supervoxel
      vecIter[auxLabel - 1]->PixelIdxList.push_back(ii);
    }
  }

  // TODO: I don't have the centroids yet since I need the intensity image for
  // that

  // release memory
  mylib::Free_Array(img);
  return 0;
}

//===================================================================================
int lineageHyperTree::readListSupervoxelsFromTifWithWeightedCentroid(
    string filenameImage, string filenameLabels, int vecPosTM) {
  // open image
  mylib::Array* imgL = mylib::Read_Image((char*)(filenameLabels.c_str()), 0);
  mylib::Array* img = mylib::Read_Image((char*)(filenameImage.c_str()), 0);

  if (imgL == NULL) {
    cout << "ERROR: at lineageHyperTree::readListSupervoxelsFromTif: file "
         << filenameLabels << " could not be opened" << endl;
    return 1;
  }
  // hack to make the code work for uin8 without changing everything to
  // templates
  // basically, parse the image to uint16, since the code was designed for
  // uint16
  if (imgL->type == mylib::UINT8_TYPE) {
    imgL = mylib::Convert_Array_Inplace(imgL, imgL->kind, mylib::UINT16_TYPE,
                                        16, 0);
  }
  // hack to make the code work for 2D without changing everything to templates
  // basically, add one black slice to the image (you should select conn3D = 4
  // or 8)
  if (imgL->ndims == 2) {
    mylib::Dimn_Type dimsAux[dimsImage];
    for (int ii = 0; ii < imgL->ndims; ii++) dimsAux[ii] = imgL->dims[ii];
    for (int ii = imgL->ndims; ii < dimsImage; ii++) dimsAux[ii] = 2;

    mylib::Array* imgAux =
        mylib::Make_Array(imgL->kind, imgL->type, dimsImage, dimsAux);
    memset(imgAux->data, 0, (imgAux->size) * sizeof(mylib::uint16));
    memcpy(imgAux->data, imgL->data, imgL->size * sizeof(mylib::uint16));

    mylib::Array* imgSwap = imgAux;
    imgL = imgAux;
    mylib::Free_Array(imgSwap);
  }
  if (imgL->type != mylib::UINT16_TYPE) {
    cout << "ERROR: at lineageHyperTree::readListSupervoxelsFromTif: file "
         << filenameLabels << " is not UINT16" << endl;
    return 1;
  }

  if (img == NULL) {
    cout << "ERROR: at lineageHyperTree::readListSupervoxelsFromTif: file "
         << filenameImage << " could not be opened" << endl;
    return 1;
  }
  // hack to make the code work for uin8 without changing everything to
  // templates
  // basically, parse the image to uint16, since the code was designed for
  // uint16
  if (img->type == mylib::UINT8_TYPE) {
    img =
        mylib::Convert_Array_Inplace(img, img->kind, mylib::UINT16_TYPE, 16, 0);
  }
  // hack to make the code work for 2D without changing everything to templates
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
    cout << "ERROR: at lineageHyperTree::readListSupervoxelsFromTif: file "
         << filenameImage << " is not UINT16" << endl;
    return 1;
  }
  for (int ii = 0; ii < dimsImage; ii++) {
    if (img->dims[ii] != imgL->dims[ii]) {
      cout << "ERROR: at lineageHyperTree::readListSupervoxelsFromTif: "
              "dimensions between label array and image do not agree"
           << endl;
      return 2;
    }
  }

  // set supervoxel dimsImage
  supervoxel::dataSizeInBytes = 1;
  for (int ii = 0; ii < dimsImage; ii++) {
    supervoxel::dataDims[ii] = imgL->dims[ii];
    supervoxel::dataSizeInBytes *= supervoxel::dataDims[ii];
  }
  supervoxel::dataSizeInBytes *= sizeof(mylib::uint16);

  // check supervoxelsList
  if (vecPosTM >= (int)maxTM) {
    cout << "ERROR: at lineageHyperTree::readListSupervoxelsFromTif: time "
            "point requested is beyond maxTM. Change that constant and "
            "recompile the code"
         << endl;
    return 2;
  } else {
    supervoxelsList[vecPosTM].clear();
  }

  list<supervoxel>* listPtr =
      (&(supervoxelsList[vecPosTM]));  // to avoid refering to it all the time

  mylib::uint16* imgLPtr = (mylib::uint16*)(imgL->data);
  mylib::uint16* imgPtr = (mylib::uint16*)(img->data);
  mylib::uint16 maxLabel = 0, auxLabel = 0;
  vector<list<supervoxel>::iterator>
      vecIter;  // to store location of each label so I can insert efficiently
  vecIter.reserve(65000);

  pointND<uint64> auxXYZ;  // to update xyz

  vector<float> wXYZ;  // calculate centroids
  vector<pointND<float> > pXYZ;
  wXYZ.reserve(65000);
  pXYZ.reserve(65000);
  for (mylib::Indx_Type ii = 0; ii < imgL->size; ii++) {
    auxLabel = imgLPtr[ii];
    if (auxLabel > 0)  // non-background
    {
      // check if this is teh first time we see a label
      if (auxLabel > maxLabel) {
        for (size_t ii = listPtr->size(); ii < auxLabel;
             ii++)  // add new supervoxel to teh list
        {
          listPtr->push_back(supervoxel(vecPosTM));
          vecIter.push_back(
              (++(listPtr->rbegin()))
                  .base());  // iterator for the last element in the list
          vecIter.back()->PixelIdxList.reserve(
              1000);  // to avoid constant dynamic allocation
          // to calculate centroid
          wXYZ.push_back(0.0f);
          pXYZ.push_back(pointND<float>());
        }
        maxLabel = auxLabel;
      }

      // add information to supervoxel
      vecIter[auxLabel - 1]->PixelIdxList.push_back(ii);
      // add information for centroid
      float auxI = (float)(imgPtr[ii]);
      wXYZ[auxLabel - 1] += auxI;
      for (int ii = 0; ii < dimsImage; ii++)
        pXYZ[auxLabel - 1].p[ii] += auxI * ((float)(auxXYZ.p[ii]));
    }
    // update XYZ counters
    int dd = 0;
    auxXYZ.p[dd]++;
    while (auxXYZ.p[dd] >= img->dims[dd]) {
      auxXYZ.p[dd++] = 0;
      if (dd >= dimsImage) break;
      auxXYZ.p[dd]++;
    }
  }

  // update centroid
  for (unsigned int ii = 0; ii < pXYZ.size(); ii++) {
    pXYZ[ii] /= wXYZ[ii];
    for (int jj = 0; jj < dimsImage; jj++)
      vecIter[ii]->centroid[jj] = pXYZ[ii].p[jj];
    vecIter[ii]->dataPtr = img->data;  // save data
  }

  // release memory
  mylib::Free_Array(imgL);
  // mylib::Free_Array(img);//we want to preserve the pointer to the data
  return 0;
}

//==============================================================================
int lineageHyperTree::readBinaryRnkTGMMfile(string filenameRnk, int vecPosTM) {
  ifstream fid(filenameRnk.c_str(), ios::binary | ios::in);

  if (!fid.is_open()) {
    cout << "ERROR: at readBinaryRnkTGMMfile: file " << filenameRnk
         << " could not be opened" << endl;
    return 1;
  }

  if (vecPosTM >= (int)maxTM) {
    cout << "ERROR: at readBinaryRnkTGMMfile: time point requested is larger "
            "than maxTM. Change the constant and reocmpile the code."
         << endl;
    return 1;
  }

  mylib::int32 numLabels = 0;
  fid.read((char*)(&numLabels), sizeof(mylib::int32));
  mylib::int32 maxGaussiansPerVoxel = 0;
  fid.read((char*)(&maxGaussiansPerVoxel), sizeof(mylib::int32));

  int ll = maxGaussiansPerVoxel * numLabels;

  mylib::float32* rnk = new mylib::float32[ll];
  mylib::int32* ind = new mylib::int32[ll];

  fid.read((char*)rnk, sizeof(mylib::float32) * ll);
  fid.read((char*)ind, sizeof(mylib::int32) * ll);

  fid.close();

  // preallocate all the nuclei space as a list and keep iterators since we need
  // to insert them in order
  list<nucleus>* listPtr = &(nucleiList[vecPosTM]);  // pointer to list
  int numNuclei = 0;
  int count = 0;
  for (int ii = 0; ii < ll; ii++) numNuclei = max(numNuclei, ind[ii]);
  numNuclei++;  // indexes start at 0, so the number of nuclei have to be
                // increased by one

  listPtr->resize(numNuclei);  // the default constructor is used.
  vector<list<nucleus>::iterator> vecIter(numNuclei);
  for (list<nucleus>::iterator iter = listPtr->begin(); iter != listPtr->end();
       ++iter) {
    vecIter[count] = iter;
    iter->TM = vecPosTM;
    count++;
  }

  // recollect supervoxels list iterator in order to be able to parse
  // information
  vector<list<supervoxel>::iterator> vecIterSupervoxel;
  getSupervoxelListIteratorsAtTM(vecIterSupervoxel, vecPosTM);

  // parse information to hypertree
  int auxInd = 0;
  count = 0;
  for (int ii = 0; ii < maxGaussiansPerVoxel; ii++) {
    for (int jj = 0; jj < numLabels; jj++) {
      if (rnk[count] > 0.5f)  // guaranteed unique assignment between a
                              // supervoxel and a nuclei among all the
      {
        // jj-th supervoxel goes to ind[count] nucleus
        vecIter[ind[count]]->treeNode.addChild(vecIterSupervoxel[jj]);
        vecIterSupervoxel[jj]->treeNode.setParent(vecIter[ind[count]]);
      }
      count++;
    }
  }

  // relase memory
  delete[] rnk;
  delete[] ind;

  return 0;
}

//====================================================
/*
int lineageHyperTree::parseTGMMframeResult(vector<GaussianMixtureModelRedux*>
&vecGM, int vecPosTM)
{
        if(vecPosTM >= (int)maxTM)
        {
                cout<<"ERROR: at lineageHyperTree::parseTGMMframeResult: list
does not contain as many time points as requested. You need to make maxTM larger
and recompile the code."<<endl;
                return 1;
        }

        list<nucleus>* listPtr = &(nucleiList[vecPosTM]);//pointer to list from
current time point

        vector< list<nucleus>::iterator > vecIterGMold;
        if(vecPosTM>0)
                getNucleiListIteratorsAtTM(vecIterGMold,vecPosTM-1);//we will
need it to find the lineage
        else//we are starting lineages
        {
                lineagesList.resize(vecGM.size());
                for(list< lineage >::iterator iter = lineagesList.begin(); iter
!= lineagesList.end(); ++iter)
                {
                        (*iter) = lineage();
                }
        }

        if(vecGM.size() != listPtr->size())
        {
                cout<<"ERROR: at lineageHyperTree::parseTGMMframeResult: number
of nuclei in vecGM does not agree with number of nuclei in the list for this
time point"<<endl;
                return 1;
        }


        vector< list<lineage>::iterator > vecIter;
        getLineageListIterators(vecIter);

        int count = 0;
        for(list<nucleus>::iterator iter = listPtr->begin(); iter !=
listPtr->end(); ++iter, count++)
        {
                for(int jj=0; jj<dimsImage; jj++)
                        iter->centroid[jj] = vecGM[count]->m_k[jj];
                iter->avgIntensity =
vecGM[count]->beta_k-vecGM[count]->beta_o;//N_k

                int parent = vecGM[count]->parentId;
                if(vecPosTM > 0)//find parent: a little bit slow since we are
going to traverse the graph until we find it
                {
                        if(parent < 0)
                        {
                                cout<<"ERROR: at
lineageHyperTree::parseTGMMframeResult: blob does not have a parent even if time
poitn>0. Something must be wrong"<<endl;
                                return 3;
                        }
                        if(parent >= (int)(vecIterGMold.size()))
                        {
                                cout<<"ERROR: at
lineageHyperTree::parseTGMMframeResult: parentId is larger than the number of
elements in previous. Something must be wrong"<<endl;
                                return 4;
                        }

                        bool errB = vecIter[ vecGM[count]->lineageId
]->bt.findTreeNodeBFSforPointers(vecIterGMold[parent]);//finds parent so we can
new element as a child

                        if(errB == false)
                        {
                                cout<<"ERROR: at
lineageHyperTree::parseTGMMframeResult: parent could not be found in the tree.
Something must be wrong"<<endl;
                                return 2;
                        }
                }else{
                        if(vecIter[ vecGM[count]->lineageId ]->bt.IsEmpty() ==
false)
                        {
                                cout<<"ERROR: at
lineageHyperTree::parseTGMMframeResult: binary tree should be empty since we are
at time point 0"<<endl;
                                return 2;
                        }
                }
                //update lineage and nucleus graph
                int errI = vecIter[ vecGM[count]->lineageId ]->bt.insert(iter);

                if(errI >0)
                        return errI;

                iter->treeNode.setParent( vecIter[ vecGM[count]->lineageId ]
);//update nucleus graph
        }
        return 0;
}
*/

//====================================================
int lineageHyperTree::parseTGMMframeResult(
    vector<GaussianMixtureModelRedux*>& vecGM, int vecPosTM) {
  if (vecPosTM >= (int)maxTM) {
    cout << "ERROR: at lineageHyperTree::parseTGMMframeResult: list does not "
            "contain as many time points as requested. You need to make maxTM "
            "larger and recompile the code."
         << endl;
    return 1;
  }

  list<nucleus>* listPtr =
      &(nucleiList[vecPosTM]);  // pointer to list from current time point

  if (vecRefLastTreeNode.empty() == true)  // we are starting lineages
  {
    lineagesList.resize(vecGM.size());
    for (list<lineage>::iterator iter = lineagesList.begin();
         iter != lineagesList.end(); ++iter) {
      (*iter) = lineage();
    }
  }

  if (vecGM.size() != listPtr->size()) {
    cout << "ERROR: at lineageHyperTree::parseTGMMframeResult: number of "
            "nuclei in vecGM does not agree with number of nuclei in the list "
            "for this time point"
         << endl;
    return 1;
  }

  // set scale
  supervoxel::setScale(vecGM[0]->scale);

  vector<TreeNode<ChildrenTypeLineage>*> vecRefLastTreeNodeBackup(
      vecGM.size());  // to keep location of new added elements, so we can save
                      // them at the end
  list<lineage>::iterator iterL =
      lineagesList.begin();  // in case this is the first time point

  int count = 0;
  for (list<nucleus>::iterator iter = listPtr->begin(); iter != listPtr->end();
       count++, ++iterL) {
    for (int jj = 0; jj < dimsImage; jj++)
      iter->centroid[jj] = vecGM[count]->m_k[jj];
    iter->avgIntensity = vecGM[count]->beta_k - vecGM[count]->beta_o;  // N_k

    if (iter->isDead())  // we want to remove this element from the list of
                         // nuclei
    {
      if (iter->treeNode.getNumChildren() >
          0)  // nothing should be attached to it in the hierarchical graph
      {
        cout << "ERROR: at lineageHyperTree::parseTGMMframeResult: A dead "
                "nucleus has supervoxels pointing to it. Something must be "
                "wrong"
             << endl;
        return 3;
      }

      // remove nuclei itself
      iter = listPtr->erase(iter);  // returns the next iter available after teh
                                    // erased position so we do not need to do
                                    // anything for the loop

      vecRefLastTreeNodeBackup[count] = NULL;
      continue;
    }

    int parent = vecGM[count]->parentId;

    if (vecRefLastTreeNode.empty() == false)  // we have previous lineages
    {
      if (parent < 0) {
        cout << "ERROR: at lineageHyperTree::parseTGMMframeResult: blob does "
                "not have a parent even if time point>0. Something must be "
                "wrong"
             << endl;
        return 3;
      }
      if (parent >= (int)(vecRefLastTreeNode.size())) {
        cout << "ERROR: at lineageHyperTree::parseTGMMframeResult: parentId is "
                "larger than the number of elements in previous. Something "
                "must be wrong"
             << endl;
        return 4;
      }
      bool flag = vecRefLastTreeNode[parent]->data->treeNode.getParent(iterL);
      if (flag == false) {
        cout << "ERROR: at lineageHyperTree::parseTGMMframeResult: parent "
                "element is not associated with a lineage. Something must be "
                "wrong"
             << endl;
        return 5;
      }
      iterL->bt.SetCurrent(vecRefLastTreeNode[parent]);  // prepare for
                                                         // insertion
    }

    // update lineage and nucleus graph
    TreeNode<ChildrenTypeLineage>* errI = iterL->bt.insert(iter);

    if (errI == NULL) return 3;

    vecRefLastTreeNodeBackup[count] = errI;  // save pointer for next iteration
    iter->treeNode.setParent(iterL);         // update nucleus graph
    iter->treeNodePtr = errI;
    ++iter;  // incrmeent by one
    //++iterL;//only necessary for the initial time point, but it does not harm
    // in other cases since iterL will be updated before being used. Now done
    // inside the for loop command
  }

  // update vecRefLastTreeNode in case we need it later
  vecRefLastTreeNode = vecRefLastTreeNodeBackup;
  return 0;
}

//=================================================================
int lineageHyperTree::parseNucleiList2TGMM(
    vector<GaussianMixtureModelRedux*>& vecGM, int vecPosTM) {
  cout << "ERROR: at lineageHyperTree::parseNucleiList2TGMM: function ot "
          "implemented yet"
       << endl;
  return 1;
}

//====================================================================================================================================
int lineageHyperTree::debugCheckHierachicalTreeConsistency() {
  cout << "DEBUGGING: at "
          "lineageHyperTree::debugCheckHierachicalTreeConsistency()"
       << endl;

  // check supervoxels-nuclei connexion
  float norm;
  for (unsigned int ii = 0; ii < maxTM; ii++) {
    int count = 0;

    for (list<nucleus>::iterator iter = nucleiList[ii].begin();
         iter != nucleiList[ii].end(); ++iter) {
      if (iter->treeNode.getNumChildren() > 100) {
        cout << "WARNING / ERROR: "
                "lineageHyperTree::debugCheckHierachicalTreeConsistency(): "
                "number of children supervoxels at nucleus "
             << count << " TM=" << ii << " is "
             << iter->treeNode.getNumChildren() << " (too large) " << endl;
        cout << "Nucleus:" << (*iter) << endl;
        return 15;
      }

      // cout<<"Checking nucleus "<<count<<endl;
      for (vector<ChildrenTypeNucleus>::iterator iter2 =
               iter->treeNode.getChildren().begin();
           iter2 != iter->treeNode.getChildren().end(); ++iter2) {
        if (((*iter2)->treeNode.hasParent() == false) ||
            ((*iter2)->treeNode.getParent() != iter)) {
          cout << "ERROR: "
                  "lineageHyperTree::debugCheckHierachicalTreeConsistency(): "
                  "disagreement between nucleus "
               << count << " and supervoxel at TM=" << ii << endl;
          cout << "Nucleus:" << (*iter) << endl;
          cout << "Supervoxel:" << (*(*iter2)) << endl;
          return 1;
        }

        if (((*iter2)->treeNode.hasParent() == true)) {
          if ((*iter2)->TM != iter->TM) {
            cout << "ERROR: "
                    "lineageHyperTree::debugCheckHierachicalTreeConsistency(): "
                    "disagreement between nucleus "
                 << count << " and supervoxel about TM" << endl;
            cout << "Nucleus:" << (*iter) << endl;
            cout << "Supervoxel:" << (*(*iter2)) << endl;
            return 10;
          }
        }
      }

      // check if nucleus centroid si equal to supervoxel centroid when there is
      // a one to one correspondence
      if (iter->treeNode.getNumChildren() == 1) {
        norm = 0.0f;
        for (int ii = 0; ii < dimsImage; ii++)
          norm += pow(iter->centroid[ii] -
                          iter->treeNode.getChildren()[0]->centroid[ii],
                      2);

        if (sqrt(norm) > 5) {
          cout << "ERROR: "
                  "lineageHyperTree::debugCheckHierachicalTreeConsistency(): "
                  "disagreement between nucleus "
               << count << " and supervoxel about centroid" << endl;
          cout << "Nucleus:" << (*iter) << endl;
          cout << "Supervoxel:" << *(iter->treeNode.getChildren()[0]) << endl;
          return 100;  // you might have to uncomment this since the code has
                       // not been completely fixed yet
        }
      } else {
        float centroidOld[dimsImage];
        memcpy(centroidOld, iter->centroid, sizeof(float) * dimsImage);
        assert(supervoxel::dataSizeInBytes /
                   (supervoxel::dataDims[0] * supervoxel::dataDims[1] *
                    supervoxel::dataDims[2]) ==
               4);
        calculateNucleiIntensityCentroid<float>(iter);
        norm = 0.0f;
        for (int ii = 0; ii < dimsImage; ii++)
          norm += pow(iter->centroid[ii] - centroidOld[ii], 2);
        if (sqrt(norm) > 5) {
          cout << "ERROR: "
                  "lineageHyperTree::debugCheckHierachicalTreeConsistency(): "
                  "disagreement between nucleus "
               << count << " and supervoxel about centroid (nucleus with more "
                           "than one supervoxel)"
               << endl;
          cout << "Centroid old:" << centroidOld[0] << " " << centroidOld[1]
               << " " << centroidOld[2] << endl;
          cout << "Nucleus:" << (*iter) << endl;
          for (size_t aa = 0; aa < iter->treeNode.getNumChildren(); aa++)
            cout << "Supervoxel " << aa << ":"
                 << *(iter->treeNode.getChildren()[aa]) << endl;
          return 100;  // you might have to uncomment this since the code has
                       // not been completely fixed yet
        }
      }

      count++;
    }
  }

  // check nuclei-lineage connexion
  int count = 0;
  for (list<lineage>::iterator iter = lineagesList.begin();
       iter != lineagesList.end(); ++iter) {
    // cout<<"Checking lineage "<<count<<endl;
    /*
    if( count == 2938)
    {
            cout<<"Checking lineage "<<count<<endl;
            debugPrintLineage(count);
    }
    */
    if (iter->bt.IsEmpty() == false) {
      // traverse the tree from the root
      queue<TreeNode<ChildrenTypeLineage>*> q;
      q.push(iter->bt.pointer_mainRoot());
      TreeNode<ChildrenTypeLineage>* aux;
      int count2 = 0;
      while (q.empty() == false) {
        aux = q.front();
        q.pop();  // delete element
        if (aux != NULL) {
          count2++;
          // if( count == 2938)	cout<<"Checking blob "<<count2<<" in while
          // loop. Nucleus "<< (*(aux->data))<<endl;

          if (aux->data->treeNode.hasParent() == false ||
              aux->data->treeNode.getParent() != iter) {
            cout << "ERROR: "
                    "lineageHyperTree::debugCheckHierachicalTreeConsistency(): "
                    "disagreement between lineage "
                 << count << " and nucleus at TM=" << aux->data->TM << endl;
            return 1;
          }

          // if( count == 2938) cout<<"Checking rule 2"<<endl;
          if (aux != aux->data->treeNodePtr) {
            cout << "ERROR: "
                    "lineageHyperTree::debugCheckHierachicalTreeConsistency(): "
                    "disagreement between lineage "
                 << count << " and nucleus pointer at TM=" << aux->data->TM
                 << endl;
            return 1;
          }

          // if( count == 2938) cout<<"Checking rule 3"<<endl;
          if (aux->parent != NULL) {
            if (aux->parent->left != aux) {
              if (aux->parent->right != aux) {
                cout << "ERROR: "
                        "lineageHyperTree::"
                        "debugCheckHierachicalTreeConsistency(): disagreement "
                        "at lineage "
                     << count << " at TM=" << aux->data->TM
                     << " between parent and son in the binary tree" << endl;
                return 1;
              }
            }
          }

          // if( count == 2938) cout<<"Checking rule 4"<<endl;

          if (aux->left != NULL) {
            if (aux->data->TM != (aux->left->data->TM - 1)) {
              cout << "ERROR: "
                      "lineageHyperTree::debugCheckHierachicalTreeConsistency()"
                      ": disagreement at lineage "
                   << count << " at TM=" << aux->data->TM
                   << " between TM parent and left son (TM="
                   << aux->left->data->TM << ") in the binary tree" << endl;
              return 1;
            }
          }

          // if( count == 2938) cout<<"Checking rule 5"<<endl;
          if (aux->right != NULL) {
            if (aux->data->TM != (aux->right->data->TM - 1)) {
              cout << "ERROR: "
                      "lineageHyperTree::debugCheckHierachicalTreeConsistency()"
                      ": disagreement at lineage "
                   << count << " at TM=" << aux->data->TM
                   << " between TM parent and right son (TM="
                   << aux->right->data->TM << ") in the binary tree" << endl;
              return 1;
            }
          }
          // if( count == 2938) cout<<"Finish with rules"<<endl;
          q.push(aux->left);
          q.push(aux->right);
        }
      }
    }
    count++;
  }

  return 0;
}

//==================================================================================================
void lineageHyperTree::debugPrintLineage(int lineageNumber) {
  cout << "DEBUGGING:lineageHyperTree::debugPrintLineage " << lineageNumber
       << endl;
  if (lineageNumber >= lineagesList.size()) {
    cout << "Lineage does not exist" << endl;
    return;
  }

  // find pointer to lineage
  list<lineage>::iterator iter = lineagesList.begin();
  int count = 0;
  for (; iter != lineagesList.end(); ++iter, count++) {
    if (count == lineageNumber) break;
  }

  if (iter->bt.IsEmpty()) {
    cout << "Lineage is empty" << endl;
    return;
  }

  vector<TreeNode<ChildrenTypeLineage>*>
      vecTreeNodes;  // to retrive pointer to all the nodes in the tree
  iter->bt.traverseBinaryTreeBFS(vecTreeNodes);

  for (vector<TreeNode<ChildrenTypeLineage>*>::const_iterator iter =
           vecTreeNodes.begin();
       iter != vecTreeNodes.end(); ++iter) {
    cout << (*((*iter)->data));
    if ((*iter)->parent != NULL)
      cout << "; Parent :" << (*((*iter)->parent->data)) << endl;
    else
      cout << "; Parent :"
           << "NULL" << endl;
  }
}

void lineageHyperTree::debugPrintNucleus(int TM, int nucleusNumber) {
  cout << "DEBUGGING: lineageHyperTree::debugPrintNucleus TM=" << TM
       << " nucleus " << nucleusNumber << endl;

  if (TM >= (int)maxTM || nucleusNumber >= (int)nucleiList[TM].size()) {
    cout << "Nucleus does not exist" << endl;
    return;
  }

  // find pointer to lineage
  list<nucleus>::iterator iter = nucleiList[TM].begin();
  int count = 0;
  for (; iter != nucleiList[TM].end(); ++iter, count++) {
    if (count == nucleusNumber) break;
  }

  cout << "Nucleus: " << (*iter) << endl;

  // print out supervoxels
  count = 0;
  for (vector<ChildrenTypeNucleus>::iterator iterS =
           iter->treeNode.getChildren().begin();
       iterS != iter->treeNode.getChildren().end(); ++iterS) {
    cout << "Supervoxel " << count << ": " << (*(*iterS)) << endl;
    count++;
  }
}

//---------------------------------------------------
void lineageHyperTree::debugPrintLineageForLocalLineageDisplayinMatlab(
    string imgPath, string imgLPath, string suffix, string imgRawPath) {
  // generate temporary folder
  string pathTemp;
  char extra[256];
#if defined(_WIN32) || defined(_WIN64)
  pathTemp = string("E:\\temp\\temporalLogicalRules\\");
  SYSTEMTIME str_t;
  GetSystemTime(&str_t);
  sprintf(extra, "%slocalLineageDisplay_%s_%d_%d_%d_%d_%d_%d_%d\\",
          pathTemp.c_str(), suffix.c_str(), str_t.wYear, str_t.wMonth,
          str_t.wDay, str_t.wHour, str_t.wMinute, str_t.wSecond,
          str_t.wMilliseconds);
#else
  pathTemp = string("E:/temp/temporalLogicalRules/");
  sprintf(extra, "%slocalLineageDisplay_%s_%ld\\", pathTemp.c_str(),
          suffix.c_str(), time(NULL));
#endif
  string pathFiles(extra);
  string cmd = string("mkdir " + pathFiles);
  int error = system(cmd.c_str());
  if (error > 0) {
    cout << "ERROR at "
            "lineageHyperTree::debugPrintLineageForLocalLineageDisplayinMatlab "
            "("
         << error << "): generating path file " << pathFiles << endl;
    cout << "With command " << cmd << endl;
    exit(error);
  } else {
    cout << "DEBUGGING: "
            "lineageHyperTree::debugPrintLineageForLocalLineageDisplayinMatlab:"
            " files saved at "
         << pathFiles << endl;
  }

  // figure out maximum dimensions needed to crop images
  uint64 minXYZ[dimsImage];
  uint64 maxXYZ[dimsImage];

  for (int ii = 0; ii < dimsImage; ii++) {
    minXYZ[ii] = (uint64)pow(2.0, 63);
    maxXYZ[ii] = 0;
  }

  for (unsigned int ii = 0; ii < maxTM; ii++) {
    for (list<nucleus>::const_iterator iter = nucleiList[ii].begin();
         iter != nucleiList[ii].end(); ++iter) {
      for (int jj = 0; jj < dimsImage; jj++) {
        minXYZ[jj] = min(minXYZ[jj], (uint64)(iter->centroid[jj]));
        maxXYZ[jj] = max(maxXYZ[jj], (uint64)(iter->centroid[jj]));
      }
    }
  }

  uint64 imPad[dimsImage] = {30, 30, 3};
  for (int ii = 0; ii < dimsImage; ii++) {
    maxXYZ[ii] = std::min(maxXYZ[ii] + imPad[ii],
                          supervoxel::dataDims[ii]);  // add some room
    minXYZ[ii] = std::max(minXYZ[ii] - imPad[ii], uint64(0));
  }

  string imgPrefix(pathFiles + "TM_");
  string imgSuffix("cropped_");

  for (unsigned int ii = 0; ii < maxTM; ii++) {
    char itoaB[16];
    sprintf(itoaB, "%.5d", ii);
    string itoa(itoaB);
    string imgBasename(imgPrefix + itoa + imgSuffix);
    debugCanvasFromSegmentationCroppedRegion(ii, minXYZ, maxXYZ, imgBasename,
                                             imgRawPath);
  }

  // generate blobStruct from lineage
  string filenameXML(pathFiles + "blobStructStackMCMC.xml");
  debugPrintLineageToMCMCxmlFile(filenameXML, imgPrefix,
                                 imgSuffix + "RawBckgSubt_");

  // write rectangular coordinates and time point
  ofstream fout((pathFiles + "configFile.txt").c_str());
  for (int ii = 0; ii < dimsImage; ii++) {
    fout << minXYZ[ii] << " " << maxXYZ[ii] << " ";
  }
  fout << nucleiList[0].front().TM << " "
       << supervoxel::getScale()[dimsImage - 1] << endl;

  // write path to image and image segmentation mask
  fout << imgRawPath << endl;
  fout << imgPath << endl;
  fout << imgLPath << endl;

  fout.close();

  return;
}

void lineageHyperTree::debugCanvasFromSegmentationCroppedRegion(
    unsigned int TM, uint64 minXYZ[dimsImage], uint64 maxXYZ[dimsImage],
    string imgBasename, string imgRawPath) {
  double betaThr = 0.1;  // percentile to saturate images to enhance contrast
  double alpha =
      0.2;  // alpha blending value to merge labels and image intensity

  if (TM >= maxTM) return;

  // make sure cropping regions do not exceed image dimensions
  for (int ii = 0; ii < dimsImage; ii++) {
    if (maxXYZ[ii] <= minXYZ[ii]) {
      cout << "ERROR: at "
              "lineageHyperTree::debugCanvasFromSegmentationCroppedRegion: min "
              "and max coordinates do not agree. min ="
           << minXYZ[ii] << ";max=" << maxXYZ[ii] << endl;
      exit(32);
    }
    if (maxXYZ[ii] >= supervoxel::dataDims[ii])
      maxXYZ[ii] = supervoxel::dataDims[ii] - 1;
  }

  // define final image name
  char itoaB[16];
  sprintf(itoaB, "%.5d", TM);
  string itoa(itoaB);
  string canvasFilename(imgBasename + "RawBckgSubtSupervoxel_" + itoa + ".tif");
  string imgRawFilename(imgBasename + "RawBckgSubt_" + itoa + ".tif");

  if (supervoxelsList[TM].empty() ==
      true)  // impossible to recover the image data: we just write blank
  {
    mylib::Dimn_Type dims[dimsImage];
    for (int ii = 0; ii < dimsImage; ii++)
      dims[ii] = maxXYZ[ii] - minXYZ[ii] + 1;
    mylib::Array* canvas =
        Make_Array(mylib::RGB_KIND, mylib::UINT8_TYPE, dimsImage, dims);
    memset(canvas->data, 0, sizeof(mylib::uint8) * canvas->size);
    if (mylib::Write_Image((char*)(imgRawFilename.c_str()), canvas,
                           mylib::DONT_PRESS) == 1) {
      cout << "ERROR: at "
              "lineageHyperTree::debugCanvasFromSegmentationCroppedRegion: "
              "file "
           << imgRawFilename << " cannot be written" << endl;
      exit(3);
    }

    mylib::Free_Array(canvas);
    return;
  }

  // regenerate segmentation mask from supervoxels
  mylib::Dimn_Type dims[dimsImage];
  for (int ii = 0; ii < dimsImage; ii++) dims[ii] = supervoxel::dataDims[ii];
  // mylib::Array* img = mylib::Make_Array_Of_Data (mylib::PLAIN_KIND,
  // mylib::UINT16_TYPE,dimsImage, dims, supervoxelsList[TM].begin()->dataPtr);
  // mylib::Array* img = mylib::Make_Array(mylib::PLAIN_KIND,
  // mylib::UINT16_TYPE,dimsImage, dims);
  // memcpy(img->data,supervoxelsList[TM].begin()->dataPtr,supervoxelsList[TM].begin()->dataSizeInBytes);//we
  // do not want array to take ownership
  // mylib::uint16* imgptr = (mylib::uint16*)(img->data);

  mylib::uint16* imgptr =
      (mylib::uint16*)(supervoxelsList[TM].begin()->dataPtr);

  mylib::Array* imgL =
      mylib::Make_Array(mylib::PLAIN_KIND, mylib::UINT16_TYPE, dimsImage, dims);
  mylib::uint16* imgLptr = (mylib::uint16*)(imgL->data);

  if (supervoxelsList[TM].size() > 65535) {
    cout << "ERROR: at "
            "lineageHyperTree::debugCanvasFromSegmentationCroppedRegion: code "
            "is not ready for more than 2^16-1 supervoxels per time point "
         << endl;
    exit(3);
  }

  memset(imgLptr, 0,
         sizeof(mylib::uint16) * (imgL->size));  // set all to background
  mylib::uint16 auxL = 1;
  for (list<supervoxel>::const_iterator iterS = supervoxelsList[TM].begin();
       iterS != supervoxelsList[TM].end(); ++iterS) {
    for (vector<uint64>::const_iterator iter = iterS->PixelIdxList.begin();
         iter != iterS->PixelIdxList.end(); ++iter) {
      imgLptr[*iter] = auxL;
    }
    auxL++;
  }

  // cropped both images into the smaller region
  /*
  mylib::Indx_Type idx, offset;
  idx = 0;offset = 1;
  for(int ii = 0; ii<dimsImage; ii++)
  {
          idx += minXYZ[ii] * offset;
          offset *= supervoxel::dataDims[ii];
  }
  mylib::Coordinate* beg = mylib::Idx2CoreA (imgL, idx);
  mylib::Coordinate* begL = mylib::Idx2CoreA (imgL, idx);
  idx = 0;offset = 1;
  for(int ii = 0; ii<dimsImage; ii++)
  {
          idx += maxXYZ[ii] * offset;
          offset *= supervoxel::dataDims[ii];
  }
  mylib::Coordinate* end = mylib::Idx2CoreA (imgL, idx);
  mylib::Coordinate* endL = mylib::Idx2CoreA (img, idx);
  */
  // mylib::Slice* imgSlice = Make_Slice (img, beg, end );
  // mylib::Slice* imgLSlice = Make_Slice (imgL, begL, endL );
  // mylib::Array* imgCropped = mylib::Make_Array_From_Slice (imgSlice );
  // mylib::Array* imgLCropped = mylib::Make_Array_From_Slice (imgSlice );

  mylib::Dimn_Type dimsSlice[dimsImage];
  for (int ii = 0; ii < dimsImage; ii++)
    dimsSlice[ii] = maxXYZ[ii] - minXYZ[ii];

  mylib::Array* imgCropped = mylib::Make_Array(
      mylib::PLAIN_KIND, mylib::UINT16_TYPE, dimsImage, dimsSlice);
  mylib::Array* imgLCropped = mylib::Make_Array(
      mylib::PLAIN_KIND, mylib::UINT16_TYPE, dimsImage, dimsSlice);
  mylib::uint16* imgCroppedptr = (mylib::uint16*)(imgCropped->data);
  mylib::uint16* imgLCroppedptr = (mylib::uint16*)(imgLCropped->data);
  mylib::Indx_Type p, pos;

  /*
  e = mylib::Set_Slice_To_Last(imgSlice);
  pos = 0;
  for (p = mylib::Set_Slice_To_First(imgSlice); 1; p =
  mylib::Next_Slice_Index(imgSlice))
  {
          imgCroppedptr[pos++] = imgptr[p];
          if (p == e) break;
  }

  e = mylib::Set_Slice_To_Last(imgLSlice);
  pos = 0;
  for (p = mylib::Set_Slice_To_First(imgLSlice); 1; p =
  mylib::Next_Slice_Index(imgLSlice))
  {
          imgLCroppedptr[pos++] = imgLptr[p];
          if (p == e) break;
  }
  */

  pos = 0;
  for (uint64 zz = minXYZ[2]; zz < maxXYZ[2]; zz++) {
    for (uint64 yy = minXYZ[1]; yy < maxXYZ[1]; yy++) {
      p = minXYZ[0] + imgL->dims[0] * (yy + imgL->dims[1] * zz);
      for (uint64 xx = minXYZ[0]; xx < maxXYZ[0]; xx++) {
        imgCroppedptr[pos] = imgptr[p];
        imgLCroppedptr[pos] = imgLptr[p];
        pos++;
        p++;
      }
    }
  }

  // crop and save raw data
  if (imgRawPath.length() > 1) {
    string imgRawPathCopy(imgRawPath);
    cout << imgRawPathCopy << endl;
    parseImagePath(imgRawPathCopy,
                   supervoxelsList[TM].begin()->TM);  // get filename
    cout << imgRawPathCopy << endl;
    // open image
    mylib::Array* imgRaw =
        mylib::Read_Image((char*)(imgRawPathCopy.c_str()), 0);

    if (imgRaw == NULL) {
      cout << "ERROR: opening raw image " << imgRawPathCopy << endl;
      exit(2);
    }
    // hack to make the code work for uin8 without changing everything to
    // templates
    // basically, parse the image to uint16, since the code was designed for
    // uint16
    if (imgRaw->type == mylib::UINT8_TYPE) {
      imgRaw = mylib::Convert_Array_Inplace(imgRaw, imgRaw->kind,
                                            mylib::UINT16_TYPE, 16, 0);
    }
    // hack to make the code work for 2D without changing everything to
    // templates
    // basically, add one black slice to the image (you should select conn3D = 4
    // or 8)
    if (imgRaw->ndims == 2) {
      mylib::Dimn_Type dimsAux[dimsImage];
      for (int ii = 0; ii < imgRaw->ndims; ii++) dimsAux[ii] = imgRaw->dims[ii];
      for (int ii = imgRaw->ndims; ii < dimsImage; ii++) dimsAux[ii] = 2;

      mylib::Array* imgAux =
          mylib::Make_Array(imgRaw->kind, imgRaw->type, dimsImage, dimsAux);
      memset(imgAux->data, 0, (imgAux->size) * sizeof(mylib::uint16));
      memcpy(imgAux->data, imgRaw->data, imgRaw->size * sizeof(mylib::uint16));

      mylib::Array* imgSwap = imgAux;
      imgRaw = imgAux;
      mylib::Free_Array(imgSwap);
    }
    if (imgRaw->type != mylib::UINT16_TYPE) {
      cout << "ERROR: raw image should be uint16" << endl;
      exit(4);
    }
    if (imgRaw->size != imgL->size) {
      cout << "ERROR: raw image should have the same size as background "
              "subtracted image"
           << endl;
      exit(4);
    }
    mylib::uint16* imgRawPtr = (mylib::uint16*)(imgRaw->data);
    mylib::Array* imgRawCropped = mylib::Make_Array(
        mylib::PLAIN_KIND, mylib::UINT16_TYPE, dimsImage, dimsSlice);
    mylib::uint16* imgRawCroppedPtr = (mylib::uint16*)(imgRawCropped->data);

    // cout<<"Preping raw data"<<endl;

    // generate cropped image
    pos = 0;
    for (uint64 zz = minXYZ[2]; zz < maxXYZ[2]; zz++) {
      for (uint64 yy = minXYZ[1]; yy < maxXYZ[1]; yy++) {
        p = minXYZ[0] + imgL->dims[0] * (yy + imgL->dims[1] * zz);
        for (uint64 xx = minXYZ[0]; xx < maxXYZ[0]; xx++) {
          imgRawCroppedPtr[pos] = imgRawPtr[p];
          pos++;
          p++;
        }
      }
    }

    // normalize image to uint8
    mylib::Array* imgRawCroppedUINT8 = mylib::Make_Array(
        mylib::PLAIN_KIND, mylib::UINT8_TYPE, dimsImage, dimsSlice);
    mylib::uint8* imgRawCroppedUINT8Ptr =
        (mylib::uint8*)(imgRawCroppedUINT8->data);

    mylib::Value v1, v2;
    v1.uval = 1;
    v2.uval = 0;
    mylib::Histogram* h =
        mylib::Histogram_Array(imgRawCropped, 0x10000, v1, v2);
    h->total -= h->counts[0];  // we do not want to count the background values
                               // in the statistics
    h->counts[0] = 0;
    unsigned int thrH = mylib::Percentile2Value(h, betaThr).uval;
    unsigned int thrL = mylib::Percentile2Value(h, 1.0 - betaThr).uval;
    double rangeLH = (double)(thrH - thrL);
    for (mylib::Size_Type ii = 0; ii < imgRawCropped->size; ii++) {
      if (*imgRawCroppedPtr <= thrL)
        *imgRawCroppedUINT8Ptr = 0;
      else if (*imgRawCroppedPtr >= thrH)
        *imgRawCroppedUINT8Ptr = 255;
      else {
        *imgRawCroppedUINT8Ptr = (mylib::uint8)floor(
            (255.0 * (((double)((*imgRawCroppedPtr) - thrL)) / rangeLH)));
      }
      imgRawCroppedUINT8Ptr++;
      imgRawCroppedPtr++;
    }

    // save raw data
    string imgRawCroppedFilename(imgBasename + "Raw_" + itoa + ".tif");
    if (mylib::Write_Image((char*)(imgRawCroppedFilename.c_str()),
                           imgRawCroppedUINT8, mylib::DONT_PRESS) == 1) {
      cout << "ERROR: at "
              "lineageHyperTree::debugCanvasFromSegmentationCroppedRegion: "
              "file "
           << imgRawCroppedFilename << " cannot be written" << endl;
      exit(3);
    }

    // release memory
    mylib::Free_Array(imgRaw);
    mylib::Free_Array(imgRawCropped);
    mylib::Free_Array(imgRawCroppedUINT8);
    mylib::Free_Histogram(h);
  }

  /*
  mylib::Free_Array(beg);
  mylib::Free_Array(end);
  mylib::Free_Array(begL);
  mylib::Free_Array(endL);
  */

  // transform cropped image into UINT8 RGB canvas
  mylib::Value val0, val1;
  val0.uval = 0;
  val1.uval = 1;
  mylib::Histogram* h = mylib::Histogram_Array(imgCropped, 0x10000, val1, val0);
  h->total -= h->counts[0];  // we do not want to count the background values in
                             // the statistics
  h->counts[0] = 0;
  unsigned int thrH = mylib::Percentile2Value(h, betaThr).uval;
  unsigned int thrL = mylib::Percentile2Value(h, 1.0 - betaThr).uval;

  mylib::Array* canvas = Make_Array(mylib::RGB_KIND, mylib::UINT8_TYPE,
                                    dimsImage, imgCropped->dims);
  mylib::uint8* canvasPtr = (mylib::uint8*)(canvas->data);
  mylib::Array* imgUINT8 = Make_Array(mylib::PLAIN_KIND, mylib::UINT8_TYPE,
                                      dimsImage, imgCropped->dims);
  mylib::uint8* imgUINT8ptr = (mylib::uint8*)(imgUINT8->data);
  double rangeLH = (double)(thrH - thrL);
  for (int cc = 0; cc < 3; cc++)  // the same for loop per color (we could speed
                                  // it up by memcpy to channels, but this is
                                  // just ad ebugging function)
  {
    mylib::uint16* imPtr = (mylib::uint16*)(imgCropped->data);
    for (mylib::Size_Type ii = 0; ii < imgCropped->size; ii++) {
      if (*imPtr <= thrL)
        *canvasPtr = 0;
      else if (*imPtr >= thrH)
        *canvasPtr = 255;
      else {
        *canvasPtr = (mylib::uint8)floor(
            (255.0 * (((double)((*imPtr) - thrL)) / rangeLH)));
      }

      if (cc == 0) {
        *imgUINT8ptr = *canvasPtr;
        imgUINT8ptr++;
      }

      canvasPtr++;
      imPtr++;
    }
  }
  mylib::Free_Histogram(h);

  // remap labels in imgLCropped
  map<mylib::uint16, mylib::uint16> mapLabels;
  mapLabels.insert(pair<mylib::uint16, mylib::uint16>(0, 0));
  pair<map<mylib::uint16, mylib::uint16>::iterator, bool> iterM;
  iterM.first = mapLabels.begin();
  pair<mylib::uint16, mylib::uint16> valM(
      0, 1);  // second stores the number of labels
  for (mylib::Size_Type ii = 0; ii < imgLCropped->size; ii++) {
    valM.first = imgLptr[ii];
    iterM = mapLabels.insert(valM);
    if (iterM.second == true)  // a new element was inserted
      valM.second++;
  }
  int numLabels = valM.second;
  for (mylib::Size_Type ii = 0; ii < imgLCropped->size; ii++) {
    imgLptr[ii] = mapLabels[imgLptr[ii]];
  }

  // generate alpha blending
  // mylib::Partition *P = mylib::Make_Partition (imgCropped, imgLCropped,
  // numLabels,0, 1); //my supervoxels might not be 2^n+1 connected -> Mylib
  // code crashes
  // canvas = mylib::Draw_Partition (canvas, P, alpha);
  alphaBlend(canvas, imgLCropped, alpha);

  // save images
  if (mylib::Write_Image((char*)(imgRawFilename.c_str()), imgUINT8,
                         mylib::DONT_PRESS) == 1) {
    cout << "ERROR: at "
            "lineageHyperTree::debugCanvasFromSegmentationCroppedRegion: file "
         << imgRawFilename << " cannot be written" << endl;
    exit(3);
  }
  if (mylib::Write_Image((char*)(canvasFilename.c_str()), canvas,
                         mylib::DONT_PRESS) == 1) {
    cout << "ERROR: at "
            "lineageHyperTree::debugCanvasFromSegmentationCroppedRegion: file "
         << canvasFilename << " cannot be written" << endl;
    exit(3);
  }

  // release memory
  mylib::Free_Array(imgCropped);
  mylib::Free_Array(imgLCropped);
  // mylib::Free_Slice(imgSlice);
  // mylib::Free_Slice(imgLSlice);
  mylib::Free_Array(imgL);
  mylib::Free_Array(imgUINT8);
  mylib::Free_Array(canvas);

  // mylib::Reset_Slice();
  mylib::Reset_Array();
  // mylib::Free_Partition (P);
  return;
}

// from mylib::water.shed.c
static double red[3] = {1., 0., 0.};
static double green[3] = {0., 1., 0.};
static double blue[3] = {0., 0., 1.};
static double yellow[3] = {1., 1., 0.};
static double cyan[3] = {0., 1., 1.};
static double magenta[3] = {1., 0., 1.};
static double orange[3] = {1., .5, 0.};
static double brown[3] = {1., .25, .5};
static double* palette[8] = {magenta, red,    green, yellow,
                             cyan,    orange, brown, blue};

void lineageHyperTree::alphaBlend(mylib::Array* imgRGB, mylib::Array* imgLabels,
                                  double alpha) {
  if (imgRGB->kind != mylib::RGB_KIND) {
    cout
        << "ERROR: at lineageHyperTree::alphaBlend: imgRGB needs to be RGB type"
        << endl;
    exit(3);
  }
  for (mylib::Dimn_Type ii = 0; ii < imgLabels->ndims; ii++) {
    if (imgRGB->ndims - 1 != imgLabels->ndims ||
        imgRGB->dims[ii] != imgLabels->dims[ii]) {
      cout << imgRGB->ndims << " " << imgLabels->ndims << " ; "
           << imgRGB->dims[ii] << " " << imgLabels->dims[ii] << endl;
      cout << "ERROR: at lineageHyperTree::alphaBlend: imgRGB and imgLabels "
              "need to be of teh same dimensions"
           << endl;
      exit(3);
    }
  }
  if (imgRGB->type != mylib::UINT8_TYPE) {
    cout << "ERROR: at lineageHyperTree::alphaBlend: imgRGB has to be UINT8"
         << endl;
    exit(3);
  }
  if (imgLabels->type != mylib::UINT16_TYPE) {
    cout << "ERROR: at lineageHyperTree::alphaBlend: imgLabels has to be UINT16"
         << endl;
    exit(3);
  }

  mylib::Size_Type span = 1;
  for (mylib::Dimn_Type ii = 0; ii < imgLabels->ndims; ii++)
    span *= imgLabels->dims[ii];
  mylib::Size_Type spanSlice =
      imgLabels->dims[0];  // to calculate if pixel is a border
  double beta = 1.0 - alpha;

  mylib::uint16* imgLptr = (mylib::uint16*)(imgLabels->data);
  mylib::uint8* imgRptr = ((mylib::uint8*)(imgRGB->data));
  mylib::uint8* imgGptr = imgRptr + span;
  mylib::uint8* imgBptr = imgGptr + span;

  int color;
  for (mylib::Size_Type ii = spanSlice; ii < imgLabels->size - spanSlice;
       ii++)  // so I do not have to check for out of bounds 2 pixels
  {
    if (imgLptr[ii] == 0)  // background
      continue;

    // check if pixel is edge of element (using 4-connectivity in 2D): it is
    // just for debugging
    if (imgLptr[ii + 1] != imgLptr[ii] || imgLptr[ii - 1] != imgLptr[ii] ||
        imgLptr[ii + spanSlice] != imgLptr[ii] ||
        imgLptr[ii - spanSlice] != imgLptr[ii]) {
      imgRptr[ii] = 0;
      imgGptr[ii] = 255;
      imgBptr[ii] = 0;
    } else {                    // blend pixel since it is not an edge color
      color = imgLptr[ii] % 8;  // to select color in the palette
      // cout<< ((double)(imgRptr[ii])) * beta + 255.0 * alpha *
      // palette[color][0] <<endl;
      imgRptr[ii] = mylib::uint8(((double)(imgRptr[ii])) * beta +
                                 255.0 * alpha * palette[color][0]);
      imgGptr[ii] = mylib::uint8(((double)(imgGptr[ii])) * beta +
                                 255.0 * alpha * palette[color][1]);
      imgBptr[ii] = mylib::uint8(((double)(imgBptr[ii])) * beta +
                                 255.0 * alpha * palette[color][2]);
    }
  }

  return;
}

void lineageHyperTree::debugPrintLineageToMCMCxmlFile(string filename,
                                                      string imgPrefix,
                                                      string imgSuffix) {
  ofstream fout(filename.c_str());
  if (fout.is_open() == false) {
    cout << "ERROR: at lineageHyperTree::debugPrintLineageToMCMCxmlFile: file "
         << filename << " could not be opened" << endl;
    exit(3);
  }

  // set id of the nucleus so we can use it later to reconstuct lineage

  for (unsigned int frame = 0; frame < maxTM; frame++) {
    int countN = 0;
    for (list<nucleus>::iterator iterN = nucleiList[frame].begin();
         iterN != nucleiList[frame].end(); ++iterN) {
      iterN->tempWilcard = (float)countN;
      countN++;
    }
  }

  GaussianMixtureModelRedux::writeXMLheader(fout);
  fout << "<Stack numMoves=\"0\" maxLabel=\"" << lineagesList.size() << "\">"
       << endl;

  int offsetTM = -1;  // in case it is a sublineage, and TM>0 for frame = 0; I
                      // need to offset parentId and childrenId
  for (unsigned int frame = 0; frame < maxTM; frame++) {
    char itoaB[16];
    sprintf(itoaB, "%.5d", frame);
    string itoa(itoaB);
    string imgFilename(imgPrefix + itoa + imgSuffix + itoa + ".tif");
    // check if file exists
    ifstream fin(imgFilename.c_str());
    if (fin.is_open() == false) {
      cout << "ERROR: at lineageHyperTree::debugPrintLineageToMCMCxmlFile: "
              "image file "
           << imgFilename << " does not exist" << endl;
      exit(4);
    }
    fin.close();

    if (offsetTM < 0 && nucleiList[frame].empty() == false)  // set offset
      offsetTM = nucleiList[frame].begin()->TM - frame;

    fout << "<Frame id=\"" << frame << "\" dims=\"" << dimsImage
         << "\" imgFilename=\"" << imgFilename << "\">" << endl;

    mylib::uint32 parentId[2];
    mylib::uint32 childrenIdLeft[2];
    mylib::uint32 childrenIdRight[2];
    int numChildren;
    for (list<nucleus>::iterator iterN = nucleiList[frame].begin();
         iterN != nucleiList[frame].end(); ++iterN) {
      fout << "<Blob id=\"" << ((int)(iterN->tempWilcard)) << "\" dims=\""
           << dimsImage << "\" center=\"";
      for (int ii = 0; ii < dimsImage; ii++) fout << iterN->centroid[ii] << " ";
      fout << "\" scale=\"";
      for (int ii = 0; ii < dimsImage; ii++)
        fout << supervoxel::getScale()[ii] << " ";
      fout << "\" frame=\"" << frame << "\" intensity=\"" << iterN->avgIntensity
           << "\" neigh=\"";
      // we use teh neighbors to write the id of each supervoxel that belongs to
      // this nucleus. We save the first element of pixelIdxList
      for (vector<ChildrenTypeNucleus>::iterator iterS =
               iterN->treeNode.getChildren().begin();
           iterS != iterN->treeNode.getChildren().end(); ++iterS) {
        if ((*iterS)->PixelIdxList.empty() == false)
          fout << (*iterS)->PixelIdxList[0]
               << " 0 ";  // all the codes expect neigh to have an even number
                          // of elements, so we need to add a bogus number of
                          // zeros
      }
      fout << "\">" << endl;

      // write surface (shape)
      fout << "<Surface name=\"Ellipsoid\" id=\"1\" numCoeffs=\""
           << (dimsImage + (dimsImage * (1 + dimsImage) / 2)) << "\" coeffs=\"";
      fout << "0.05 0.0 0.000000 0.05 0.000000 0.3 ";  // TODO: calculate
                                                       // covariance matrix
      for (int ii = 0; ii < dimsImage; ii++) fout << iterN->centroid[ii] << " ";
      fout << "\" covarianceMatrixSize=\"" << dimsImage << "\" > ";
      fout << "</Surface>" << endl;

      // write solution (lineage)
      if (iterN->treeNodePtr->parent == NULL) {
        parentId[0] = 4294967295;
        parentId[1] = 0;
      } else {
        parentId[0] = iterN->treeNodePtr->parent->data->TM - offsetTM;
        parentId[1] =
            (mylib::uint32)(iterN->treeNodePtr->parent->data
                                ->tempWilcard);  // we set it up before
      }
      numChildren = 0;
      if (iterN->treeNodePtr->left == NULL) {
        childrenIdLeft[0] = 4294967295;
        childrenIdLeft[1] = 0;
      } else {
        numChildren++;
        childrenIdLeft[0] = iterN->treeNodePtr->left->data->TM - offsetTM;
        childrenIdLeft[1] =
            (mylib::uint32)(iterN->treeNodePtr->left->data
                                ->tempWilcard);  // we set it up before
      }
      if (iterN->treeNodePtr->right == NULL) {
        childrenIdRight[0] = 4294967295;
        childrenIdRight[1] = 0;
      } else {
        numChildren++;
        childrenIdRight[0] = iterN->treeNodePtr->right->data->TM - offsetTM;
        childrenIdRight[1] =
            (mylib::uint32)(iterN->treeNodePtr->right->data
                                ->tempWilcard);  // we set it up before
      }

      fout << "<BlobSolution score=\"" << -1e32 << "\" label=\"3\" parentIdx=\""
           << parentId[0] << " " << parentId[1]
           << "\" ";  // TODO: calculate split score
      if (numChildren == 1) {
        fout << "childrenIdx=\"";
        if (childrenIdLeft[0] == 4294967295) {
          fout << childrenIdRight[0] << " " << childrenIdRight[1] << "\"";
        } else {
          fout << childrenIdLeft[0] << " " << childrenIdLeft[1] << "\"";
        }
      } else if (numChildren == 2) {
        fout << "childrenIdx=\"";
        fout << childrenIdLeft[0] << " " << childrenIdLeft[1] << " "
             << childrenIdRight[0] << " " << childrenIdRight[1] << "\"";
      }
      fout << "></BlobSolution>" << endl;

      // repeat blob solution
      fout << "<BlobSolution score=\"" << -1e32 << "\" label=\"3\" parentIdx=\""
           << parentId[0] << " " << parentId[1]
           << "\" ";  // TODO: calculate split score
      if (numChildren == 1) {
        fout << "childrenIdx=\"";
        if (childrenIdLeft[0] == 4294967295) {
          fout << childrenIdRight[0] << " " << childrenIdRight[1] << "\"";
        } else {
          fout << childrenIdLeft[0] << " " << childrenIdLeft[1] << "\"";
        }
      } else if (numChildren == 2) {
        fout << "childrenIdx=\"";
        fout << childrenIdLeft[0] << " " << childrenIdLeft[1] << " "
             << childrenIdRight[0] << " " << childrenIdRight[1] << "\"";
      }
      fout << "></BlobSolution>" << endl;

      // close blob element
      fout << "</Blob>" << endl;
    }

    fout << "</Frame>" << endl;
  }
  fout << "</Stack>" << endl;
  GaussianMixtureModelRedux::writeXMLfooter(fout);
  fout.close();
}

//================================================================================
// extracts a sublineage starting at vecRoot[ii] for lengthTM time points
int lineageHyperTree::cutSublineage(vector<rootSublineage>& vecRoot,
                                    lineageHyperTree& lht) {
  lht.clear();
  lht.setIsSublineage(true);
  // lht.setRootSublineage(vecRoot);//done by cutSingleSublineageFromRoot

  // create a lineage for each root
  int err = 0;
  for (vector<rootSublineage>::iterator iter = vecRoot.begin();
       iter != vecRoot.end(); ++iter) {
    err = lht.cutSingleSublineageFromRoot(*iter);
    if (err > 0) return err;
  }

  return 0;
}

int lineageHyperTree::cutSingleSublineageFromRoot(rootSublineage& root) {
  if (isSublineage == false) {
    cout << "ERROR: at lineageHyperTree::cutSingleSublineageFromRoot: you "
            "cannot cut sublineage into a nonSublineage graph"
         << endl;
    return (3);
  }

  if (root == NULL) return 0;
  // check if root has al ready been added
  for (vector<rootSublineage>::iterator iter = sublineageRootVec.begin();
       iter != sublineageRootVec.end(); ++iter) {
    if ((*iter) == root) {
      // cout<<"WARNING: at lineageHyperTree::cutSingleSublineageFromRoot: root
      // had already been added to this sublineage"<<endl;
      return 0;
    }
  }

  // add root to root vector
  sublineageRootVec.push_back(root);
  // generate new lineage
  lineagesList.push_back(lineage());
  list<lineage>::iterator iterL =
      (++(lineagesList.rbegin())).base();  // iterator to last element added

  int iniTM = root->data->TM;

  // first copy just sublineage
  iterL->bt.SetMainRoot(CopyPartialLineage(
      root, NULL,
      maxTM + iniTM));  // maxTM indicates the length of the sublineage
  iterL->bt.reset();    // current = root

  // now we have a sublineage but data points to the elements of the "parent"
  // graph. We have to make a copy of those objects and modify data
  queue<TreeNode<ChildrenTypeLineage>*> q;
  q.push(iterL->bt.pointer_current());

  TreeNode<ChildrenTypeLineage>* auxQ;
  while (q.empty() == false) {
    auxQ = q.front();
    q.pop();

    // check if we should add daughters to the sublineage
    if ((auxQ->left != NULL)) q.push(auxQ->left);
    if ((auxQ->right != NULL)) q.push(auxQ->right);

    int effectiveTM = auxQ->data->TM - iniTM;

    // add current point in the lineage to
    nucleiList[effectiveTM].push_back(*(auxQ->data));  // add nucleus
    list<nucleus>::iterator iterN =
        (++(nucleiList[effectiveTM].rbegin()))
            .base();  // iter pointing to the added nucleus
    iterN->treeNode.reset();
    // add all the supervoxels
    for (vector<ChildrenTypeNucleus>::iterator iter =
             auxQ->data->treeNode.getChildren().begin();
         iter != auxQ->data->treeNode.getChildren().end(); ++iter) {
      supervoxelsList[effectiveTM].push_back(*(*(iter)));  // I don't need to
                                                           // reset treeNode for
                                                           // supervoxels since
                                                           // we only care about
                                                           // parent and it will
                                                           // be addressed here
      iterN->treeNode.addChild(
          (++(supervoxelsList[effectiveTM].rbegin())).base());
      supervoxelsList[effectiveTM].back().treeNode.setParent(iterN);
    }
    // update data in the lineage to point to the right nucleus
    auxQ->data = iterN;
    iterN->treeNode.setParent(iterL);
    iterN->treeNodePtr = auxQ;
  }

  return 0;
}

//=========================================================================================================================
TreeNode<ChildrenTypeLineage>* lineageHyperTree::CopyPartialLineage(
    TreeNode<ChildrenTypeLineage>* root, TreeNode<ChildrenTypeLineage>* parent,
    int boundsTM) {
  if (root == NULL || root->data->TM >= boundsTM)  // base case - if the node
                                                   // doesn't exist, return NULL
                                                   // or if it is beyond our
                                                   // interest
    return NULL;
  TreeNode<ChildrenTypeLineage>* tmp =
      new TreeNode<ChildrenTypeLineage>;  // make a new location in memory
  tmp->data = root->data;                 // make a copy of the node's data
  tmp->parent = parent;                   // set the new node's parent
  tmp->left = CopyPartialLineage(root->left, tmp,
                                 boundsTM);  // copy the left subtree of the
                                             // current node. pass the current
                                             // node as the subtree's parent
  tmp->right = CopyPartialLineage(
      root->right, tmp, boundsTM);  // do the same with the right subtree
  return tmp;  // return a pointer to the newly created node.
};

//=========================================================================================================================
int lineageHyperTree::pasteOpenEndSublineage(lineageHyperTree& lhtOrig) {
  if (isSublineage == false) {
    cout << "ERROR: at lineageHyperTree::pasteOpenEndSublineage: the hypertree "
            "has to be a sublineage"
         << endl;
    return (2);
  }
  if (sublineageRootVec.size() != lineagesList.size()) {
    cout << "ERROR: at lineageHyperTree::pasteOpenEndSublineage: the number of "
            "roots has to be equal to the number of sublineages to paste"
         << endl;
    return (3);
  }

  int err = 0;
  for (size_t ii = 0; ii < sublineageRootVec.size(); ii++) {
    err = pasteSingleOpenEndSublineageFromRoot(lhtOrig, ii);
    if (err > 0) return err;
  }

  return 0;
}

// TODO: be smarter and instead of reallocating and allocating so much memory,
// reuse resources. Keep some sort of bookkeeping for each time point of deleted
// nuclei, supervoxels, etc
int lineageHyperTree::pasteSingleOpenEndSublineageFromRoot(
    lineageHyperTree& lhtOrig, size_t pos) {
  if (pos < 0 || pos >= sublineageRootVec.size()) {
    cout << "ERROR: at lineageHyperTree::pasteSingleOpenEndSublineageFromRoot: "
            "requested sublineage is too outside the bounds"
         << endl;
    return (2);
  }

  // collect pointers to original lineage and sublineage
  rootSublineage root = sublineageRootVec[pos];
  list<lineage>::iterator iterSubL = lineagesList.begin();
  size_t count = 0;
  for (; iterSubL != lineagesList.end(); ++iterSubL) {
    if (count == pos) break;
    count++;
  }

  int iniTM = root->data->TM;

  // eliminate all the elements from the root down
  queue<TreeNode<ChildrenTypeLineage>*> q;
  q.push(root);

  // decide where to "paste" lineage before deleting them
  TreeNode<ChildrenTypeLineage>* anchor =
      root->parent;  // so we know where to start pasting teh new lineage
  int pastePos;      // 0->we need to create new lineage; 1->paste on the left
  // child; 2->paste on the right child (in case cut right after
  // a split)
  list<lineage>::iterator iterOrigL =
      root->data->treeNode.getParent();  // pointer to original lineage so we
                                         // can update info in new nuclei
  if (anchor == NULL) {
    pastePos = 0;
  } else if (anchor->left == root) {
    pastePos = 1;
    anchor->left = NULL;  // detach branch
  } else if (anchor->right == root) {
    pastePos = 2;
    anchor->right = NULL;  // detach branch
  } else {
    cout << "ERROR: at lineageHyperTree::pasteSingleOpenEndSublineageFromRoot: "
            "children(parent(root)) != root. There is an error in the binary "
            "tree"
         << endl;
    return (5);
  }

  TreeNode<ChildrenTypeLineage>* auxQ;
  ChildrenTypeLineage iterN;
  int auxTM;
  while (q.empty() == false) {
    auxQ = q.front();
    q.pop();

    // check if we should add daughters to the sublineage
    if ((auxQ->left != NULL)) q.push(auxQ->left);
    if ((auxQ->right != NULL)) q.push(auxQ->right);

    // check that it is true that it can be an open end lineage
    unsigned int effectiveTM = auxQ->data->TM - iniTM;
    if (effectiveTM >= maxTM) {
      cout << "ERROR: at "
              "lineageHyperTree::pasteSingleOpenEndSublineageFromRoot: the "
              "original lineage continues further than expected. Thus, you "
              "cannot use an open end paste operation"
           << endl;
      return (3);
    }

    // remove nuclei and supervoxels attached to the node
    iterN = auxQ->data;
    auxTM = iterN->TM;
    for (vector<ChildrenTypeNucleus>::iterator iter =
             iterN->treeNode.getChildren().begin();
         iter != iterN->treeNode.getChildren().end(); ++iter) {
      lhtOrig.supervoxelsList[auxTM].erase(*iter);
    }
    lhtOrig.nucleiList[auxTM].erase(iterN);

    // remove node from the lineage
    if (iterOrigL->bt.pointer_mainRoot() == auxQ)  // main root is being deleted
    {
      delete auxQ;
      iterOrigL->bt.SetMainRootToNULL();
    } else {
      delete auxQ;
    }
  }

  // add new lineage starting from anchor
  // first copy just sublineage
  root = CopyPartialLineage(iterSubL->bt.pointer_mainRoot(), anchor,
                            maxTM + iniTM);
  switch (pastePos) {
    case 0:
      // we do not need to attach root to any anchor in this case
      iterOrigL->bt.SetMainRoot(root);
      break;
    case 1:
      // root = CopyPartialLineage(iterSubL->bt.pointer_mainRoot(),anchor, maxTM
      // + iniTM);
      anchor->left = root;
      break;
    case 2:
      // root = CopyPartialLineage(iterSubL->bt.pointer_mainRoot(), anchor,
      // maxTM + iniTM);
      anchor->right = root;
      break;
    default:
      cout << "ERROR: at "
              "lineageHyperTree::pasteSingleOpenEndSublineageFromRoot: "
              "children(parent(root)) != root. There is an error in the binary "
              "tree"
           << endl;
      return (5);
      break;
  }
  iterOrigL->bt.SetCurrent(
      root);  // just in case current was set in one of the deleted elements

  // now I need to add supervoxel and nuclei and alter the data in each tree
  // node
  queue<TreeNode<ChildrenTypeLineage> *> qOrig, qSubL;
  qOrig.push(root);
  qSubL.push(iterSubL->bt.pointer_mainRoot());

  TreeNode<ChildrenTypeLineage> *auxQorig, *auxQsubL;
  while (qOrig.empty() == false) {
    auxQorig = qOrig.front();
    auxQsubL = qSubL.front();
    qOrig.pop();
    qSubL.pop();

    // check if we should add daughters to the sublineage
    if ((auxQsubL->left != NULL)) {
      qSubL.push(auxQsubL->left);
      qOrig.push(auxQorig->left);
    }
    if ((auxQsubL->right != NULL)) {
      qSubL.push(auxQsubL->right);
      qOrig.push(auxQorig->right);
    }
    int origTM =
        auxQsubL->data
            ->TM;  // sublineage preserves the original TM, so it should be fine

    // add current point in the lineage to
    nucleiList[origTM].push_back(*(auxQsubL->data));  // add nucleus
    list<nucleus>::iterator iterN =
        (++(nucleiList[origTM].rbegin()))
            .base();  // iter pointing to the added nucleus
    iterN->treeNode.reset();
    // add all the supervoxels
    for (vector<ChildrenTypeNucleus>::iterator iter =
             auxQsubL->data->treeNode.getChildren().begin();
         iter != auxQsubL->data->treeNode.getChildren().end(); ++iter) {
      supervoxelsList[origTM].push_back(*(*(iter)));  // I don't need to reset
                                                      // treeNode for
                                                      // supervoxels since we
                                                      // only care about parent
                                                      // and it will be
                                                      // addressed here
      iterN->treeNode.addChild((++(supervoxelsList[origTM].rbegin())).base());
      supervoxelsList[origTM].back().treeNode.setParent(iterN);
    }
    // update data in the lineage to point to the right nucleus
    auxQorig->data = iterN;
    iterN->treeNode.setParent(iterOrigL);
    iterN->treeNodePtr = auxQorig;
  }

  return 0;
}

//================================================================================
int lineageHyperTree::cutSublineageCellDeathDivisionEvents(
    unsigned int winRadiusTime, unsigned int TM,
    vector<lineageHyperTree>& lhtVec) {
  lhtVec.clear();

  if (TM >= maxTM) return 0;

  // check for dead / splitting cells in current time point
  int minTM = TM - winRadiusTime;  // find the root at TM-winRadiusTime

  // we need to preallocate ahead of time. Otherwise with dynamic allocation,
  // pointers are shifted around and data structures are useless
  vector<TreeNode<ChildrenTypeLineage>*> vecIter;
  for (list<nucleus>::const_iterator iter = nucleiList[TM].begin();
       iter != nucleiList[TM].end(); ++iter) {
    if (iter->treeNodePtr->getNumChildren() != 1) {
      // lhtVec.push_back (lineageHyperTree(2 * winRadiusTime + 1));
      // lhtVec.back().setIsSublineage(true);

      vecIter.push_back(iter->treeNodePtr);
    }
  }

  // allocate memory: IF YOU DYNAMICALLY CHANGE THE SIZE OF THIS ARRAY AFTER THE
  // OPERATIONS IT WILL GIVE WRONG RESULTS
  lhtVec.reserve(vecIter.size() * 2);  // we reserve twice the space needed just
                                       // in case, so there won't be
                                       // reallocation

  // cut sublineages
  for (vector<TreeNode<ChildrenTypeLineage>*>::iterator iter = vecIter.begin();
       iter != vecIter.end(); ++iter) {
    lhtVec.push_back(lineageHyperTree(2 * winRadiusTime + 1));
    lhtVec.back().setIsSublineage(true);

    queue<TreeNode<ChildrenTypeLineage>*> q;
    q.push(*iter);
    cutSublineageCellDeathDivisionEventsRecursive(minTM, lhtVec, q);

    if (lhtVec.back().lineagesList.empty() == true)  // nothing was added
      lhtVec.pop_back();
  }
  // recalculate nearest neighbors for sublineages
  // Not really necessary. I can do it later if I think I need it

  return 0;
}

int lineageHyperTree::cutSublineageCellDeathDivisionEventsRecursive(
    int minTM, vector<lineageHyperTree>& lhtVec,
    queue<TreeNode<ChildrenTypeLineage>*>& q) {
  if (q.empty() == true) return 0;
  TreeNode<ChildrenTypeLineage>* anchor = q.front();
  q.pop();

  // find root upstream
  TreeNode<ChildrenTypeLineage>* root = findRoot(anchor, minTM);

  // check if root has been previously selected
  bool exist = false;
  for (vector<lineageHyperTree>::iterator iterL = lhtVec.begin();
       iterL != lhtVec.end(); ++iterL) {
    if (iterL->findRootSublineage(root) >=
        0)  // returns -1 if root is not found
    {
      exist = true;
      break;
    }
  }

  // if it does not exist we can add to the LAST sublineage in lhtVec
  if (exist == false) {
    int err = lhtVec.back().cutSingleSublineageFromRoot(root);
    if (err > 0) return err;
    // check if there are more elements to be added to the queue: traverse the
    // ORIGINAL lineage looking for cell death / split and add neighbors
    int endTM = root->data->TM + lhtVec.back().getMaxTM();
    queue<TreeNode<ChildrenTypeLineage>*> qL;
    qL.push(root);
    TreeNode<ChildrenTypeLineage>* aux;
    ChildrenTypeLineage auxIterN;
    while (qL.empty() == false) {
      aux = qL.front();
      qL.pop();

      if (aux != NULL && aux->data->TM < (endTM - 1))  // endTM -1 because we do
                                                       // not want to look into
                                                       // the last element: all
                                                       // could be dead just
                                                       // because we are at the
                                                       // forefront of tracking
      {
        if (aux->left != NULL) qL.push(aux->left);
        if (aux->right != NULL) qL.push(aux->right);

        if (aux->getNumChildren() != 1) {
          // add element
          q.push(aux);  // I NEED TO ADD POINTERS IN THE ORIGINAL LINEAGE!!! NOT
                        // IN THE SUBLINEAGE: OTHERWISE ROOTS ARE WRONG!!!
          // add nearest neighbor to the queue
          float dist =
              findNearestNucleusNeighborInSpaceEuclideanL2(aux->data, auxIterN);
          if (dist < 1e31)                  // nearest neighbor in space found
            q.push(auxIterN->treeNodePtr);  // I NEED TO ADD POINTERS IN THE
                                            // ORIGINAL LINEAGE!!! NOT IN THE
                                            // SUBLINEAGE: OTHERWISE ROOTS ARE
                                            // WRONG!!!
        }
      }
    }
  }

  while (q.empty() == false) {
    int err = cutSublineageCellDeathDivisionEventsRecursive(minTM, lhtVec,
                                                            q);  // recursive
    if (err > 0) return err;
  }

  return 0;
}

//=======================================================================================
float lineageHyperTree::findNearestNucleusNeighborInSpaceEuclideanL2(
    const ChildrenTypeLineage& iterNucleus,
    ChildrenTypeLineage&
        iterNucleusNN)  // uses supervoxels to find nearest neighbor
{
  float dist = 1e32f, auxDist;

  for (vector<ChildrenTypeNucleus>::iterator iter =
           iterNucleus->treeNode.getChildren().begin();
       iter != iterNucleus->treeNode.getChildren().end();
       ++iter)  // explore over all possible supoervoxels for this nuclei
  {
    for (vector<ChildrenTypeNucleus>::iterator iterS =
             (*iter)->nearestNeighborsInSpace.begin();
         iterS != (*iter)->nearestNeighborsInSpace.end();
         ++iterS)  // check all nearest neighbors in space
    {
      if ((*iterS)->treeNode.hasParent() == true) {
        auxDist = iterNucleus->Euclidean2Distance(
            *((*iterS)->treeNode.getParent()), (*iterS)->getScale());
        if (auxDist < dist &&
            auxDist > 1e-3f)  // below minimum and it is not the same nucleus
        {
          dist = auxDist;
          iterNucleusNN = (*iterS)->treeNode.getParent();
        }
      }
    }
  }

  return dist;
}

//=======================================================================================
int lineageHyperTree::findKNearestNucleiNeighborInSpaceEuclideanL2(
    const ChildrenTypeLineage& iterNucleus,
    vector<ChildrenTypeLineage>& iterNucleusNNvec,
    vector<float>& distVec)  // uses supervoxels to find K nearest neighbor
{
  unsigned int K = iterNucleusNNvec.size();
  if (K == 0) {
    cout << "WARNING: at "
            "lineageHyperTree::findKNearestNucleiNeighborInSpaceEuclideanL2: "
            "vector size is 0"
         << endl;
    return 0;
  }
  if (K != distVec.size()) {
    cout << "ERROR: at "
            "lineageHyperTree::findKNearestNucleiNeighborInSpaceEuclideanL2: "
            "vector sizes are not the same"
         << endl;
    return 1;
  }
  // reset distances
  for (vector<float>::iterator iterD = distVec.begin(); iterD != distVec.end();
       ++iterD)
    (*iterD) = 1e32f;

  float auxDist;
  int pos;
  for (vector<ChildrenTypeNucleus>::iterator iter =
           iterNucleus->treeNode.getChildren().begin();
       iter != iterNucleus->treeNode.getChildren().end();
       ++iter)  // explore over all possible supervoxels for this nuclei
  {
    for (vector<ChildrenTypeNucleus>::iterator iterS =
             (*iter)->nearestNeighborsInSpace.begin();
         iterS != (*iter)->nearestNeighborsInSpace.end();
         ++iterS)  // check all nearest neighbors in space
    {
      if ((*iterS)->treeNode.hasParent() == true) {
        auxDist = iterNucleus->Euclidean2Distance(
            *((*iterS)->treeNode.getParent()), (*iterS)->getScale());
        // see if we can iserted into the array in ascending order
        if (auxDist < distVec.back() &&
            auxDist > 1e-3f)  // it is not the same nucleus
        {
          // linear insertion into a sorted array. Since K should be very small
          // this should be fast
          for (pos = K - 1; pos >= 0; pos--) {
            if (distVec[pos] < auxDist) break;
          }
          for (int ii = K - 1; ii > pos + 1; ii--) {
            distVec[ii] = distVec[ii - 1];
            iterNucleusNNvec[ii] = iterNucleusNNvec[ii - 1];
          }
          distVec[pos + 1] = auxDist;
          iterNucleusNNvec[pos + 1] = (*iterS)->treeNode.getParent();
        }
      }
    }
  }

  return 0;
}
//=======================================================================================
int lineageHyperTree::findKNearestNucleiNeighborInTimeForwardEuclideanL2(
    const ChildrenTypeLineage& iterNucleus,
    vector<ChildrenTypeLineage>& iterNucleusNNvec,
    vector<float>& distVec)  // uses supervoxels to find K nearest neighbor
{
  unsigned int K = iterNucleusNNvec.size();
  if (K == 0) {
    cout << "WARNING: at "
            "lineageHyperTree::findKNearestNucleiNeighborInSpaceEuclideanL2: "
            "vector size is 0"
         << endl;
    return 0;
  }
  if (K != distVec.size()) {
    cout << "ERROR: at "
            "lineageHyperTree::findKNearestNucleiNeighborInSpaceEuclideanL2: "
            "vector sizes are not the same"
         << endl;
    return 1;
  }
  // reset distances
  for (vector<float>::iterator iterD = distVec.begin(); iterD != distVec.end();
       ++iterD)
    (*iterD) = 1e32f;

  float auxDist;
  int pos;
  for (vector<ChildrenTypeNucleus>::iterator iter =
           iterNucleus->treeNode.getChildren().begin();
       iter != iterNucleus->treeNode.getChildren().end();
       ++iter)  // explore over all possible supervoxels for this nuclei
  {
    for (vector<ChildrenTypeNucleus>::iterator iterS =
             (*iter)->nearestNeighborsInTimeForward.begin();
         iterS != (*iter)->nearestNeighborsInTimeForward.end();
         ++iterS)  // check all nearest neighbors in space
    {
      if ((*iterS)->treeNode.hasParent() == true) {
        auxDist = iterNucleus->Euclidean2Distance(
            *((*iterS)->treeNode.getParent()), (*iterS)->getScale());
        // see if we can iserted into the array in ascending order

        // linear insertion into a sorted array. Since K should be very small
        // this should be fast
        if (auxDist < distVec.back())  // make sure that at least is closer than
                                       // the furthest NN in the list
        {
          for (pos = K - 1; pos >= 0; pos--) {
            if (distVec[pos] < auxDist) break;
          }
          for (int ii = K - 1; ii > pos + 1; ii--) {
            distVec[ii] = distVec[ii - 1];
            iterNucleusNNvec[ii] = iterNucleusNNvec[ii - 1];
          }
          distVec[pos + 1] = auxDist;
          iterNucleusNNvec[pos + 1] = (*iterS)->treeNode.getParent();
        }
      }
    }
  }

  return 0;
}

//=======================================================================================
int lineageHyperTree::findKNearestNucleiNeighborInSpaceSupervoxelEuclideanL2(
    const ChildrenTypeLineage& iterNucleus,
    vector<ChildrenTypeLineage>& iterNucleusNNvec,
    vector<float>& distVec)  // uses supervoxels to find K nearest neighbor
{
  unsigned int K = iterNucleusNNvec.size();
  if (K == 0) {
    cout << "WARNING: at "
            "lineageHyperTree::findKNearestNucleiNeighborInSpaceEuclideanL2: "
            "vector size is 0"
         << endl;
    return 0;
  }
  if (K != distVec.size()) {
    cout << "ERROR: at "
            "lineageHyperTree::findKNearestNucleiNeighborInSpaceEuclideanL2: "
            "vector sizes are not the same"
         << endl;
    return 1;
  }
  // reset distances
  for (vector<float>::iterator iterD = distVec.begin(); iterD != distVec.end();
       ++iterD)
    (*iterD) = 1e32f;

  float auxDist;
  int pos;
  bool nucleiExist;
  for (vector<ChildrenTypeNucleus>::iterator iter =
           iterNucleus->treeNode.getChildren().begin();
       iter != iterNucleus->treeNode.getChildren().end();
       ++iter)  // explore over all possible supervoxels for this nuclei
  {
    for (vector<ChildrenTypeNucleus>::iterator iterS =
             (*iter)->nearestNeighborsInSpace.begin();
         iterS != (*iter)->nearestNeighborsInSpace.end();
         ++iterS)  // check all nearest neighbors in space
    {
      if ((*iterS)->treeNode.hasParent() == true) {
        if ((*iterS)->treeNode.getParent() ==
            iterNucleus)  // both supervoxels belong to the main nucleus
          continue;

        // auxDist = iterNucleus->Euclidean2Distance(
        // *((*iterS)->treeNode.getParent()), (*iterS)->getScale());
        auxDist = (*iter)->Euclidean2Distance(
            *(*iterS));  // distance between supervoxels

        // see if this nucleus has laready been selected through a different
        // supervoxel
        nucleiExist = false;
        for (unsigned int ii = 0; ii < K; ii++) {
          if (distVec[ii] > 1e31f) break;
          if (iterNucleusNNvec[ii] ==
              (*iterS)->treeNode.getParent())  // this nucleus was already in
                                               // the list
          {
            if (auxDist < distVec[ii])  // I need to update the position of the
                                        // nucleus in the vector
            {
              distVec[ii] = auxDist;
              pos = ii - 1;
              while (pos >= 0 && distVec[pos] > distVec[pos + 1]) {
                // swap elements
                ChildrenTypeLineage auxN = iterNucleusNNvec[pos];
                iterNucleusNNvec[pos] = iterNucleusNNvec[pos + 1];
                iterNucleusNNvec[pos + 1] = auxN;
                float auxD = distVec[pos];
                distVec[pos] = distVec[pos + 1];
                distVec[pos + 1] = auxD;
                // check next element
                pos--;
              }
            }
            nucleiExist = true;
            break;
          }
        }
        if (nucleiExist == true) continue;

        // see if we can iserted into the array in ascending order

        // linear insertion into a sorted array. Since K should be very small
        // this should be fast
        if (auxDist < distVec.back())  // make sure that at least is closer than
                                       // the furthest NN in the list
        {
          for (pos = K - 1; pos >= 0; pos--) {
            if (distVec[pos] < auxDist) break;
          }
          for (int ii = K - 1; ii > pos + 1; ii--) {
            distVec[ii] = distVec[ii - 1];
            iterNucleusNNvec[ii] = iterNucleusNNvec[ii - 1];
          }
          distVec[pos + 1] = auxDist;
          iterNucleusNNvec[pos + 1] = (*iterS)->treeNode.getParent();
        }
      }
    }
  }

  return 0;
}

//=======================================================================================
int lineageHyperTree::
    findKNearestNucleiNeighborInTimeForwardSupervoxelEuclideanL2(
        const ChildrenTypeLineage& iterNucleus,
        vector<ChildrenTypeLineage>& iterNucleusNNvec,
        vector<float>& distVec)  // uses supervoxels to find K nearest neighbor
{
  unsigned int K = iterNucleusNNvec.size();
  if (K == 0) {
    cout << "WARNING: at "
            "lineageHyperTree::findKNearestNucleiNeighborInSpaceEuclideanL2: "
            "vector size is 0"
         << endl;
    return 0;
  }
  if (K != distVec.size()) {
    cout << "ERROR: at "
            "lineageHyperTree::findKNearestNucleiNeighborInSpaceEuclideanL2: "
            "vector sizes are not the same"
         << endl;
    return 1;
  }
  // reset distances
  for (vector<float>::iterator iterD = distVec.begin(); iterD != distVec.end();
       ++iterD)
    (*iterD) = 1e32f;

  float auxDist;
  int pos;
  bool nucleiExist;
  for (vector<ChildrenTypeNucleus>::iterator iter =
           iterNucleus->treeNode.getChildren().begin();
       iter != iterNucleus->treeNode.getChildren().end();
       ++iter)  // explore over all possible supervoxels for this nuclei
  {
    for (vector<ChildrenTypeNucleus>::iterator iterS =
             (*iter)->nearestNeighborsInTimeForward.begin();
         iterS != (*iter)->nearestNeighborsInTimeForward.end();
         ++iterS)  // check all nearest neighbors in space
    {
      if ((*iterS)->treeNode.hasParent() == true) {
        // auxDist = iterNucleus->Euclidean2Distance(
        // *((*iterS)->treeNode.getParent()), (*iterS)->getScale());
        auxDist = (*iter)->Euclidean2Distance(
            *(*iterS));  // distance between supervoxels

        // see if this nucleus has laready been selected through a different
        // supervoxel
        nucleiExist = false;
        for (unsigned int ii = 0; ii < K; ii++) {
          if (distVec[ii] > 1e31f) break;
          if (iterNucleusNNvec[ii] ==
              (*iterS)->treeNode.getParent())  // this nucleus was already in
                                               // the list
          {
            if (auxDist < distVec[ii])  // I need to update the position of the
                                        // nucleus in the vector
            {
              distVec[ii] = auxDist;
              pos = ii - 1;
              while (pos >= 0 && distVec[pos] > distVec[pos + 1]) {
                // swap elements
                ChildrenTypeLineage auxN = iterNucleusNNvec[pos];
                iterNucleusNNvec[pos] = iterNucleusNNvec[pos + 1];
                iterNucleusNNvec[pos + 1] = auxN;
                float auxD = distVec[pos];
                distVec[pos] = distVec[pos + 1];
                distVec[pos + 1] = auxD;
                // check next element
                pos--;
              }
            }
            nucleiExist = true;
            break;
          }
        }
        if (nucleiExist == true) continue;

        // see if we can iserted into the array in ascending order

        // linear insertion into a sorted array. Since K should be very small
        // this should be fast
        if (auxDist < distVec.back())  // make sure that at least is closer than
                                       // the furthest NN in the list
        {
          for (pos = K - 1; pos >= 0; pos--) {
            if (distVec[pos] < auxDist) break;
          }
          for (int ii = K - 1; ii > pos + 1; ii--) {
            distVec[ii] = distVec[ii - 1];
            iterNucleusNNvec[ii] = iterNucleusNNvec[ii - 1];
          }
          distVec[pos + 1] = auxDist;
          iterNucleusNNvec[pos + 1] = (*iterS)->treeNode.getParent();
        }
      }
    }
  }

  return 0;
}

//=============================================
TreeNode<ChildrenTypeLineage>* lineageHyperTree::findRoot(
    TreeNode<ChildrenTypeLineage>* root,
    int minTM)  // goes upstreams the lineage up to minTM to return parent
{
  TreeNode<ChildrenTypeLineage>* aux = root;

  while (aux->data->TM > minTM) {
    if (aux->parent == NULL)  // we reached the top of the lineage
      break;
    else
      aux = aux->parent;
  }

  return aux;
}

//==============================================================================
//=====================nearest neighbor functions===============================
int lineageHyperTree::supervoxelNearestNeighborsInSpace(
    unsigned int TM, unsigned int KmaxNumNN, float KmaxDistKNN,
    int devCUDA)  // you need to setup scale to calculate this properly
{
  supervoxel::setKmaxDistKNN(KmaxDistKNN);
  supervoxel::setKmaxNumNN(KmaxNumNN);

  KmaxNumNN++;  // because one of the nearest neighbors is going to be itself

  if (TM >= maxTM || TM < 0) return 0;

  if (KmaxNumNN > maxKNN) {
    cout << "ERROR: at supervoxelNearestNeighborsInSpace: maximum number of NN "
         << KmaxNumNN << " is superior to maxKNN " << maxKNN
         << ". Please recompile knnCUDA.h code with a larger constant" << endl;
    return 2;
  }

  // preallocate memory for all the centroids from supervoxels
  int ref_nb = supervoxelsList[TM].size();
  long long int query_nb = ref_nb;

  int* ind = new int[KmaxNumNN * query_nb];
  float* dist = new float[KmaxNumNN * query_nb];
  float* xyz = new float[dimsImage * query_nb];  // stores centroids

  // copy centroids to temporary array
  long long int count = 0, offset;
  vector<SibilingTypeSupervoxel> vecIter(
      ref_nb);  // we will need it later to assign ind to nearest neighbor
  for (list<supervoxel>::iterator iter = supervoxelsList[TM].begin();
       iter != supervoxelsList[TM].end(); ++iter) {
    offset = count;
    for (int ii = 0; ii < dimsImage; ii++) {
      xyz[offset] = iter->centroid[ii];
      offset += query_nb;
    }
    vecIter[count] = iter;
    count++;
  }

  // calculate nearest neighbors
  int err = knnCUDA_(ind, dist, xyz, xyz, query_nb, ref_nb, KmaxNumNN,
                     supervoxel::getScale(), devCUDA);
  if (err > 0) return err;

  // parse results to supervoxel structure
  count = 0;
  float auxMaxDist =
      KmaxDistKNN * KmaxDistKNN;  // knNCUDA returns squared distance
  for (list<supervoxel>::iterator iter = supervoxelsList[TM].begin();
       iter != supervoxelsList[TM].end(); ++iter) {
    iter->nearestNeighborsInSpace.clear();
    offset = count;
    for (long long int ii = 0; ii < KmaxNumNN; ii++) {
      if (dist[offset] < auxMaxDist &&
          vecIter[ind[offset]] != iter)  // to avoid selecting itself
      {
        iter->nearestNeighborsInSpace.push_back(vecIter[ind[offset]]);
      }
      offset += query_nb;
    }
    count++;
  }

  // release memory
  delete[] ind;
  delete[] dist;
  delete[] xyz;

  return 0;
}

int lineageHyperTree::supervoxelNearestNeighborsInTimeForward(
    unsigned int TM, unsigned int KmaxNumNN, float KmaxDistKNN,
    int devCUDA)  // you need to setup scale to calculate this properly
{
  supervoxel::setKmaxDistKNN(KmaxDistKNN);
  supervoxel::setKmaxNumNN(KmaxNumNN);

  if (TM >= maxTM - 1 || TM < 0) return 0;

  if (supervoxelsList[TM].empty() == true) return 0;

  if (KmaxNumNN > maxKNN) {
    cout << "ERROR: at supervoxelNearestNeighborsInTimeForward: maximum number "
            "of NN "
         << KmaxNumNN << " is superior to maxKNN " << maxKNN
         << ". Please recompile knnCUDA.h code with a larger constant" << endl;
    return 2;
  }

  // preallocate memory for all the centroids from supervoxels
  int ref_nb = supervoxelsList[TM + 1].size();
  long long int query_nb = supervoxelsList[TM].size();  // we find kNN for each
                                                        // query_nb point with
                                                        // respect to ref_nb

  int* ind = new int[KmaxNumNN * query_nb];
  float* dist = new float[KmaxNumNN * query_nb];
  float* query_xyz = new float[dimsImage * query_nb];  // stores centroids
  float* ref_xyz = new float[dimsImage * ref_nb];      // stores centroids

  // copy centroids to temporary array
  long long int count = 0, offset;
  for (list<supervoxel>::iterator iter = supervoxelsList[TM].begin();
       iter != supervoxelsList[TM].end(); ++iter) {
    offset = count;
    for (int ii = 0; ii < dimsImage; ii++) {
      query_xyz[offset] = iter->centroid[ii];
      offset += query_nb;
    }
    count++;
  }

  count = 0, offset;
  vector<SibilingTypeSupervoxel> vecIter(
      ref_nb);  // we will need it later to assign ind to nearest neighbor
  for (list<supervoxel>::iterator iter = supervoxelsList[TM + 1].begin();
       iter != supervoxelsList[TM + 1].end(); ++iter) {
    offset = count;
    for (int ii = 0; ii < dimsImage; ii++) {
      ref_xyz[offset] = iter->centroid[ii];
      offset += ref_nb;
    }
    vecIter[count] = iter;
    count++;
  }

  // calculate nearest neighbors
  int err = knnCUDA_(ind, dist, query_xyz, ref_xyz, query_nb, ref_nb, KmaxNumNN,
                     supervoxel::getScale(), devCUDA);
  if (err > 0) return err;

  // parse results to supervoxel structure
  count = 0;
  float auxMaxDist =
      KmaxDistKNN * KmaxDistKNN;  // knNCUDA returns squared distance
  for (list<supervoxel>::iterator iter = supervoxelsList[TM].begin();
       iter != supervoxelsList[TM].end(); ++iter) {
    iter->nearestNeighborsInTimeForward.clear();
    offset = count;
    for (long long int ii = 0; ii < KmaxNumNN; ii++) {
      // cout<<offset<<" "<<dist[offset]<<" "<<ind[offset]<<"; Supervoxel:
      // "<<(*(vecIter[ ind[ offset] ]))<<endl;
      if (dist[offset] < auxMaxDist) {
        iter->nearestNeighborsInTimeForward.push_back(vecIter[ind[offset]]);
      }
      offset += query_nb;
    }
    count++;
  }

  // release memory
  delete[] ind;
  delete[] dist;
  delete[] query_xyz;
  delete[] ref_xyz;

  return 0;
}

int lineageHyperTree::supervoxelNearestNeighborsInTimeBackward(
    unsigned int TM, unsigned int KmaxNumNN, float KmaxDistKNN,
    int devCUDA)  // you need to setup scale to calculate this properly
{
  supervoxel::setKmaxDistKNN(KmaxDistKNN);
  supervoxel::setKmaxNumNN(KmaxNumNN);

  if (TM >= maxTM || TM <= 0) return 0;

  if (KmaxNumNN > maxKNN) {
    cout << "ERROR: at supervoxelNearestNeighborsInTimeBackward: maximum "
            "number of NN "
         << KmaxNumNN << " is superior to maxKNN " << maxKNN
         << ". Please recompile knnCUDA.h code with a larger constant" << endl;
    return 2;
  }

  // preallocate memory for all the centroids from supervoxels
  int ref_nb = supervoxelsList[TM - 1].size();
  long long int query_nb = supervoxelsList[TM].size();  // we find kNN for each
                                                        // query_nb point with
                                                        // respect to ref_nb

  int* ind = new int[KmaxNumNN * query_nb];
  float* dist = new float[KmaxNumNN * query_nb];
  float* query_xyz = new float[dimsImage * query_nb];  // stores centroids
  float* ref_xyz = new float[dimsImage * ref_nb];      // stores centroids

  // copy centroids to temporary array
  long long int count = 0, offset;
  for (list<supervoxel>::iterator iter = supervoxelsList[TM].begin();
       iter != supervoxelsList[TM].end(); ++iter) {
    offset = count;
    for (int ii = 0; ii < dimsImage; ii++) {
      query_xyz[offset] = iter->centroid[ii];
      offset += query_nb;
    }
    count++;
  }

  count = 0, offset;
  vector<SibilingTypeSupervoxel> vecIter(
      ref_nb);  // we will need it later to assign ind to nearest neighbor
  for (list<supervoxel>::iterator iter = supervoxelsList[TM - 1].begin();
       iter != supervoxelsList[TM - 1].end(); ++iter) {
    offset = count;
    for (int ii = 0; ii < dimsImage; ii++) {
      ref_xyz[offset] = iter->centroid[ii];
      offset += ref_nb;
    }
    vecIter[count] = iter;
    count++;
  }

  // calculate nearest neighbors
  int err = knnCUDA_(ind, dist, query_xyz, ref_xyz, query_nb, ref_nb, KmaxNumNN,
                     supervoxel::getScale(), devCUDA);
  if (err > 0) return err;

  // parse results to supervoxel structure
  count = 0;
  float auxMaxDist =
      KmaxDistKNN * KmaxDistKNN;  // knNCUDA returns squared distance
  for (list<supervoxel>::iterator iter = supervoxelsList[TM].begin();
       iter != supervoxelsList[TM].end(); ++iter) {
    iter->nearestNeighborsInTimeBackward.clear();
    offset = count;
    for (long long int ii = 0; ii < KmaxNumNN; ii++) {
      // cout<<offset<<" "<<dist[offset]<<" "<<ind[offset]<<"; Supervoxel:
      // "<<(*(vecIter[ ind[ offset] ]))<<endl;
      if (dist[offset] < auxMaxDist)  // to avoid selecting itself
      {
        iter->nearestNeighborsInTimeBackward.push_back(vecIter[ind[offset]]);
      }
      offset += query_nb;
    }
    count++;
  }

  // release memory
  delete[] ind;
  delete[] dist;
  delete[] query_xyz;
  delete[] ref_xyz;

  return 0;
}

//===================================================================================
//=============================metrics===============================================
float lineageHyperTree::offsetJaccardDistance(
    TreeNode<ChildrenTypeLineage>* node) {
  if (node->parent == NULL) return 0;
  if (node->data->treeNode.getNumChildren() == 0 ||
      node->parent->data->treeNode.getNumChildren() == 0)
    return 0;

  // check if we need to merge multiple supervoxels
  supervoxel *svPar, *svCh;
  bool deleteSvPar = true, deleteSvCh = true;
  if (node->data->treeNode.getNumChildren() == 1) {
    svCh = &(*(node->data->treeNode.getChildren().front()));
    deleteSvCh = false;
  } else {
    // merge all superovxels into one
    vector<SibilingTypeSupervoxel>::iterator iterS =
        node->data->treeNode.getChildren().begin();
    svCh = new supervoxel(*(*(iterS)));
    ++iterS;
    vector<supervoxel*> svVec;
    for (; iterS != node->data->treeNode.getChildren().end(); ++iterS) {
      svVec.push_back(&(*(*iterS)));
    }
    svCh->mergeSupervoxels(svVec);
  }

  if (node->parent->data->treeNode.getNumChildren() == 1) {
    svPar = &(*(node->parent->data->treeNode.getChildren().front()));
    deleteSvPar = false;
  } else {
    // merge all superovxels into one
    vector<SibilingTypeSupervoxel>::iterator iterS =
        node->parent->data->treeNode.getChildren().begin();
    svPar = new supervoxel(*(*(iterS)));
    ++iterS;
    vector<supervoxel*> svVec;
    for (; iterS != node->parent->data->treeNode.getChildren().end(); ++iterS) {
      svVec.push_back(&(*(*iterS)));
    }
    svPar->mergeSupervoxels(svVec);
  }

  float d = 1.0 - svPar->JaccardIndexWithOffset(*svCh);
  if (deleteSvCh) delete svCh;
  if (deleteSvPar) delete svPar;

  return d;
}

//=====================================================================================
//=============================statistics=============================================
//--------------------------------------------------------------------
float lineageHyperTree::deathJaccardRatio(
    TreeNode<ChildrenTypeLineage>* rootSplit) const  // checks Jaccard distance
                                                     // ratios to see if one
                                                     // cell has invaded teh
                                                     // space of another after
                                                     // death
{
  if (rootSplit == NULL) {
    cout << "ERROR: at lineageHypertree::deathJaccardRatio: rootSplit node is "
            "null"
         << endl;
    exit(3);
  }
  if (rootSplit->left != NULL || rootSplit->right != NULL) {
    cout << "ERROR: at lineageHypertree::deathJaccardRatio: rootSplit is not a "
            "death"
         << endl;
    exit(2);
  }

  // find nearest neighbor
  ChildrenTypeLineage iterNucleusNN;
  float dist = findNearestNucleusNeighborInSpaceEuclideanL2(rootSplit->data,
                                                            iterNucleusNN);

  if (dist > 1e31) return 1.2f;  // this a value outisde the possible range

  // calculate Jaccarc ratio
  float JNN = 1.0f, JDeath = 1.0f;  // largest distance allowed
  if (iterNucleusNN->treeNodePtr->left != NULL) {
    ChildrenTypeLineage nuc = iterNucleusNN->treeNodePtr->left->data;
    for (vector<ChildrenTypeNucleus>::iterator iter =
             nuc->treeNode.getChildren().begin();
         iter != nuc->treeNode.getChildren().end();
         ++iter)  // check over all supervoxels
    {
      for (vector<ChildrenTypeNucleus>::iterator iter1 =
               iterNucleusNN->treeNode.getChildren().begin();
           iter1 != iterNucleusNN->treeNode.getChildren().end(); ++iter1)
        JNN = min(JNN, (*iter)->JaccardDistance(*(*iter1)));
      for (vector<ChildrenTypeNucleus>::iterator iter1 =
               rootSplit->data->treeNode.getChildren().begin();
           iter1 != rootSplit->data->treeNode.getChildren().end(); ++iter1)
        JDeath = min(JDeath, (*iter)->JaccardDistance(*(*iter1)));
    }
  }
  // repeat for right daughter
  if (iterNucleusNN->treeNodePtr->right != NULL) {
    ChildrenTypeLineage nuc = iterNucleusNN->treeNodePtr->right->data;
    for (vector<ChildrenTypeNucleus>::iterator iter =
             nuc->treeNode.getChildren().begin();
         iter != nuc->treeNode.getChildren().end();
         ++iter)  // check over all supervoxels
    {
      for (vector<ChildrenTypeNucleus>::iterator iter1 =
               iterNucleusNN->treeNode.getChildren().begin();
           iter1 != iterNucleusNN->treeNode.getChildren().end(); ++iter1)
        JNN = min(JNN, (*iter)->JaccardDistance(*(*iter1)));
      for (vector<ChildrenTypeNucleus>::iterator iter1 =
               rootSplit->data->treeNode.getChildren().begin();
           iter1 != rootSplit->data->treeNode.getChildren().end(); ++iter1)
        JDeath = min(JDeath, (*iter)->JaccardDistance(*(*iter1)));
    }
  }

  float rr = (JNN - JDeath) / (JNN + 1e-4);  // to avoid division by zero
  return rr;
}

void lineageHyperTree::deathJaccardRatioAll(
    vector<float>& ll, TreeNode<ChildrenTypeLineage>* mainRoot) const {
  ll.clear();

  // traverse the tree and for each splitting event we find calculate distance
  queue<TreeNode<ChildrenTypeLineage>*> q;
  q.push(mainRoot);
  TreeNode<ChildrenTypeLineage>* aux;

  while (q.empty() == false) {
    aux = q.front();
    q.pop();

    if (aux != NULL) {
      if (aux->left != NULL) q.push(aux->left);
      if (aux->right != NULL) q.push(aux->right);

      if (aux->getNumChildren() == 0)  // death event
      {
        ll.push_back(deathJaccardRatio(aux));
      }
    }
  }
}

int lineageHyperTree::daughterLengthToNearestNeighborDivision(
    TreeNode<ChildrenTypeLineage>* rootSplit) const  // checks Jaccard distance
                                                     // ratios to see if one
                                                     // cell has invaded teh
                                                     // space of another after
                                                     // death
{
  if (rootSplit == NULL) {
    cout << "ERROR: at lineageHyperTree::deathJaccardRatio: rootSplit node is "
            "null"
         << endl;
    exit(3);
  }
  if (rootSplit->left != NULL || rootSplit->right != NULL) {
    cout << "ERROR: at lineageHyperTree::deathJaccardRatio: rootSplit is not a "
            "death"
         << endl;
    exit(2);
  }

  // find nearest neighbor
  ChildrenTypeLineage iterNucleusNN;
  float dist = findNearestNucleusNeighborInSpaceEuclideanL2(rootSplit->data,
                                                            iterNucleusNN);

  if (dist > 1e31)      // ther eis no nearest neighbor
    return 2147483647;  // this a value outside the possible range

  // calculate length until next cell division for nearest neighbor
  int llAux1 = 0;
  TreeNode<ChildrenTypeLineage>* aux = iterNucleusNN->treeNodePtr;
  while (aux->getNumChildren() == 1) {
    if (aux->left != NULL)
      aux = aux->left;
    else
      aux = aux->right;

    llAux1++;
  }
  if (aux->getNumChildren() == 0)  // daughter has died
    llAux1 = -llAux1;

  return llAux1;
}

void lineageHyperTree::daughterLengthToNearestNeighborDivisionAll(
    vector<int>& ll, TreeNode<ChildrenTypeLineage>* mainRoot) const {
  ll.clear();

  // traverse the tree and for each splitting event we find calculate distance
  queue<TreeNode<ChildrenTypeLineage>*> q;
  q.push(mainRoot);
  TreeNode<ChildrenTypeLineage>* aux;

  while (q.empty() == false) {
    aux = q.front();
    q.pop();

    if (aux != NULL) {
      if (aux->left != NULL) q.push(aux->left);
      if (aux->right != NULL) q.push(aux->right);

      if (aux->getNumChildren() == 0)  // death event
      {
        ll.push_back(daughterLengthToNearestNeighborDivision(aux));
      }
    }
  }
}

//=============================================================================================================================
int lineageHyperTree::mergeShortLivedDaughters(
    TreeNode<ChildrenTypeLineage>* rootSplit, int lengthTMthr) {
  if (rootSplit == NULL || rootSplit->getNumChildren() != 2)
    return 0;  // rootSplit is not a split so we cannnot do anything

  int mergeLineages = 0;  // 0->indicates NO. >0->YES: we merge lineages

  int iniTM = rootSplit->data->TM;
  // calculate length for left branch
  TreeNode<ChildrenTypeLineage>* auxL = rootSplit->left;
  TreeNode<ChildrenTypeLineage>* auxR = rootSplit->right;
  while (auxR->getNumChildren() == 1 && auxL->getNumChildren() == 1 &&
         ((auxL->data->TM - iniTM) <
          lengthTMthr))  // we do not need to keep looking. It is strict <
                         // because we check for death after the while(), so if
                         // (auxL->data->TM - iniTM) == lengthTMthr ), we are
                         // still going to check if it was dead
  {
    if (auxL->left != NULL)
      auxL = auxL->left;
    else
      auxL = auxL->right;

    if (auxR->left != NULL)
      auxR = auxR->left;
    else
      auxR = auxR->right;
  }
  if (auxL->getNumChildren() + auxR->getNumChildren() <=
      1)  // dead cell and there was no division
    mergeLineages = 1;

  // merge lineages if the test is positive
  if (mergeLineages > 0) {
    int err = mergeBranches(rootSplit->left, rootSplit->right);
    if (err > 0) exit(err);
  }

  return mergeLineages;
}
//========================================================================
int lineageHyperTree::mergeShortLivedDaughtersAll(int lengthTMthr, int maxTM,
                                                  int& numCorrections,
                                                  int& numSplits) {
  numCorrections = 0;
  numSplits = 0;
  TreeNode<ChildrenTypeLineage>* aux;
  for (list<lineage>::iterator iterL = lineagesList.begin();
       iterL != lineagesList.end(); ++iterL) {
    queue<TreeNode<ChildrenTypeLineage>*> q;
    if (iterL->bt.pointer_mainRoot() == NULL) continue;
    q.push(iterL->bt.pointer_mainRoot());
    while (!q.empty()) {
      aux = q.front();
      q.pop();
      if (aux->data->TM >=
          maxTM)  // we have reached maximum time point to look into
        continue;
      if (aux->getNumChildren() == 2)  // split
      {
        // check if it it satisfies teh condition. If it does the code merges
        // the two branches
        numSplits++;
        if (mergeShortLivedDaughters(aux, lengthTMthr) > 0) numCorrections++;
      }

      // keep going down the lineage
      if (aux->left != NULL) q.push(aux->left);
      if (aux->right != NULL) q.push(aux->right);
    }
  }
  return 0;
}

//=============================================================================================================================
int lineageHyperTree::deleteShortLivedDaughters(
    TreeNode<ChildrenTypeLineage>* rootSplit, int lengthTMthr) {
  if (rootSplit == NULL || rootSplit->getNumChildren() != 2)
    return 0;  // rootSplit is not a split so we cannnot do anything

  int mergeLineages = 0;  // 0->indicates NO. >0->YES: we merge lineages

  int iniTM = rootSplit->data->TM;
  // calculate length for left branch
  TreeNode<ChildrenTypeLineage>* aux = rootSplit->left;
  // check left branch
  while (aux->getNumChildren() == 1 &&
         ((aux->data->TM - iniTM) <
          lengthTMthr))  // we do not need to keep looking. It is strict <
                         // because we check for death after the while(), so if
                         // (auxL->data->TM - iniTM) == lengthTMthr ), we are
                         // still going to check if it was dead
  {
    if (aux->left != NULL)
      aux = aux->left;
    else
      aux = aux->right;
  }
  if (aux->getNumChildren() == 0) {
    deleteBranch(rootSplit->left);
    mergeLineages = 1;
  } else {
    // check right branch
    aux = rootSplit->right;
    while (aux->getNumChildren() == 1 &&
           ((aux->data->TM - iniTM) <
            lengthTMthr))  // we do not need to keep looking. It is strict <
                           // because we check for death after the while(), so
                           // if (auxL->data->TM - iniTM) == lengthTMthr ), we
                           // are still going to check if it was dead
    {
      if (aux->left != NULL)
        aux = aux->left;
      else
        aux = aux->right;
    }
    if (aux->getNumChildren() == 0)  // right dauhgter died
    {
      deleteBranch(rootSplit->right);
      mergeLineages = 1;
    }
  }

  return mergeLineages;
}
//========================================================================
int lineageHyperTree::deleteShortLivedDaughtersAll(int lengthTMthr, int maxTM,
                                                   int& numCorrections,
                                                   int& numSplits) {
  numCorrections = 0;
  numSplits = 0;
  TreeNode<ChildrenTypeLineage>* aux;
  for (list<lineage>::iterator iterL = lineagesList.begin();
       iterL != lineagesList.end(); ++iterL) {
    queue<TreeNode<ChildrenTypeLineage>*> q;
    if (iterL->bt.pointer_mainRoot() == NULL) continue;
    q.push(iterL->bt.pointer_mainRoot());
    while (!q.empty()) {
      aux = q.front();
      q.pop();
      if (aux->data->TM >=
          maxTM)  // we have reached maximum time point to look into
        continue;
      if (aux->getNumChildren() == 2)  // split
      {
        // check if it it satisfies teh condition. If it does the code merges
        // the two branches
        numSplits++;
        if (deleteShortLivedDaughters(aux, lengthTMthr) > 0) numCorrections++;
      }

      // keep going down the lineage
      if (aux->left != NULL) q.push(aux->left);
      if (aux->right != NULL) q.push(aux->right);
    }
  }
  return 0;
}

//==========================================================================================================
int lineageHyperTree::breakCellDivisionBasedOnCellDivisionPlaneConstraint(
    int TM, double thrCellDivisionPlaneDistance, int& numCorrections,
    int& numSplits) {
  numCorrections = 0;
  numSplits = 0;
  TreeNode<ChildrenTypeLineage> *aux, *auxD;
  float scale[dimsImage];
  supervoxel::getScale(scale);

  float p0, n, m, norm, d;
  for (list<nucleus>::iterator iterN = nucleiList[TM].begin();
       iterN != nucleiList[TM].end(); ++iterN) {
    aux = iterN->treeNodePtr;

    if (aux->getNumChildren() != 2) continue;  // not a cell division

    numSplits++;

    // calculate midplane feature
    norm = 0.0f;
    d = 0.0f;
    for (int ii = 0; ii < dimsImage; ii++) {
      p0 = 0.5 * (aux->left->data->centroid[ii] +
                  aux->right->data->centroid[ii]);           // midpoint
      n = (aux->left->data->centroid[ii] - p0) * scale[ii];  // normal
      norm += (n * n);
      // calculate distance of mother cell to division plane
      m = (iterN->centroid[ii] - p0) * scale[ii];

      d += (n * m);
    }

    d = fabs(d) / sqrt(norm);  // midplane distance

    /*equivalent Matlab code
    %apply scale
    xyzPar = xyzPar .*scale;
    xyzCh1 = xyzCh1 .*scale;
    xyzCh2 = xyzCh2 .*scale;


    %calculat eplane of division based on daughters
    p0 = 0.5 * (xyzCh1 + xyzCh2);
    n = xyzCh1 - p0;%normal
    n = n / norm(n);

    %calculate distance of mother cell to division plane
    m = xyzPar - p0;

    d = abs(dot(m,n));
    */

    if (d > thrCellDivisionPlaneDistance)  // cut linkage between mother and the
                                           // fursthest daughter
    {
      numCorrections++;

      // check which one is thefurthest daughter
      double dL = 0, dR = 0;
      for (int ii = 0; ii < dimsImage; ii++) {
        dL += pow(aux->left->data->centroid[ii] - iterN->centroid[ii], 2) *
              scale[ii];
        dR += pow(aux->right->data->centroid[ii] - iterN->centroid[ii], 2) *
              scale[ii];
      }

      if (dR > dL)  // disconnect right daughter
      {
        auxD = aux->right;
        // remove link in mother
        aux->right = NULL;
      } else {  // disconnect left daughter
        auxD = aux->left;
        // remove link in mother
        aux->left = NULL;
      }
      // remove link in daughter
      auxD->parent = NULL;

      // daughter starts a new lineage
      lineagesList.push_back(lineage());
      list<lineage>::iterator listLineageIter =
          ((++(lineagesList.rbegin()))
               .base());  // iterator for the last element in the list

      // set main root to "paste" sublineage starting at daughter
      listLineageIter->bt.SetMainRoot(auxD);

      // change all the elements downstrem of the daughter to point to the
      // correct lineage
      queue<TreeNode<ChildrenTypeLineage>*> q;
      q.push(auxD);
      while (q.empty() == false) {
        auxD = q.front();
        q.pop();
        auxD->data->treeNode.setParent(listLineageIter);

        if (auxD->left != NULL) q.push(auxD->left);
        if (auxD->right != NULL) q.push(auxD->right);
      }
    }
  }
  return 0;
}

//=============================================================================================================================
int lineageHyperTree::extendDeadNuclei(
    TreeNode<ChildrenTypeLineage>* rootDead) {
  if (rootDead == NULL || rootDead->getNumChildren() != 0)
    return 0;  // rootDead is not a dead split so we cannnot do anything

  // try to find the most obvious continuation

  // 1.-Generate a super-supervoxel by merging all the supervoxel belonging to
  // the nucleus
  vector<ChildrenTypeNucleus>::iterator iterS =
      rootDead->data->treeNode.getChildren().begin();
  supervoxel supervoxelFromNucleus(*(*iterS));
  ++iterS;
  vector<supervoxel*> auxVecS;
  for (; iterS != rootDead->data->treeNode.getChildren().end(); ++iterS) {
    auxVecS.push_back(&(*(*iterS)));
  }
  supervoxelFromNucleus.mergeSupervoxels(
      auxVecS);  // we add all the components at once

  // 2.-find candidate with largest intersection
  uint64 intersectionSize = 0, auxI;
  SibilingTypeSupervoxel intersectionS;
  for (iterS = rootDead->data->treeNode.getChildren().begin();
       iterS != rootDead->data->treeNode.getChildren().end(); ++iterS) {
    for (vector<SibilingTypeSupervoxel>::iterator iterS2 =
             (*iterS)->nearestNeighborsInTimeForward.begin();
         iterS2 != (*iterS)->nearestNeighborsInTimeForward.end(); ++iterS2) {
      auxI = supervoxelFromNucleus.intersectionSize(*(*(iterS2)));
      if (auxI > intersectionSize) {
        intersectionSize = auxI;
        intersectionS = (*iterS2);
      }
    }
  }

  if (intersectionSize == 0) return 0;  // no clear option for extending death

  // 3.find the nuclei that "owns" the supervoxel
  list<nucleus>::iterator iterNucOwner, iterNucOwnerDaughterL,
      iterNucOwnerDaughterR, iterNucNew;
  int intIsCellDivision = 0x000;  // 0->no children;0x0001->left
                                  // children;0x0010->right
                                  // children;0x0011->both children
  if (intersectionS->treeNode.hasParent() == false)  // the supervoxel with
                                                     // highest intersection has
                                                     // not been claimed by
                                                     // anybody->just take it
  {
    iterNucNew = addNucleusFromSupervoxel(
        rootDead->data->TM + 1,
        intersectionS);  // returns iterator to newly created nucleus

    // update lineage-nucleus hypergraph
    iterNucNew->treeNode.setParent(rootDead->data->treeNode.getParent());
    rootDead->data->treeNode.getParent()->bt.SetCurrent(rootDead);
    iterNucNew->treeNodePtr =
        rootDead->data->treeNode.getParent()->bt.insert(iterNucNew);
    if (iterNucNew->treeNodePtr == NULL) exit(3);

    return 1;  // we have added one nucleus
  } else {
    iterNucOwner = intersectionS->treeNode.getParent();
    if (iterNucOwner->treeNodePtr->parent !=
        NULL)  // we were one time step ahead
    {
      iterNucOwner = iterNucOwner->treeNodePtr->parent->data;

      if (iterNucOwner->treeNodePtr->getNumChildren() > 1) {
        intIsCellDivision = 0x0011;
        iterNucOwnerDaughterR = iterNucOwner->treeNodePtr->right->data;
        iterNucOwnerDaughterL = iterNucOwner->treeNodePtr->left->data;
      } else {
        if (iterNucOwner->treeNodePtr->left != NULL) {
          intIsCellDivision = 0x0001;
          iterNucOwnerDaughterL = iterNucOwner->treeNodePtr->left->data;
          iterNucOwnerDaughterR = iterNucOwnerDaughterL;
        } else {
          intIsCellDivision = 0x0010;
          iterNucOwnerDaughterR = iterNucOwner->treeNodePtr->right->data;
          iterNucOwnerDaughterL = iterNucOwnerDaughterR;
        }
      }

    } else {
      return 0;  // there is no parent
    }
  }

  if (iterNucOwner->TM != rootDead->data->TM) {
    cout << "ERROR: lineageHyperTree::extendDeadNuclei: TM does not agree "
            "between two candidate nucleus"
         << endl;
    exit(5);
  }

  // 4.-run a small Hungarian algorithm in order to decide what is the best
  // matching to solve this issue
  list<supervoxel> svListT0;
  for (iterS = rootDead->data->treeNode.getChildren().begin();
       iterS != rootDead->data->treeNode.getChildren().end(); ++iterS) {
    svListT0.push_back(*(*iterS));
  }
  for (iterS = iterNucOwner->treeNode.getChildren().begin();
       iterS != iterNucOwner->treeNode.getChildren().end(); ++iterS) {
    svListT0.push_back(*(*iterS));
  }

  list<supervoxel> nullAssignmentList;  // temporary supervoxel list to simulate
                                        // null assignment
  nullAssignmentList.push_back(supervoxel());
  SibilingTypeSupervoxel nullAssignment = nullAssignmentList.begin();
  nullAssignment->centroid[0] =
      -1e32f;  // characteristic to find out no assignment
  vector<SibilingTypeSupervoxel> assignmentId;
  int err = calculateTrackletsWithSparseHungarianAlgorithm(
      svListT0, 0, 0.9, assignmentId, &nullAssignment);
  if (err > 0) exit(err);

  // 5.-Parse results and modify assignment accordingly
  int extendedLineages = 0;  // 1->we have extended it
  list<supervoxel>::iterator svListT0iter = svListT0.begin();
  int count = 0;
  for (iterS = rootDead->data->treeNode.getChildren().begin();
       iterS != rootDead->data->treeNode.getChildren().end();
       ++iterS, ++svListT0iter, ++count) {
    if (assignmentId[count]->centroid[0] < 0.0f)
      continue;  // not assigned to anything

    if (assignmentId[count]->treeNode.hasParent() ==
        false)  // the assigned element has no parent-> we can claim it directly
    {
      if (extendedLineages == 0)  // we need to create new nucleus in the list
      {
        nucleiList[rootDead->data->TM + 1].push_back(
            nucleus(rootDead->data->TM + 1, assignmentId[count]->centroid));
        iterNucNew = (++(nucleiList[rootDead->data->TM + 1].rbegin()))
                         .base();  // iterator to last added nucleus

        // update lineage-nucleus hypergraph
        iterNucNew->treeNode.setParent(rootDead->data->treeNode.getParent());
        rootDead->data->treeNode.getParent()->bt.SetCurrent(rootDead);
        iterNucNew->treeNodePtr =
            rootDead->data->treeNode.getParent()->bt.insert(iterNucNew);
        if (iterNucNew->treeNodePtr == NULL) exit(3);
        // update supervoxel-nucleus hypergraph
        iterNucNew->addSupervoxelToNucleus(assignmentId[count]);
        assignmentId[count]->treeNode.setParent(iterNucNew);

        extendedLineages++;
      }
    } else if ((assignmentId[count]->treeNode.getParent() ==
                iterNucOwnerDaughterL) ||
               (assignmentId[count]->treeNode.getParent() ==
                iterNucOwnerDaughterR))  // to confirm it is not null assignment
                                         // && we are "stealing" a supervoxel
                                         // from iterNucOwner and not from
                                         // anothe nuclei
    {
      if (extendedLineages == 0)  // we need to create new nucleus in the list
      {
        nucleiList[rootDead->data->TM + 1].push_back(
            nucleus(rootDead->data->TM + 1, assignmentId[count]->centroid));
        iterNucNew = (++(nucleiList[rootDead->data->TM + 1].rbegin()))
                         .base();  // iterator to last added nucleus

        // update lineage-nucleus hypergraph
        iterNucNew->treeNode.setParent(rootDead->data->treeNode.getParent());
        rootDead->data->treeNode.getParent()->bt.SetCurrent(rootDead);
        iterNucNew->treeNodePtr =
            rootDead->data->treeNode.getParent()->bt.insert(iterNucNew);
        if (iterNucNew->treeNodePtr == NULL) exit(3);

        extendedLineages++;
      }
      // update supervoxel-nucleus hypergraph
      if (assignmentId[count]->treeNode.getParent() == iterNucOwnerDaughterL) {
        iterNucOwnerDaughterL->removeSupervoxelFromNucleus(assignmentId[count]);
        // if( ret > 0 )
        //	cout<<"WARNING: lineageHyperTree::extendDeadNuclei: supervoxel
        // not found to be removed from nucleus"<<endl;
      } else {
        iterNucOwnerDaughterR->removeSupervoxelFromNucleus(assignmentId[count]);
        // if( ret > 0 )
        //	cout<<"WARNING: lineageHyperTree::extendDeadNuclei: supervoxel
        // not found to be removed from nucleus"<<endl;
      }
      iterNucNew->addSupervoxelToNucleus(assignmentId[count]);
      assignmentId[count]->treeNode.setParent(iterNucNew);
    }
  }

  // make sure original nuclei still has some supervoxels associated
  if (((intIsCellDivision & 0x0001) != 0) &&
      (iterNucOwnerDaughterL->treeNode.getNumChildren() == 0)) {
    int TMaux = iterNucOwnerDaughterL->TM;
    delete iterNucOwnerDaughterL->treeNodePtr;
    iterNucOwner->treeNodePtr->left = NULL;
    nucleiList[TMaux].erase(iterNucOwnerDaughterL);
  }
  if (((intIsCellDivision & 0x0010) != 0) &&
      (iterNucOwnerDaughterR->treeNode.getNumChildren() == 0)) {
    int TMaux = iterNucOwnerDaughterR->TM;
    delete iterNucOwnerDaughterR->treeNodePtr;
    iterNucOwner->treeNodePtr->right = NULL;
    nucleiList[TMaux].erase(iterNucOwnerDaughterR);
  }

  return extendedLineages;  // returns 1 if extension was achieved
}
//========================================================================
int lineageHyperTree::extendDeadNucleiAtTM(int TM, int& numExtensions,
                                           int& numDeaths) {
  numExtensions = 0;
  numDeaths = 0;

  if (TM < 0 || TM >= (int)(maxTM)) return 0;

  TreeNode<ChildrenTypeLineage>* aux;
  for (list<nucleus>::iterator iterN = nucleiList[TM].begin();
       iterN != nucleiList[TM].end(); ++iterN) {
    if (iterN->treeNodePtr->getNumChildren() == 0) {
      numDeaths++;
      numExtensions += extendDeadNuclei(iterN->treeNodePtr);
    }
  }

  return 0;
}

//========================================================================
int lineageHyperTree::deleteDeadBranchesAll(int maxTM, int& numDelete) {
  numDelete = 0;

  TreeNode<ChildrenTypeLineage> *root1, *root2, *aux;
  list<lineage>::iterator iterL = lineagesList.begin();
  list<lineage>::iterator iterLnext;

  // int count = 0;
  // first delete lineages that die even before splitting
  while (iterL != lineagesList.end()) {
    root1 = iterL->bt.pointer_mainRoot();
    // to be able to jump to next element even if we delete current lineage
    // because of deletion
    iterL++;
    iterLnext = iterL;
    iterL--;
    if (root1 != NULL) {
      queue<TreeNode<ChildrenTypeLineage>*> q;
      q.push(root1);
      while (!q.empty()) {
        aux = q.front();
        q.pop();
        if (aux->data->TM >=
            maxTM)  // we have reached maximum time point to look into
          continue;
        if (aux->getNumChildren() == 2)  // split
        {
          break;                                // we reach a split
        } else if (aux->getNumChildren() == 0)  // death->delete lineage
        {
          numDelete++;

          list<lineage>::iterator iterLerase = aux->data->treeNode.getParent();
          TreeNode<ChildrenTypeLineage>* auxErase;
          while (aux != NULL) {
            auxErase = aux;
            aux = aux->parent;  // to keep going upstream
            if (aux != NULL) {
              if (aux->left == auxErase)
                aux->left = NULL;  // delete binary tree
              else
                aux->right = NULL;

              if (aux->getNumChildren() >
                  0)  // there was a cell division, so we stop removing here
                aux = NULL;  // so I stop removing the lineage
            }

            // remove supervoxel-nuclei hypergraph
            for (vector<ChildrenTypeNucleus>::iterator iterS =
                     auxErase->data->treeNode.getChildren().begin();
                 iterS != auxErase->data->treeNode.getChildren().end();
                 ++iterS) {
              (*iterS)->treeNode.deleteParent();
            }
            // remove nuclei-lineage connection
            nucleiList[auxErase->data->TM].erase(auxErase->data);
            iterLerase->bt.remove(auxErase);  // aux has been released now
          }

          lineagesList.erase(iterLerase);  // delete lineage
          break;
        }

        // keep going down the lineage
        if (aux->left != NULL) q.push(aux->left);
        if (aux->right != NULL) q.push(aux->right);
      }
    }

    iterL = iterLnext;
    // count++;
  }
  return 0;
}

//========================================================================
int lineageHyperTree::mergeParallelLineagesAll(int conn3D,
                                               size_t minNeighboringVoxels,
                                               int& numMerges) {
  int64 boundarySize[dimsImage];
  int64* neighOffset =
      supervoxel::buildNeighboorhoodConnectivity(conn3D, boundarySize);
  numMerges = 0;

  TreeNode<ChildrenTypeLineage> *root1, *root2;
  list<lineage>::iterator iterL = lineagesList.begin();
  list<lineage>::iterator iterLnext;

  bool flag;
  // int count = 0;
  while (iterL != lineagesList.end()) {
    root1 = iterL->bt.pointer_mainRoot();
    // to be able to jump to next element even if we delete current lineage
    // because of merging
    iterL++;
    iterLnext = iterL;
    iterL--;
    if (root1 != NULL) {
      // check if it needs to be merged with the element above in the
      // hierarchical segmentation
      cout << "WARNING: lineageHyperTree::mergeParallelLineagesAll: need to be "
              "modified"
           << endl;

      flag = false;
      for (vector<ChildrenTypeNucleus>::iterator iterS2 =
               root1->data->treeNode.getChildren().begin();
           iterS2 != root1->data->treeNode.getChildren().end(); ++iterS2) {
        for (vector<ChildrenTypeNucleus>::iterator iterS1 =
                 (*iterS2)->nearestNeighborsInSpace.begin();
             iterS1 != (*iterS2)->nearestNeighborsInSpace.end(); ++iterS1) {
          root2 = (*iterS1)->treeNode.getParent()->treeNodePtr;
          // VIP: WE ALWAYS MERGE ROOT2 TO ROOT1 (SO ROOT2 IS DESTROYED). THIS
          // IS IMPORTANT IF THEY BELONG TO DIFFERENT LINEAGES
          if (mergeParallelLineages(root2, root1, conn3D, neighOffset,
                                    minNeighboringVoxels) >
              0)  // root1 is distroyed here
          {
            numMerges++;
            flag = true;
            break;
          }
        }
        if (flag == true) break;
      }
    }

    iterL = iterLnext;
    // count++;
  }

  delete[] neighOffset;
  return 0;
}

int lineageHyperTree::deleteShortLineagesWithHighBackgroundClassifierScore(
    int maxLengthBackgroundCheck, int frame, float thrMinScoreFx,
    int& numPositiveChecks, int numActions) {
  numPositiveChecks = 0;
  numActions = 0;

  if (frame >= maxTM) return 0;

  TreeNode<ChildrenTypeLineage>* aux;
  vector<ChildrenTypeLineage> iterNucleusNNvec(4);
  vector<float> distVec(4);
  for (list<nucleus>::iterator iterN = nucleiList[frame].begin();
       iterN != nucleiList[frame].end(); ++iterN) {
    // check if it is a short lineage
    if (iterN->treeNodePtr->getNumChildren() == 0)  // lineage has died
    {
      aux = iterN->treeNodePtr->parent;
      int ll = 1;
      while (aux != NULL && ll <= maxLengthBackgroundCheck) {
        aux = aux->parent;
        ll++;
      }

      if (ll <= maxLengthBackgroundCheck)  // lineage satisfies condition of
                                           // being shorter
      {
        numPositiveChecks++;
        // check minimum value of split classifier
        bool isBackground = true;
        bool existCellDivision = false;
        aux = iterN->treeNodePtr->parent;
        while (aux != NULL && isBackground == true) {
          if (aux->getNumChildren() > 1) existCellDivision = true;
          // check for each supervoxel
          for (vector<ChildrenTypeNucleus>::iterator iterS =
                   aux->data->treeNode.getChildren().begin();
               iterS != aux->data->treeNode.getChildren().end(); ++iterS) {
            if ((*iterS)->probClassifier < thrMinScoreFx) {
              isBackground = false;
              break;
            }
          }
          aux = aux->parent;
        }

        // if length == 1, then also remove if there are no birth nearby in the
        // next frame
        if (isBackground == false && ll == 1) {
          isBackground = true;
          findKNearestNucleiNeighborInTimeForwardEuclideanL2(
              iterN, iterNucleusNNvec, distVec);
          for (size_t aa = 0; aa < distVec.size(); aa++) {
            if (distVec[aa] < 1e31) {
              if (iterNucleusNNvec[aa]->treeNodePtr->parent == NULL)  // birth
              {
                isBackground = false;
                break;
              }
            }
          }
        }

        // delete lineage if it passed the test
        if (isBackground == true) {
          numActions++;

          aux = iterN->treeNodePtr->parent;
          list<lineage>::iterator iterLerase = aux->data->treeNode.getParent();
          TreeNode<ChildrenTypeLineage>* auxErase;
          while (aux != NULL) {
            auxErase = aux;
            aux = aux->parent;  // to keep going upstream
            if (aux != NULL) {
              if (aux->left == auxErase)
                aux->left = NULL;  // delete binary tree
              else
                aux->right = NULL;

              if (aux->getNumChildren() >
                  0)  // there was a cell division, so we stop removing here
                aux = NULL;  // so I stop removing the lineage
            }

            // remove supervoxel-nuclei hypergraph
            for (vector<ChildrenTypeNucleus>::iterator iterS =
                     auxErase->data->treeNode.getChildren().begin();
                 iterS != auxErase->data->treeNode.getChildren().end();
                 ++iterS) {
              (*iterS)->treeNode.deleteParent();
            }
            // remove nuclei-lineage connection
            nucleiList[frame].erase(aux->data);
            aux->data->treeNode.getParent()->bt.remove(
                aux);  // aux has been released now
          }

          if (existCellDivision == false)  // delete lineage
          {
            lineagesList.erase(iterLerase);
          }
        }
      }
    }
  }

  return 0;
}

//========================================================================
int lineageHyperTree::mergeNonSeparatingDaughtersAll(
    int maxTM, size_t minNeighboringVoxels, int conn3D, int& numCorrections,
    int& numSplits) {
  int64 boundarySize[dimsImage];
  int64* neighOffset =
      supervoxel::buildNeighboorhoodConnectivity(conn3D, boundarySize);
  numCorrections = 0;
  numSplits = 0;
  TreeNode<ChildrenTypeLineage>* aux;
  for (list<lineage>::iterator iterL = lineagesList.begin();
       iterL != lineagesList.end(); ++iterL) {
    queue<TreeNode<ChildrenTypeLineage>*> q;
    if (iterL->bt.pointer_mainRoot() == NULL) continue;
    q.push(iterL->bt.pointer_mainRoot());
    while (!q.empty()) {
      aux = q.front();
      q.pop();
      if (aux->data->TM >=
          maxTM)  // we have reached maximum time point to look into
        continue;
      if (aux->getNumChildren() == 2)  // split
      {
        numSplits++;
        if (mergeNonSeparatingDaughters(aux, conn3D, neighOffset,
                                        minNeighboringVoxels) > 0)
          numCorrections++;
      }

      // keep going down the lineage
      if (aux->left != NULL) q.push(aux->left);
      if (aux->right != NULL) q.push(aux->right);
    }
  }

  delete[] neighOffset;
  return 0;
}

//========================================================================
int lineageHyperTree::mergeShortLivedAndCloseByDaughtersAll(
    int lengthTMthr, int maxTM, size_t minNeighboringVoxels, int conn3D,
    int& numCorrections, int& numSplits) {
  int64 boundarySize[dimsImage];
  int64* neighOffset =
      supervoxel::buildNeighboorhoodConnectivity(conn3D, boundarySize);
  numCorrections = 0;
  numSplits = 0;
  TreeNode<ChildrenTypeLineage>* aux;
  for (list<lineage>::iterator iterL = lineagesList.begin();
       iterL != lineagesList.end(); ++iterL) {
    queue<TreeNode<ChildrenTypeLineage>*> q;
    if (iterL->bt.pointer_mainRoot() == NULL) continue;
    q.push(iterL->bt.pointer_mainRoot());
    while (!q.empty()) {
      aux = q.front();
      q.pop();
      if (aux->data->TM >=
          maxTM)  // we have reached maximum time point to look into
        continue;
      if (aux->getNumChildren() == 2)  // split
      {
        // check if it it satisfies teh condition. If it does the code merges
        // the two branches
        numSplits++;
        if (mergeShortLivedDaughters(aux, lengthTMthr) > 0) {
          numCorrections++;
        } else {  // do a more costly check to see if they are inc ontact all
                  // the time
          if (mergeNonSeparatingDaughters(aux, conn3D, neighOffset,
                                          minNeighboringVoxels) > 0)
            numCorrections++;
        }
      }

      // keep going down the lineage
      if (aux->left != NULL) q.push(aux->left);
      if (aux->right != NULL) q.push(aux->right);
    }
  }

  delete[] neighOffset;
  return 0;
}

//================================================================================================================
int lineageHyperTree::mergeNonSeparatingDaughters(
    TreeNode<ChildrenTypeLineage>* rootSplit, int conn3D, int64* neighOffset) {
  if (rootSplit == NULL || rootSplit->getNumChildren() != 2)
    return 0;  // rootSplit is not a split so we cannnot do anything

  int mergeLineages = 1;  // 0->indicates NO. >0->YES: we merge lineages

  // int iniTM = rootSplit->data->TM;
  // calculate length for left branch
  TreeNode<ChildrenTypeLineage>* auxL = rootSplit->left;
  TreeNode<ChildrenTypeLineage>* auxR = rootSplit->right;

  while (auxR->getNumChildren() == 1 && auxL->getNumChildren() == 1 &&
         mergeLineages > 0) {
    // check if they are neighbors
    bool isNeighboring = false;
    for (vector<ChildrenTypeNucleus>::iterator iterS1 =
             auxL->data->treeNode.getChildren().begin();
         iterS1 != auxL->data->treeNode.getChildren().end(); ++iterS1) {
      for (vector<ChildrenTypeNucleus>::iterator iterS2 =
               auxR->data->treeNode.getChildren().begin();
           iterS2 != auxR->data->treeNode.getChildren().end(); ++iterS2) {
        if ((*iterS1)->isNeighboring(*(*iterS2), conn3D, neighOffset) == true) {
          isNeighboring = true;
          break;
        }
      }
      if (isNeighboring == true) break;
    }

    cout << "DEBUGGING: mergeNonSeparatingDaughters: isNeighboring = "
         << isNeighboring << " at TM=" << auxR->data->TM << endl;

    if (isNeighboring == false)  // nuclei are not neighbors anymore->no merging
    {
      mergeLineages = 0;
    }

    // update to next element in the tree
    if (auxL->left != NULL)
      auxL = auxL->left;
    else
      auxL = auxL->right;

    if (auxR->left != NULL)
      auxR = auxR->left;
    else
      auxR = auxR->right;
  }
  // check the last element
  if (mergeLineages > 0) {
    bool isNeighboring = false;
    for (vector<ChildrenTypeNucleus>::iterator iterS1 =
             auxL->data->treeNode.getChildren().begin();
         iterS1 != auxL->data->treeNode.getChildren().end(); ++iterS1) {
      for (vector<ChildrenTypeNucleus>::iterator iterS2 =
               auxR->data->treeNode.getChildren().begin();
           iterS2 != auxR->data->treeNode.getChildren().end(); ++iterS2) {
        if ((*iterS1)->isNeighboring(*(*iterS2), conn3D, neighOffset) == true) {
          isNeighboring = true;
          break;
        }
      }
      if (isNeighboring == true) break;
    }
    cout << "DEBUGGING: mergeNonSeparatingDaughters: isNeighboring = "
         << isNeighboring << " at TM=" << auxR->data->TM << endl;
    if (isNeighboring == false)  // nuclei are not neighbors anymore->no merging
    {
      mergeLineages = 0;
    }
  }

  // merge lineages if the test is positive
  if (mergeLineages > 0) {
    int err = mergeBranches(rootSplit->left, rootSplit->right);
    if (err > 0) exit(err);
  }

  return mergeLineages;
}

//===================================================================================================
int lineageHyperTree::mergeNonSeparatingDaughters(
    TreeNode<ChildrenTypeLineage>* rootSplit, int conn3D, int64* neighOffset,
    size_t minNeighboringVoxels)  // TODO: generate a function that once it has
                                  // found minNeighboringVoxels it stops
                                  // checking
{
  if (rootSplit == NULL || rootSplit->getNumChildren() != 2)
    return 0;  // rootSplit is not a split so we cannnot do anything

  int mergeLineages = 1;  // 0->indicates NO. >0->YES: we merge lineages

  // int iniTM = rootSplit->data->TM;
  // calculate length for left branch
  TreeNode<ChildrenTypeLineage>* auxL = rootSplit->left;
  TreeNode<ChildrenTypeLineage>* auxR = rootSplit->right;

  size_t numNeighboringVoxels;
  vector<uint64> PixelIdxList1, PixelIdxList2;
  PixelIdxList1.reserve(100);
  PixelIdxList2.reserve(100);

  while (auxR->getNumChildren() == 1 && auxL->getNumChildren() == 1 &&
         mergeLineages > 0) {
    // check if they are neighbors
    bool isNeighboring = false;
    numNeighboringVoxels = 0;
    for (vector<ChildrenTypeNucleus>::iterator iterS1 =
             auxL->data->treeNode.getChildren().begin();
         iterS1 != auxL->data->treeNode.getChildren().end(); ++iterS1) {
      for (vector<ChildrenTypeNucleus>::iterator iterS2 =
               auxR->data->treeNode.getChildren().begin();
           iterS2 != auxR->data->treeNode.getChildren().end(); ++iterS2) {
        (*iterS1)->neighboringVoxels(*(*iterS2), conn3D, neighOffset,
                                     PixelIdxList1, PixelIdxList2);
        numNeighboringVoxels = max(numNeighboringVoxels, PixelIdxList1.size());
      }
    }

    // cout<<"DEBUGGING: mergeNonSeparatingDaughters: isNeighboring =
    // "<<numNeighboringVoxels<<" at TM="<< auxR->data->TM <<endl;

    if (numNeighboringVoxels <
        minNeighboringVoxels)  // nuclei are not neighbors anymore->no merging
    {
      mergeLineages = 0;
    }

    // update to next element in the tree
    if (auxL->left != NULL)
      auxL = auxL->left;
    else
      auxL = auxL->right;

    if (auxR->left != NULL)
      auxR = auxR->left;
    else
      auxR = auxR->right;
  }

  // if we finished because there was a division, we cannot merge lineages
  // easily (mergeBranches will give error)
  if (auxR->getNumChildren() > 1 || auxL->getNumChildren() > 1) {
    mergeLineages = 0;
  }

  // check the last element
  if (mergeLineages > 0) {
    bool isNeighboring = false;
    numNeighboringVoxels = 0;
    for (vector<ChildrenTypeNucleus>::iterator iterS1 =
             auxL->data->treeNode.getChildren().begin();
         iterS1 != auxL->data->treeNode.getChildren().end(); ++iterS1) {
      for (vector<ChildrenTypeNucleus>::iterator iterS2 =
               auxR->data->treeNode.getChildren().begin();
           iterS2 != auxR->data->treeNode.getChildren().end(); ++iterS2) {
        (*iterS1)->neighboringVoxels(*(*iterS2), conn3D, neighOffset,
                                     PixelIdxList1, PixelIdxList2);
        numNeighboringVoxels = max(numNeighboringVoxels, PixelIdxList1.size());
      }
    }

    // cout<<"DEBUGGING: mergeNonSeparatingDaughters: isNeighboring =
    // "<<numNeighboringVoxels<<" at TM="<< auxR->data->TM <<endl;

    if (numNeighboringVoxels <
        minNeighboringVoxels)  // nuclei are not neighbors anymore->no merging
    {
      mergeLineages = 0;
    }
  }

  // merge lineages if the test is positive
  if (mergeLineages > 0) {
    int err = mergeBranches(rootSplit->left, rootSplit->right);
    if (err > 0) exit(err);
  }

  return mergeLineages;
}

//======================================================================================================
int lineageHyperTree::mergeParallelLineages(
    TreeNode<ChildrenTypeLineage>* root1, TreeNode<ChildrenTypeLineage>* root2,
    int conn3D, int64* neighOffset, size_t minNeighboringVoxels) {
  if (root1 == NULL || root2 == NULL)
    return 0;  // rootSplit is not a split so we cannnot do anything
  if (root1 == root2) return 0;  // we are trying to merge the same lineage
  if (root1->data->TM != root2->data->TM)
    return 0;  // we cannot merge nuclei from different time points

  int mergeLineages = 1;  // 0->indicates NO. >0->YES: we merge lineages

  // int iniTM = rootSplit->data->TM;
  // calculate length for left branch
  TreeNode<ChildrenTypeLineage>* auxL = root1;
  TreeNode<ChildrenTypeLineage>* auxR = root2;

  size_t numNeighboringVoxels;
  vector<uint64> PixelIdxList1, PixelIdxList2;
  PixelIdxList1.reserve(100);
  PixelIdxList2.reserve(100);

  while (auxR->getNumChildren() == 1 && auxL->getNumChildren() == 1 &&
         mergeLineages > 0) {
    // check if they are neighbors
    bool isNeighboring = false;
    numNeighboringVoxels = 0;
    for (vector<ChildrenTypeNucleus>::iterator iterS1 =
             auxL->data->treeNode.getChildren().begin();
         iterS1 != auxL->data->treeNode.getChildren().end(); ++iterS1) {
      for (vector<ChildrenTypeNucleus>::iterator iterS2 =
               auxR->data->treeNode.getChildren().begin();
           iterS2 != auxR->data->treeNode.getChildren().end(); ++iterS2) {
        (*iterS1)->neighboringVoxels(*(*iterS2), conn3D, neighOffset,
                                     PixelIdxList1, PixelIdxList2);
        numNeighboringVoxels = max(numNeighboringVoxels, PixelIdxList1.size());
      }
    }

    // cout<<"DEBUGGING: mergeNonSeparatingDaughters: isNeighboring =
    // "<<numNeighboringVoxels<<" at TM="<< auxR->data->TM <<endl;

    if (numNeighboringVoxels <
        minNeighboringVoxels)  // nuclei are not neighbors anymore->no merging
    {
      mergeLineages = 0;
    }

    // update to next element in the tree
    if (auxL->left != NULL)
      auxL = auxL->left;
    else
      auxL = auxL->right;

    if (auxR->left != NULL)
      auxR = auxR->left;
    else
      auxR = auxR->right;
  }

  // if we finished because there was a division, we cannot merge lineages
  // easily (mergeBranches will give error)
  if (auxR->getNumChildren() > 1 || auxL->getNumChildren() > 1) {
    mergeLineages = 0;
  }

  // check the last element
  if (mergeLineages > 0) {
    bool isNeighboring = false;
    numNeighboringVoxels = 0;
    for (vector<ChildrenTypeNucleus>::iterator iterS1 =
             auxL->data->treeNode.getChildren().begin();
         iterS1 != auxL->data->treeNode.getChildren().end(); ++iterS1) {
      for (vector<ChildrenTypeNucleus>::iterator iterS2 =
               auxR->data->treeNode.getChildren().begin();
           iterS2 != auxR->data->treeNode.getChildren().end(); ++iterS2) {
        (*iterS1)->neighboringVoxels(*(*iterS2), conn3D, neighOffset,
                                     PixelIdxList1, PixelIdxList2);
        numNeighboringVoxels = max(numNeighboringVoxels, PixelIdxList1.size());
      }
    }

    // cout<<"DEBUGGING: mergeNonSeparatingDaughters: isNeighboring =
    // "<<numNeighboringVoxels<<" at TM="<< auxR->data->TM <<endl;

    if (numNeighboringVoxels <
        minNeighboringVoxels)  // nuclei are not neighbors anymore->no merging
    {
      mergeLineages = 0;
    }
  }

  // merge lineages if the test is positive
  if (mergeLineages > 0) {
    //---------------------debug------------------------------------------------
    /*
            vector< rootSublineage > vecRoot;
            vecRoot.push_back(root1);
            vecRoot.push_back(root2);
            lineageHyperTree lhtSub(11);//length of the sublineage
            cutSublineage(vecRoot, lhtSub);
            cout<<"DEBUGGING: numNeighboringVoxels =
       "<<numNeighboringVoxels<<endl;
            string
       imgPath("G:/12-07-17/TimeFused_BackgrSubtraction_thrPctile40_maxSize3000_otzu/TM?????/CM0_CM1_CHN00_CHN01.fusedStack_bckgSub_?????.tif");
            string
       imgLpath("G:/12-07-17/TimeFused_BackgrSubtraction_thrPctile40_maxSize3000_otzu/TM?????/CM0_CM1_CHN00_CHN01.fusedStack_bckgSub_PersistanceSeg_tau14_?????.tif");
            string
       imgRawPath("G:/12-07-17/TimeFused_BackgrSubtraction_thrPctile40_maxSize3000_otzu/TM?????/CM0_CM1_CHN00_CHN01.fusedStack_bckgSub_?????.tif");
            lhtSub.debugPrintLineageForLocalLineageDisplayinMatlab(imgPath,
       imgLpath, "debugginRule3", imgRawPath);
            */
    //-------------------------------------------------------------------------
    int err = mergeBranches(root1, root2);
    if (err > 0) exit(err);
  }

  return mergeLineages;
}

//========================================================================
int lineageHyperTree::splitDeathDivisionPattern(
    TreeNode<ChildrenTypeLineage>* rootSplit, int lengthTMthr) {
  if (rootSplit == NULL || rootSplit->getNumChildren() > 0)
    return 0;  // rootSplit is not a dead cell so we cannnot do anything

  // find nearest neighbor
  list<nucleus>::iterator iterNN;
  float dist =
      findNearestNucleusNeighborInSpaceEuclideanL2(rootSplit->data, iterNN);
  if (dist > 1e30)  // no nearest neighbor was found
    return 0;

  // check if the nearest neighbor divides within lengthTMthr
  int iniTM = rootSplit->data->TM;

  TreeNode<ChildrenTypeLineage>* aux = iterNN->treeNodePtr;
  while (aux->getNumChildren() == 1 &&
         ((aux->data->TM - iniTM) <
          lengthTMthr))  // we do not need to keep looking. It is strict <
                         // because we check for death after the while(), so if
                         // (auxL->data->TM - iniTM) == lengthTMthr ), we are
                         // still going to check if it was dead
  {
    if (aux->left != NULL)
      aux = aux->left;
    else
      aux = aux->right;
  }
  if (aux->getNumChildren() == 2)  // element is dividing soon enough
  {
    // TODO: make function to perform split (hopefully we have two supervoxels)
    return 1;
  } else {  // no pattern of death + nearest neighbro division
    return 0;
  }
}

//=============================================================================================
int lineageHyperTree::mergeBranches(TreeNode<ChildrenTypeLineage>* root1,
                                    TreeNode<ChildrenTypeLineage>* root2) {
  if (root1 == NULL || root2 == NULL) return 0;
  if (root1 == root2) return 0;

  int TMoffset = 0;
  if (isSublineage == true)  // we need an offset for TM
    TMoffset = nucleiList[0].begin()->TM;

  bool flagDeleteLineage = false;
  bool flagChangeLineageParent = false;
  list<lineage>::iterator iterL1 = root1->data->treeNode.getParent();
  list<lineage>::iterator iterL2 = root2->data->treeNode.getParent();

  if ((iterL1 != iterL2) &&
      (root2->parent ==
       NULL))  // we need to delete iterL2 from lineageList after the merge
  {
    flagDeleteLineage = true;
  }

  TreeNode<ChildrenTypeLineage>* aux1;
  TreeNode<ChildrenTypeLineage>* aux2;
  TreeNode<ChildrenTypeLineage> *aux1ch = root1, *aux2ch = root2;
  unsigned int numCh1, numCh2;

  // detach root2 from its parent before starting
  if (root2->parent != NULL) {
    if (root2->parent->left == root2)
      root2->parent->left = NULL;
    else
      root2->parent->right = NULL;
  }

  while (aux1ch != NULL && aux2ch != NULL) {
    // copy pointers from previous iteration
    aux1 = aux1ch;
    aux2 = aux2ch;

    // check the next visit before erasing all the info
    numCh1 = aux1->getNumChildren();
    numCh2 = aux2->getNumChildren();
    if (numCh2 == 0)  // we have reached the end of the merging
    {
      aux2ch = NULL;
    } else if (numCh1 == 0)  // we have reached the end of the merging but aux1
                             // has finished first->since we have been attaching
                             // everything to aux1 we need connect
    {
      aux1ch = NULL;
      if (aux2->left != NULL) {
        aux1->left = aux2->left;
        aux1->left->parent = aux1;
      }
      if (aux2->right != NULL) {
        aux1->right = aux2->right;
        aux1->right->parent = aux1;
      }
      flagChangeLineageParent = true;  // we will need to change the pointer for
                                       // all the elements down the line
    } else if (numCh1 > 1 || numCh2 > 1) {
      cout << "ERROR: at lineageHyperTree::mergeBranches: found a cell "
              "division before death. I cannot not merge these two branches"
           << endl;
      return 2;
    }

    if (aux1ch != NULL)  // update next child
    {
      if (aux1->left != NULL)
        aux1ch = aux1->left;
      else
        aux1ch = aux1->right;
    }
    if (aux2ch != NULL)  // update next child
    {
      if (aux2->left != NULL)
        aux2ch = aux2->left;
      else
        aux2ch = aux2->right;
    }
    // check that we are still synchronized in time
    if (aux1->data->TM != aux2->data->TM) {
      cout << "ERROR: at lineageHyperTree::mergeBranches: both elements need "
              "to have the same time point"
           << endl;
      return 3;
    }

    // copy edges for supervoxel from aux1 to aux2
    for (vector<ChildrenTypeNucleus>::iterator iter2 =
             aux2->data->treeNode.getChildren().begin();
         iter2 != aux2->data->treeNode.getChildren().end(); ++iter2) {
      aux1->data->treeNode.getChildren().push_back(*iter2);
      (*iter2)->treeNode.setParent(aux1->data);
    }
    // recalculate centroid
    assert(supervoxel::dataSizeInBytes /
               (supervoxel::dataDims[0] * supervoxel::dataDims[1] *
                supervoxel::dataDims[2]) ==
           4);
    int err = calculateNucleiIntensityCentroid<float>(aux1->data);
    if (err > 0) return err;

    // remove aux2 nuclei from list
    nucleiList[aux2->data->TM - TMoffset].erase(aux2->data);

    // remove aux2 tree node from lineage
    delete aux2;
  }

  if (flagDeleteLineage == true) {
    iterL2->bt.SetMainRootToNULL();  // we have already deallocated memory, so
                                     // if we do not set it to null the
                                     // destructor tries ot do it again
    lineagesList.erase(iterL2);
  }

  if (flagChangeLineageParent == true) {
    // we need to make sure all the the elements in the binary tree point to the
    // correct lineage
    queue<TreeNode<ChildrenTypeLineage>*> q;
    q.push(root1);
    TreeNode<ChildrenTypeLineage>* aux;
    while (!q.empty()) {
      aux = q.front();
      q.pop();
      if (aux->left != NULL) q.push(aux->left);
      if (aux->right != NULL) q.push(aux->right);
      aux->data->treeNode.setParent(iterL1);
    }
  }

  return 0;
}

//=============================================================================================
int lineageHyperTree::deleteBranch(TreeNode<ChildrenTypeLineage>* root) {
  if (root == NULL) return 0;

  int TMoffset = 0;
  if (isSublineage == true)  // we need an offset for TM
    TMoffset = nucleiList[0].begin()->TM;

  bool flagDeleteLineage = false;
  list<lineage>::iterator iterL = root->data->treeNode.getParent();
  if (root->parent == NULL)  // we need to delete iterL from lineageList because
                             // root is the beginning of a lineage
  {
    flagDeleteLineage = true;
  } else  // detach root from its parent before starting
  {
    if (root->parent->left == root)
      root->parent->left = NULL;
    else
      root->parent->right = NULL;
  }

  TreeNode<ChildrenTypeLineage>* aux;
  queue<TreeNode<ChildrenTypeLineage>*> q;
  q.push(root);

  while (q.empty() == false) {
    // copy pointers from previous iteration
    aux = q.front();
    q.pop();

    // check the next visit before erasing all the info
    if (aux->left != NULL) q.push(aux->left);
    if (aux->right != NULL) q.push(aux->right);

    // remove supervoxel-nuclei connection in hypergraph
    for (vector<ChildrenTypeNucleus>::iterator iterS =
             aux->data->treeNode.getChildren().begin();
         iterS != aux->data->treeNode.getChildren().end(); ++iterS) {
      (*iterS)->treeNode.deleteParent();
    }

    // remove aux nuclei from list
    nucleiList[aux->data->TM - TMoffset].erase(aux->data);

    // remove aux tree node from lineage
    delete aux;
  }

  if (flagDeleteLineage == true) {
    iterL->bt.SetMainRootToNULL();  // we have already deallocated memory, so if
                                    // we do not set it to null the destructor
                                    // tries ot do it again
    lineagesList.erase(iterL);
  }

  return 0;
}

//============================================================================
template <class imgTypeC>
int lineageHyperTree::calculateNucleiIntensityCentroid(
    SibilingTypeNucleus& iterN) {
  if (iterN->treeNode.getNumChildren() == 0)  // no supervoxels
  {
    iterN->setDead();
    return 0;
  }

  float w = 0.0f;  // weight
  float auxCentroid[dimsImage];
  memset(auxCentroid, 0, sizeof(float) * dimsImage);
  uint64 auxCoord1, auxCoord2;

  if (iterN->treeNode.getChildren().front()->dataPtr ==
      NULL)  // no image associated with supervoxels
  {
    for (vector<ChildrenTypeNucleus>::const_iterator iter =
             iterN->treeNode.getChildren().begin();
         iter != iterN->treeNode.getChildren().end(); ++iter) {
      for (vector<uint64>::const_iterator iterS = (*iter)->PixelIdxList.begin();
           iterS != (*iter)->PixelIdxList.end(); ++iterS) {
        // calculate coordinates of this point
        auxCoord2 = *iterS;
        for (int ii = 0; ii < dimsImage; ii++) {
          auxCoord1 = auxCoord2 % (supervoxel::dataDims[ii]);
          auxCoord2 -= auxCoord1;
          auxCoord2 /= supervoxel::dataDims[ii];
          auxCentroid[ii] += (float)(auxCoord1);
        }
        w += 1.0f;
      }
    }
  } else {  // we have an image associated with the supervoxel->weighted
            // centroid
    float wAux;
    for (vector<ChildrenTypeNucleus>::const_iterator iter =
             iterN->treeNode.getChildren().begin();
         iter != iterN->treeNode.getChildren().end(); ++iter) {
      mylib::uint16* imgPtr = (mylib::uint16*)((*iter)->dataPtr);
      for (vector<uint64>::const_iterator iterS = (*iter)->PixelIdxList.begin();
           iterS != (*iter)->PixelIdxList.end(); ++iterS) {
        // calculate coordinates of this point
        auxCoord2 = *iterS;
        wAux = imgPtr[auxCoord2];
        for (int ii = 0; ii < dimsImage; ii++) {
          auxCoord1 = auxCoord2 % (supervoxel::dataDims[ii]);
          auxCoord2 -= auxCoord1;
          auxCoord2 /= supervoxel::dataDims[ii];
          auxCentroid[ii] += wAux * ((float)(auxCoord1));
        }
        w += wAux;
      }
    }
  }

  for (int ii = 0; ii < dimsImage; ii++)
    iterN->centroid[ii] = auxCentroid[ii] / w;

  return 0;
}

//===========================================================================
uint64 lineageHyperTree::debugNumberOfTotalNuclei() {
  uint64 ll = 0;
  for (unsigned int ii = 0; ii < maxTM; ii++) ll += nucleiList[ii].size();

  return ll;
}

//============================================================================
ParentTypeSupervoxel lineageHyperTree::addNucleusFromSupervoxel(
    unsigned int TM, ChildrenTypeNucleus& sv) {
  if (TM >= maxTM) {
    cout << "ERROR: at lineageHyperTree::addNucleusFromSupervoxel: time point "
            "out of bounds"
         << endl;
    exit(3);
  }

  nucleiList[TM].push_back(nucleus(sv->TM, sv->centroid));

  ParentTypeSupervoxel iterN =
      (++(nucleiList[TM].rbegin())).base();  // get added element

  // update hypergraph
  sv->treeNode.setParent(iterN);
  iterN->treeNode.addChild(sv);

  return iterN;
}

//================================================================
void lineageHyperTree::parseImagePath(string& imgRawPath, int frame) {
  int intPrecision = 0;
  size_t found = imgRawPath.find_first_of("?");
  while ((imgRawPath[found] == '?') && found != string::npos) {
    intPrecision++;
    found++;
  }

  found = imgRawPath.find_first_of("?");
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
  while (found != string::npos) {
    imgRawPath.replace(found, intPrecision, itoaTM);
    found = imgRawPath.find_first_of("?");
  }
}

//=====================================================================
// I am estimating parameters as if they were no priors, given that I assume
// r_nk = 1.0 and the super-voxels define the associations
// nu_k = alpha_k = beta_k = N_k since there are no priors
template <class imgTypeC>
void lineageHyperTree::calculateGMMparametersFromNuclei(
    list<nucleus>::iterator& iterN, float* m_k, float* N_k, float* S_k) {
  if (iterN->treeNode.getNumChildren() == 0) {
    cout << "ERROR: lineageHyperTree::calculateGMMparametersFromNuclei: nuclei "
            "has no supervoxel as children"
         << endl;
    cout << "Nucleus information: " << *iterN << endl;
    exit(3);
  }

  if (dimsImage != 3) {
    cout << "ERROR: lineageHyperTree::calculateGMMparametersFromNuclei: code "
            "only ready for dimsImage =3 in order to be optimized"
         << endl;
    exit(3);
  }

  *N_k = 0.0;
  memset(m_k, 0, sizeof(float) * dimsImage);
  memset(S_k, 0, sizeof(float) * dimsImage * (1 + dimsImage) / 2);
  int64 coord[dimsImage];
  int count;
  for (vector<ChildrenTypeNucleus>::iterator iterS =
           iterN->treeNode.getChildren().begin();
       iterS != iterN->treeNode.getChildren().end(); ++iterS) {
    imgTypeC* imgPtr = (imgTypeC*)((*iterS)->dataPtr);
    mylib::float32 imgVal;
    int64 coordAux;
    for (vector<uint64>::iterator iterP = (*iterS)->PixelIdxList.begin();
         iterP != (*iterS)->PixelIdxList.end(); ++iterP) {
      coordAux = (*iterP);
      imgVal = imgPtr[coordAux];
      for (int aa = 0; aa < dimsImage - 1; aa++) {
        coord[aa] = coordAux % (supervoxel::dataDims[aa]);
        coordAux -= coord[aa];
        coordAux /= (supervoxel::dataDims[aa]);
      }
      coord[dimsImage - 1] = coordAux;

      (*N_k) += imgVal;
      count = 0;
      for (int ii = 0; ii < dimsImage; ii++) {
        m_k[ii] += imgVal * coord[ii];
        for (int jj = ii; jj < dimsImage; jj++) {
          S_k[count] += imgVal * coord[ii] * coord[jj];
          count++;
        }
      }
    }
  }

  // finish calculating sufficient statistics
  for (int ii = 0; ii < dimsImage; ii++) m_k[ii] /= (*N_k);

  count = 0;
  for (int ii = 0; ii < dimsImage; ii++) {
    for (int jj = ii; jj < dimsImage; jj++) {
      S_k[count] = S_k[count] / (*N_k) - m_k[ii] * m_k[jj];
      count++;
    }
  }
}

//================================================================================================================================
// se use this function sequentially, so we assume we only need to look at TM-1
// to delete elements (call if sequentially if you want to really delete a range
// of TM). The most difficult part is when nuclei divide since we have to
// generate two different sublineages for each of the daughters
void lineageHyperTree::setFrameAsT_o(unsigned int TM) {
  if (TM == 0 || TM >= maxTM)  // nothing to do
    return;

  TreeNode<ChildrenTypeLineage>* aux;
  TreeNode<ChildrenTypeLineage>* auxCh;
  for (list<nucleus>::iterator iterN = nucleiList[TM - 1].begin();
       iterN != nucleiList[TM - 1].end();
       ++iterN)  // we have to explore from the parent perspective, otherwise we
                 // won't eliminate dead cells
  {
    aux = iterN->treeNodePtr;
    if (aux->parent != NULL ||
        aux != iterN->treeNode.getParent()->bt.pointer_mainRoot()) {
      cout << "ERROR: at lineageHyperTree::setFrameAsT_o: parent should be "
              "NULL since we assume this function is applied sequentially"
           << endl;
      exit(4);
    }

    switch (aux->getNumChildren()) {
      case 0:  // lineage ends here->erase lineage
      {
        list<lineage>::iterator iterLaux = aux->data->treeNode.getParent();
        iterLaux->bt.clear();          // release memory form bt
        lineagesList.erase(iterLaux);  // erase lineage
        // iterN = nucleiList[aux->data->TM].erase(aux->data);//erase nucleus.
        // All nuclei and supervoxels for TM-1 will be removed at the end
        break;
      }

      case 1:  // easy case (and most common): just displacement
      {
        if (aux->left != NULL)
          auxCh = aux->left;
        else
          auxCh = aux->right;

        // set child as new root
        auxCh->parent = NULL;
        iterN->treeNode.getParent()->bt.SetMainRootToNULL();
        iterN->treeNode.getParent()->bt.SetMainRoot(auxCh);
        iterN->treeNode.getParent()->bt.SetCurrent(auxCh);

        // delete lineage node and nucleus
        // iterN = nucleiList[aux->data->TM].erase(aux->data);//erase nucleus.
        // All nuclei and supervoxels for TM-1 will be removed at the end
        delete aux;
        break;
      }

      case 2:  // cell division: we have to create a new lineage for one of teh
               // daughters
        {
          // set left child as new root of the current lineage
          auxCh = aux->left;
          auxCh->parent = NULL;
          iterN->treeNode.getParent()->bt.SetMainRootToNULL();
          iterN->treeNode.getParent()->bt.SetMainRoot(auxCh);
          iterN->treeNode.getParent()->bt.SetCurrent(auxCh);

          // set right child as root of a new lineage
          auxCh = aux->right;
          auxCh->parent = NULL;

          lineagesList.push_back(lineage());
          list<lineage>::iterator iterL =
              ((++(lineagesList.rbegin()))
                   .base());  // iterator for the last element in the list
          // iterN->treeNode.getParent()->bt.SetMainRootToNULL();
          iterL->bt.SetMainRoot(auxCh);
          iterL->bt.SetCurrent(auxCh);
          // change all nuclei in the right child to point to iterL to have
          // correct hypergraph
          TreeNode<ChildrenTypeLineage>* auxQ;
          queue<TreeNode<ChildrenTypeLineage>*> q;
          q.push(auxCh);
          while (q.empty() == false) {
            auxQ = q.front();
            q.pop();
            if (auxQ->left != NULL) q.push(auxQ->left);
            if (auxQ->right != NULL) q.push(auxQ->right);

            auxQ->data->treeNode.setParent(iterL);
          }

          // delete lineage node and nucleus
          // iterN = nucleiList[aux->data->TM].erase(aux->data);//erase nucleus.
          // All nuclei and supervoxels for TM-1 will be removed at the end
          delete aux;

          break;
        }
    }
  }

  // delete all nuclei at TM-1
  nucleiList[TM - 1].clear();
  // remove supervoxels
  if (supervoxelsList[TM - 1].empty() == false) {
    if (supervoxelsList[TM - 1].begin()->dataPtr != NULL)
      free(supervoxelsList[TM - 1]
               .begin()
               ->dataPtr);  // free because it was allocated using mylib

    supervoxelsList[TM - 1]
        .clear();  // note: it does not free dataPtr for each supervoxel
  }
}

//===============================================================================================================
int lineageHyperTree::numberOfSeparatedSupervoxelClustersInNucleus(
    TreeNode<ChildrenTypeLineage>* rootNuclei, int conn3D, int64* neighOffset,
    size_t minNeighboringVoxels, vector<int>& supervoxelIdx) {
  list<nucleus>::iterator iterN = rootNuclei->data;
  size_t numS = iterN->treeNode.getNumChildren();

  if (numS == 0) {
    supervoxelIdx.clear();
    return numS;
  } else if (numS == 1) {
    supervoxelIdx.resize(1);
    supervoxelIdx[0] = 0;
    return numS;
  }

  int numClusters = 0;
  supervoxelIdx.resize(numS);
  for (size_t ii = 0; ii < numS; ii++)
    supervoxelIdx[ii] = -1;  // not visited yet

  int idxS;
  vector<uint64> PixelIdxListD, PixelIdxListM;
  PixelIdxListD.reserve(300);
  PixelIdxListM.reserve(300);
  for (size_t ii = 0; ii < numS; ii++) {
    if (supervoxelIdx[ii] >= 0)  // already visited and assigned
      continue;
    // found a new seed for a cluster
    queue<int> q;
    q.push(ii);

    while (q.empty() == false) {
      idxS = q.front();
      q.pop();
      supervoxelIdx[idxS] = numClusters;
      for (size_t jj = 0; jj < numS; jj++) {
        if (supervoxelIdx[jj] >= 0)  // already visited and assigned
          continue;
        // calculate if supervoxels satisfy connectivity criteria
        iterN->treeNode.getChildren()[idxS]->neighboringVoxels(
            *(iterN->treeNode.getChildren()[jj]), conn3D, neighOffset,
            PixelIdxListD, PixelIdxListM);

        if (PixelIdxListD.size() >=
            minNeighboringVoxels)  // supervoxels are connected
        {
          q.push(jj);
        }
      }
    }
    numClusters++;
  }

  return numClusters;
}

//====================================================================================
void lineageHyperTree::debugPrintGMMwithDisconnectedSupervoxels(
    unsigned int TM, int minNumDisconnectedSv, int conn3D,
    size_t minNeighboringVoxels, vector<int>& nucleiIdx) {
  if (TM >= maxTM) return;

  int64 boundarySize[dimsImage];
  int64* neighOffset =
      supervoxel::buildNeighboorhoodConnectivity(conn3D, boundarySize);

  nucleiIdx.clear();
  vector<int> vecAux;
  int count = 0;
  for (list<nucleus>::iterator iterN = nucleiList[TM].begin();
       iterN != nucleiList[TM].end(); ++iterN, count++) {
    if (numberOfSeparatedSupervoxelClustersInNucleus(
            iterN->treeNodePtr, conn3D, neighOffset, minNeighboringVoxels,
            vecAux) >= minNumDisconnectedSv) {
      nucleiIdx.push_back(count);
    }
  }

  delete[] neighOffset;
}

int lineageHyperTree::writeListSupervoxelsToBinaryFile(string filename,
                                                       unsigned int TM) {
  if (TM >= maxTM) return 0;

  ofstream out(filename.c_str(), ios::binary | ios::out);

  if (!out.is_open()) {
    cout << "ERROR: at lineageHyperTree::writeListSupervoxelsToBinaryFile: "
            "could not open file "
         << filename << " to save supervoxels" << endl;
    return 1;
  }

  // write number of supevoxels
  int aux = supervoxelsList[TM].size();
  out.write((char*)(&aux), sizeof(int));

  // write supervoxels
  for (list<supervoxel>::iterator iter = supervoxelsList[TM].begin();
       iter != supervoxelsList[TM].end(); ++iter)
    iter->writeToBinary(out);

  out.close();
  return 0;
}
//=========================================================================================
int lineageHyperTree::writeLineageInArray(string filename)  // saves lineage
                                                            // (only nuclei
                                                            // centroids) as an
                                                            // array with
                                                            // columns id, type,
                                                            // x, y, z, radius,
                                                            // parent_id, time,
                                                            // confidence
{
  ofstream fout(filename.c_str(), ios::binary | ios::out);

  if (fout.is_open() == false) {
    cout << "ERROR: lineageHyperTree::writeLineageInArray: file " << filename
         << " could not be opened to write to" << endl;
    return 2;
  }

  // find out number of total points
  int numPts = 0;
  for (unsigned int ii = 0; ii < maxTM; ii++) {
    numPts += nucleiList[ii].size();
  }

  // write number of time points
  int aux = maxTM;
  fout.write((char*)(&aux), sizeof(int));
  fout.write((char*)(&numPts), sizeof(int));
  int nodeId = 0;
  float row[9];  // it is faster to save everything at once as float (I can
                 // parse later)
  for (unsigned int ii = 0; ii < maxTM; ii++) {
    // write number of nuclei in this time point
    aux = nucleiList[ii].size();
    fout.write((char*)(&aux), sizeof(int));

    row[1] = 0;      // type not used yet
    row[5] = -1.0f;  // radius not used yet
    row[7] = ii;
    for (list<nucleus>::iterator iterN = nucleiList[ii].begin();
         iterN != nucleiList[ii].end(); ++iterN) {
      iterN->treeNodePtr->nodeId = nodeId;
      row[0] = nodeId;

      row[2] = iterN->centroid[0];
      row[3] = iterN->centroid[1];
      row[4] = iterN->centroid[2];

      row[6] = -1;
      if (iterN->treeNodePtr->parent != NULL)
        row[6] = iterN->treeNodePtr->parent->nodeId;  // I save ordered by time,
                                                      // so this nodeId has been
                                                      // updated already
      row[8] = iterN->confidence;

      fout.write((char*)(row), sizeof(float) * 9);

      nodeId++;
    }
  }

  fout.close();

  return 0;
}

//======================================================================================
int lineageHyperTree::readLineageInArray(string filename) {
  clear();
  ifstream fin(filename.c_str(), ios::binary | ios::in);

  if (fin.is_open() == false) {
    cout << "ERROR: lineageHyperTree::writeLineageInArray: file " << filename
         << " could not be opened to read from" << endl;
    return 2;
  }

  int numTM, numPts;
  fin.read((char*)(&numTM), sizeof(int));
  fin.read((char*)(&numPts), sizeof(int));

  if (numTM > (int)maxTM) {
    cout << "ERROR: lineageHyperTree::writeLineageInArray: lineage hyper tree "
            "does not have enough time points to allocate reading"
         << endl;
    return 3;
  }

  float row[9];
  int numNodes, nodeId = 0, parentId;
  vector<TreeNode<ChildrenTypeLineage>*> mapNodeId2ptr(numPts);
  vector<list<lineage>::iterator> mapNodeId2lineage(numPts);
  list<lineage>::iterator iterL;
  list<nucleus>::iterator iterN;
  TreeNode<ChildrenTypeLineage>* par;
  for (int ii = 0; ii < numTM; ii++) {
    fin.read((char*)(&numNodes), sizeof(int));

    for (int jj = 0; jj < numNodes; jj++) {
      fin.read((char*)(row), sizeof(float) * 9);
      parentId = (int)row[6];
      if (parentId < 0)  // new lineage
      {
        lineagesList.push_back(lineage());
        iterL = ((++(lineagesList.rbegin()))
                     .base());  // iterator for the last element in the list
        par = NULL;
      } else {
        par = mapNodeId2ptr[parentId];
        iterL = mapNodeId2lineage[parentId];
        iterL->bt.SetCurrent(par);  // prepare for insertion
      }

      // insert nucleus
      nucleiList[ii].push_back(nucleus(ii, &(row[2])));
      iterN = ((++(nucleiList[ii].rbegin())).base());

      iterN->treeNode.setParent(iterL);
      iterN->treeNodePtr = iterL->bt.insert(iterN);
      mapNodeId2ptr[nodeId] = iterN->treeNodePtr;

      // confidence
      iterN->confidence = row[8];  // confidence

      mapNodeId2lineage[nodeId] = iterL;

      nodeId++;
    }
  }

  fin.close();

  return 0;
}

//=========================================================================================
int lineageHyperTree::confidenceScoreForNucleus(
    TreeNode<ChildrenTypeLineage>* rootConfidence, float thrDist2) {
  if (rootConfidence == NULL) return 0;

  int confidence = 3;  // default confidence is 3 (5 is reserved for human
                       // revieing and correction)

  // death or cell division
  if (rootConfidence->getNumChildren() != 1) confidence = 0;

  // birth not in the first frame
  if (rootConfidence->parent == NULL && rootConfidence->data->TM != 0)
    confidence = 0;

  // large displacements
  if (rootConfidence->parent != NULL &&
      rootConfidence->data->Euclidean2Distance(
          *(rootConfidence->parent->data), supervoxel::getScale()) > thrDist2)
    confidence = 0;

  return confidence;
}

//=========================================================================================
int lineageHyperTree::debugCheckOneToOneCorrespondenceNucleiSv(int TM) {
  cout << "---------------- DEBUGGING:  "
          "lineageHyperTree::debugCheckOneToOneCorrespondenceNucleiSv "
          "--------------------"
       << endl;
  // check that the one to one correspondence exists
  int count = 0;
  for (list<nucleus>::iterator iterN = nucleiList[TM].begin();
       iterN != nucleiList[TM].end(); ++iterN, count++) {
    if (iterN->treeNode.getNumChildren() > 1) {
      cout << "ERROR: "
              "lineageHyperTree::debugCheckOneToOneCorrespondenceNucleiSv: "
              "nuclei "
           << count << " has more than 1 supervoxel" << endl;
      return count;
    }
  }
  return 0;
}

//==============================================================================
// predefine templates
template void lineageHyperTree::calculateGMMparametersFromNuclei<float>(
    list<nucleus>::iterator& iterN, float* m_k, float* N_k, float* S_k);
template void
lineageHyperTree::calculateGMMparametersFromNuclei<unsigned short int>(
    list<nucleus>::iterator& iterN, float* m_k, float* N_k, float* S_k);
template void lineageHyperTree::calculateGMMparametersFromNuclei<unsigned char>(
    list<nucleus>::iterator& iterN, float* m_k, float* N_k, float* S_k);

template int lineageHyperTree::calculateNucleiIntensityCentroid<float>(
    SibilingTypeNucleus& iterN);
template int lineageHyperTree::calculateNucleiIntensityCentroid<unsigned char>(
    SibilingTypeNucleus& iterN);
template int lineageHyperTree::calculateNucleiIntensityCentroid<
    unsigned short int>(SibilingTypeNucleus& iterN);
