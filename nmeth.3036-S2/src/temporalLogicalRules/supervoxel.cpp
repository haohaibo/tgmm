/*
 * See license.txt for full license and copyright notice.
 *
 * \brief
 *
 */

#include "supervoxel.h"
#include <algorithm>
#include <cmath>
#include <queue>

#include "knnCUDA/knnCuda.h"
#include "utilsAmatf.h"

//====================================================
// to sort based on intensity
struct VoxelFA {
  float voxelVal;   // value
  uint64 voxelIdx;  // position in the image

  friend bool operator<(
      VoxelFA& lhs,
      VoxelFA& rhs);  // descending order (we have inverted the order)
};
bool operator<(VoxelFA& lhs, VoxelFA& rhs) {
  return rhs.voxelVal < lhs.voxelVal;
}

//====================================================
// static variables
uint64 supervoxel::dataDims[dimsImage];
uint64 supervoxel::dataSizeInBytes;
float supervoxel::scale[dimsImage];
unsigned int supervoxel::KmaxNumNN;  // maximum nearest neighbors
float supervoxel::KmaxDistKNN;  // maximum distance to select nearest neighbors
paramMergeSplit supervoxel::pMergeSplit;  //

// for trimming
int supervoxel::maxSizeTrim;
float supervoxel::maxPrecentileTrim;
int supervoxel::conn3Dtrim;
int64* supervoxel::neighOffsetTrim;
int64 supervoxel::boundarySizeTrim[dimsImage];

unsigned int supervoxel::getmaxKNNCUDA() { return maxKNN; }

ostream& operator<<(std::ostream& os, const supervoxel m) {
  os << "Centroid = ";
  for (int ii = 0; ii < dimsImage; ii++) os << m.centroid[ii] << " ";
  os << "; TM = " << m.TM;
  os << "; size(pixels) = " << m.PixelIdxList.size();
  return os;
}

supervoxel& supervoxel::operator=(const supervoxel& p) {
  if (this != &p) {
    PixelIdxList = p.PixelIdxList;
    memcpy(centroid, p.centroid, sizeof(float) * dimsImage);
    memcpy(precisionW, p.precisionW,
           sizeof(float) * dimsImage * (1 + dimsImage) / 2);
    intensity = p.intensity;
    TM = p.TM;
    dataPtr = p.dataPtr;
    treeNode = p.treeNode;
    probClassifier = p.probClassifier;
    nearestNeighborsInSpace = p.nearestNeighborsInSpace;
    nearestNeighborsInTimeForward = p.nearestNeighborsInTimeForward;
    nearestNeighborsInTimeBackward = p.nearestNeighborsInTimeBackward;

    nodeHSptr = p.nodeHSptr;
    localBackgroundSubtracted = p.localBackgroundSubtracted;
  }
  return *this;
}
// copy constructor
supervoxel::supervoxel(const supervoxel& p) {
  PixelIdxList = p.PixelIdxList;
  memcpy(centroid, p.centroid, sizeof(float) * dimsImage);
  memcpy(precisionW, p.precisionW,
         sizeof(float) * dimsImage * (1 + dimsImage) / 2);
  intensity = p.intensity;
  TM = p.TM;
  dataPtr = p.dataPtr;
  treeNode = p.treeNode;
  probClassifier = p.probClassifier;

  nearestNeighborsInSpace = p.nearestNeighborsInSpace;
  nearestNeighborsInTimeForward = p.nearestNeighborsInTimeForward;
  nearestNeighborsInTimeBackward = p.nearestNeighborsInTimeBackward;

  nodeHSptr = p.nodeHSptr;
  localBackgroundSubtracted = p.localBackgroundSubtracted;
}

// constructoir with default values
supervoxel::supervoxel() {
  PixelIdxList.reserve(200);  // reserve to avoid initial realloc
  dataPtr = NULL;
  TM = -1;

  nodeHSptr = NULL;
  localBackgroundSubtracted = false;
  // centroid and dataDims can be left at random initialization
}
supervoxel::supervoxel(int TM_) {
  PixelIdxList.reserve(200);  // reserve to avoid realloc
  dataPtr = NULL;
  TM = TM_;

  nodeHSptr = NULL;
  localBackgroundSubtracted = false;
  // centroid and dataDims can be left at random initialization
}
supervoxel::~supervoxel() {
  PixelIdxList.clear();
  // supervoxel calss does not take care of freeing dataPtr
}

supervoxel::supervoxel(istream& is)  // create supervoxel from biunary file
{
  is.read((char*)(&TM), sizeof(int));
  is.read((char*)(&dataSizeInBytes), sizeof(uint64));
  is.read((char*)(&dataDims), sizeof(uint64) * dimsImage);
  unsigned int ll;
  is.read((char*)(&ll), sizeof(unsigned int));
  PixelIdxList.resize(ll);
  is.read((char*)(&(PixelIdxList[0])), sizeof(uint64) * ll);

  // set the otehr properties to default
  dataPtr = NULL;

  nodeHSptr = NULL;
  localBackgroundSubtracted = false;
}
void supervoxel::writeToBinary(ostream& os)  // write ot binary file
{
  os.write((char*)(&TM), sizeof(int));
  os.write((char*)(&dataSizeInBytes), sizeof(uint64));
  os.write((char*)(&dataDims), sizeof(uint64) * dimsImage);
  unsigned int ll = PixelIdxList.size();
  os.write((char*)(&ll), sizeof(unsigned int));
  os.write((char*)(&(PixelIdxList[0])), sizeof(uint64) * ll);
}
//====================================================
inline uint64 supervoxel::getDataSize() {
  uint64 size = dataDims[0];
  for (int ii = 1; ii < dimsImage; ii++) size *= dataDims[ii];
  return size;
}

//==================================================
int supervoxel::debugPixelIdxListIsSorted() {
  cout << "DEBUGGING: at supervoxel::debugPixelIdxListIsSorted: supervoxel "
       << (*this) << endl;
  if (PixelIdxList.empty()) return 1;

  uint64 pOld = PixelIdxList[0];
  for (vector<uint64>::iterator iter = PixelIdxList.begin() + 1;
       iter != PixelIdxList.end(); ++iter) {
    if (pOld >= (*iter)) {
      cout << "ERROR: not sorted!!!!" << endl;
      return 0;
    }
    pOld = (*iter);
  }

  return 1;
}

//=============================================================================================================
int supervoxel::neighboringVoxels(const supervoxel& s, int conn3D,
                                  int64* neighOffset,
                                  vector<uint64>& PixelIdxListBorderThis,
                                  vector<uint64>& PixelIdxListBorderS) {
  // delete memory
  PixelIdxListBorderS.clear();
  PixelIdxListBorderThis.clear();

  // quick check for far away supervoxels
  if (s.PixelIdxList.empty()) return 0;
  if (PixelIdxList.empty()) return 0;
  if (s.PixelIdxList[0] >
      PixelIdxList[PixelIdxList.size() - 1] + neighOffset[conn3D - 1])
    return 0;
  if ((int64)(s.PixelIdxList[s.PixelIdxList.size() - 1]) <
      (int64)(PixelIdxList[0]) + neighOffset[0])
    return 0;

  // exhaustive check
  for (int ii = 0; ii < conn3D; ii++) {
    int64 offset = neighOffset[ii];
    size_t pos1 = 0, pos2 = 0;
    int64 val1 = (int64)(PixelIdxList[0]) + offset, val2 = s.PixelIdxList[0];

    while (pos1 < PixelIdxList.size() - 1 &&
           pos2 < s.PixelIdxList.size() -
                      1)  //-1 to avoid if inside when we do posXX++
    {
      if (val1 == val2)  // found a neighboring voxel
      {
        PixelIdxListBorderS.push_back(val2);
        PixelIdxListBorderThis.push_back(PixelIdxList[pos1]);
        pos1++;
        val1 = (int64)(PixelIdxList[pos1]) + offset;
        pos2++;
        val2 = s.PixelIdxList[pos2];
      } else if (val1 < val2) {
        pos1++;
        val1 = (int64)(PixelIdxList[pos1]) + offset;
      } else {
        pos2++;
        val2 = s.PixelIdxList[pos2];
      }
    }

    // final comparison
    if (pos1 == PixelIdxList.size() - 1) {
      for (; pos2 < s.PixelIdxList.size(); pos2++) {
        val2 = s.PixelIdxList[pos2];
        if (val1 == val2) {
          PixelIdxListBorderS.push_back(val2);
          PixelIdxListBorderThis.push_back(PixelIdxList[pos1]);
          break;
        }
      }
    } else {
      for (; pos1 < PixelIdxList.size(); pos1++) {
        val1 = (int64)(PixelIdxList[pos1]) + offset;
        if (val1 == val2) {
          PixelIdxListBorderS.push_back(val2);
          PixelIdxListBorderThis.push_back(PixelIdxList[pos1]);
          break;
        }
      }
    }
  }

  // keep only unique elements for each vector
  sort(PixelIdxListBorderS.begin(), PixelIdxListBorderS.end());
  sort(PixelIdxListBorderThis.begin(), PixelIdxListBorderThis.end());

  vector<uint64>::iterator iter =
      unique(PixelIdxListBorderS.begin(), PixelIdxListBorderS.end());
  PixelIdxListBorderS.resize(iter - PixelIdxListBorderS.begin());

  iter = unique(PixelIdxListBorderThis.begin(), PixelIdxListBorderThis.end());
  PixelIdxListBorderThis.resize(iter - PixelIdxListBorderThis.begin());

  return 0;
}
//==================================================================================================

//=============================================================================================================
int supervoxel::neighboringVoxels(const supervoxel& s, int conn3D,
                                  int64* neighOffset,
                                  int satNeighboringVoxels) {
  cout << "ERROR: at supervoxel::neighboringVoxels: code not finished!!!"
       << endl;
  exit(3);

  // quick check for far away supervoxels
  if (s.PixelIdxList.empty()) return 0;
  if (PixelIdxList.empty()) return 0;
  if (s.PixelIdxList[0] >
      PixelIdxList[PixelIdxList.size() - 1] + neighOffset[conn3D - 1])
    return 0;
  if ((int64)(s.PixelIdxList[s.PixelIdxList.size() - 1]) <
      (int64)(PixelIdxList[0]) + neighOffset[0])
    return 0;

  // exhaustive check
  int numNeighboringElem = 0;
  for (int ii = 0; ii < conn3D; ii++) {
    int64 offset = neighOffset[ii];
    size_t pos1 = 0, pos2 = 0;
    int64 val1 = (int64)(PixelIdxList[0]) + offset, val2 = s.PixelIdxList[0];

    while (pos1 < PixelIdxList.size() - 1 &&
           pos2 < s.PixelIdxList.size() -
                      1)  //-1 to avoid if inside when we do posXX++
    {
      if (val1 == val2)  // found a neighboring voxel
      {
        // PixelIdxListBorderS.push_back(val2);
        // PixelIdxListBorderThis.push_back( PixelIdxList[pos1] );
        pos1++;
        val1 = (int64)(PixelIdxList[pos1]) + offset;
        pos2++;
        val2 = s.PixelIdxList[pos2];
      } else if (val1 < val2) {
        pos1++;
        val1 = (int64)(PixelIdxList[pos1]) + offset;
      } else {
        pos2++;
        val2 = s.PixelIdxList[pos2];
      }
    }

    // final comparison
    if (pos1 == PixelIdxList.size() - 1) {
      for (; pos2 < s.PixelIdxList.size(); pos2++) {
        val2 = s.PixelIdxList[pos2];
        if (val1 == val2) {
          // PixelIdxListBorderS.push_back(val2);
          // PixelIdxListBorderThis.push_back( PixelIdxList[pos1] );
          break;
        }
      }
    } else {
      for (; pos1 < PixelIdxList.size(); pos1++) {
        val1 = (int64)(PixelIdxList[pos1]) + offset;
        if (val1 == val2) {
          // PixelIdxListBorderS.push_back(val2);
          // PixelIdxListBorderThis.push_back( PixelIdxList[pos1] );
          break;
        }
      }
    }
  }

  // keep only unique elements for each vector
  // sort(PixelIdxListBorderS.begin(), PixelIdxListBorderS.end());
  // sort(PixelIdxListBorderThis.begin(), PixelIdxListBorderThis.end());

  // vector<uint64>::iterator iter = unique(PixelIdxListBorderS.begin(),
  // PixelIdxListBorderS.end());
  // PixelIdxListBorderS.resize(iter - PixelIdxListBorderS.begin());

  // iter = unique(PixelIdxListBorderThis.begin(),
  // PixelIdxListBorderThis.end());
  // PixelIdxListBorderThis.resize(iter - PixelIdxListBorderThis.begin());

  return 0;
}

//============================================================================================================
bool supervoxel::isNeighboring(const supervoxel& s, int conn3D,
                               int64* neighOffset) {
  // quick check for far away supervoxels
  if (s.PixelIdxList.empty()) return false;
  if (PixelIdxList.empty()) return false;
  if (s.PixelIdxList[0] >
      PixelIdxList[PixelIdxList.size() - 1] + neighOffset[conn3D - 1])
    return false;
  if ((int64)(s.PixelIdxList[s.PixelIdxList.size() - 1]) <
      (int64)(PixelIdxList[0]) + neighOffset[0])
    return false;

  // exhaustive check
  for (int ii = 0; ii < conn3D; ii++) {
    int64 offset = neighOffset[ii];
    size_t pos1 = 0, pos2 = 0;
    int64 val1 = (int64)(PixelIdxList[0]) + offset, val2 = s.PixelIdxList[0];

    while (pos1 < PixelIdxList.size() - 1 &&
           pos2 < s.PixelIdxList.size() -
                      1)  //-1 to avoid if inside when we do posXX++
    {
      if (val1 == val2)  // found a neighboring voxel
      {
        return true;

      } else if (val1 < val2) {
        pos1++;
        val1 = (int64)(PixelIdxList[pos1]) + offset;
      } else {
        pos2++;
        val2 = s.PixelIdxList[pos2];
      }
    }

    // final comparison
    if (pos1 == PixelIdxList.size() - 1) {
      for (; pos2 < s.PixelIdxList.size(); pos2++) {
        val2 = s.PixelIdxList[pos2];
        if (val1 == val2) return true;
      }
    } else {
      for (; pos1 < PixelIdxList.size(); pos1++) {
        val1 = (int64)(PixelIdxList[pos1]) + offset;
        if (val1 == val2) return true;
      }
    }
  }

  return false;
}

//==========================================================
float supervoxel::neighboringVoxelsSelf(int conn3D, int64* neighOffset) {
  cout << "===WARNING!!::supervoxel::neighboringVoxelsSelf: not tested "
          "yet!!!!!======"
       << endl;

  const uint64 maxJumpX = 5;  // to detect "border"pixels(at least in 2D) and
                              // exclude them from the counting

  int mm = 0;  // mean counter
  int count = 0;

  int64 p;
  for (size_t ii = 0; ii < PixelIdxList.size() - 1; ii++) {
    p = PixelIdxList[ii];
    if (PixelIdxList[ii] - p > maxJumpX)
      continue;  // we are in a border pixel in the supervoxel (it would distor
                 // the counting)

    count++;
    for (int jj = 0; jj < conn3D; jj++) {
      if (std::binary_search(PixelIdxList.begin(), PixelIdxList.end(),
                             p + neighOffset[jj]) == true)  // element found
        mm++;
    }
  }

  if (count == 0)
    return conn3D;  // all border elements
  else
    return ((float)mm / (float)count);
}

//==========================================================
// the main idea is to "shoot" lines long X and count gaps. Inspired by how
// mylib stores arbitrary regions
int supervoxel::numHoles() {
  const uint64 maxJumpX =
      dataDims[0] / 3;  // to avoid checking coordinates. If the gap is larger
                        // than a third the image, we assume we are in the next
                        // line for the "line scan" of holes
  uint64 count = 0, aux;

  for (size_t ii = 1; ii < PixelIdxList.size(); ii++) {
    aux = PixelIdxList[ii] - PixelIdxList[ii - 1] - 1;
    if (aux > 0 && aux < maxJumpX)  // we are still traversing a line
      count += aux;
  }

  return (int)count;
}

//===========================================================================
// returns if idx belongs to pixelIdxList or not. If you have a clue of the
// position in the list (for example, looking at neighboring elements), search
// would be faster. We use a simple linear search since the list tends to be
// small
bool supervoxel::idxBelongsToPixelIdxListLinearSearch(uint64 idx,
                                                      size_t iniPosSearch) {
  iniPosSearch = min(iniPosSearch, PixelIdxList.size() - 1);

  // PixelIdxList is a sorted list
  int64 ii;
  if (PixelIdxList[iniPosSearch] > idx) {
    for (ii = iniPosSearch - 1; ii >= 0; ii--) {
      if (PixelIdxList[ii] <= idx) break;
    }
  } else {
    for (ii = iniPosSearch; ii < PixelIdxList.size(); ii++) {
      if (PixelIdxList[ii] >= idx) break;
    }
  }

  if (ii >= 0 && ii < PixelIdxList.size()) {
    if (PixelIdxList[ii] == idx)
      return true;
    else
      return false;
  }

  return false;
}

//===================================================================
int64* supervoxel::buildNeighboorhoodConnectivity(int conn3D,
                                                  int64* boundarySize) {
  int64* neighOffset = new int64[conn3D];
  int64 imgDims[dimsImage];
  for (int ii = 0; ii < dimsImage; ii++) {
    imgDims[ii] = dataDims[ii];  // we need int64 to have negative offset
    if (dataDims[ii] <= 0) {
      cout << "ERROR: supervoxel::buildNeighboorhoodConnectivity: dataDims "
              "seem to be wrong (negative or zero ): you need to set them "
           << endl;
      exit(3);
    }
  }
  // int64 boundarySize[dimsImage];//margin we need to calculate connected
  // components
  // VIP: NEIGHOFFSET HAS TO BE IN ORDER (FROM LOWEST OFFSET TO HIGHEST OFFSET
  // IN ORDER TO CALCULATE FAST IF WE ARE IN A BORDER)
  switch (conn3D) {
    case 4:  // 4 connected component (so considering neighbors only in 2D
             // slices)
      {
        neighOffset[1] = -1;
        neighOffset[2] = 1;
        neighOffset[0] = -imgDims[0];
        neighOffset[3] = imgDims[0];

        boundarySize[0] = 1;
        boundarySize[1] = 1;
        boundarySize[2] = 0;
        break;
      }
    case 6:  // 6 connected components as neighborhood
    {
      neighOffset[2] = -1;
      neighOffset[3] = 1;
      neighOffset[1] = -imgDims[0];
      neighOffset[4] = imgDims[0];
      neighOffset[0] = -imgDims[0] * imgDims[1];
      neighOffset[5] = imgDims[0] * imgDims[1];

      boundarySize[0] = 1;
      boundarySize[1] = 1;
      boundarySize[2] = 1;
      break;
    }
    case 8:  // 8 connected component (so considering neighbors only in 2D
             // slices)
      {
        int countW = 0;
        for (int64 yy = -1; yy <= 1; yy++) {
          for (int64 xx = -1; xx <= 1; xx++) {
            if (yy == 0 && xx == 0) continue;  // skipp the middle point
            neighOffset[countW++] = xx + imgDims[0] * yy;
          }
        }

        if (countW != 8) {
          cout << "ERROR: at watershedPersistanceAgglomeration: Window size "
                  "structure has not been completed correctly"
               << endl;
          return NULL;
        }

        boundarySize[0] = 1;
        boundarySize[1] = 1;
        boundarySize[2] = 0;
        break;
      }
    case 26:  // a cube around teh pixel in 3D
    {
      int countW = 0;
      for (int64 zz = -1; zz <= 1; zz++) {
        for (int64 yy = -1; yy <= 1; yy++) {
          for (int64 xx = -1; xx <= 1; xx++) {
            if (zz == 0 && yy == 0 && xx == 0)
              continue;  // skipp the middle point
            neighOffset[countW++] = xx + imgDims[0] * (yy + imgDims[1] * zz);
          }
        }
      }
      if (countW != 26) {
        cout << "ERROR: at watershedPersistanceAgglomeration: Window size "
                "structure has not been completed correctly"
             << endl;
        return NULL;
      }
      boundarySize[0] = 1;
      boundarySize[1] = 1;
      boundarySize[2] = 1;
      break;
    }
    case 74:  // especial case for nuclei in DLSM: [2,2,1] radius windows to
              // make sure local maxima are not that local
      {
        int countW = 0;
        for (int64 zz = -1; zz <= 1; zz++)
          for (int64 yy = -2; yy <= 2; yy++)
            for (int64 xx = -2; xx <= 2; xx++) {
              if (zz == 0 && yy == 0 && xx == 0)
                continue;  // skipp the middle point
              neighOffset[countW++] = xx + imgDims[0] * (yy + imgDims[1] * zz);
            }
        if (countW != 74) {
          cout << "ERROR: at watershedPersistanceAgglomeration: Window size "
                  "structure has not been completed correctly"
               << endl;
          return NULL;
        }
        boundarySize[0] = 2;
        boundarySize[1] = 2;
        boundarySize[2] = 1;
        break;
      }
    case 75:  // especial case to coarsely xplore negihborhood around
              // [+-6,+-3,0] in XY and [+-1,0] in Z. It is also 74 elements but
              // we need to distinguish it from above
      {
        int countW = 0;
        for (int64 zz = -1; zz <= 1; zz++)
          for (int64 yy = -6; yy <= 6; yy += 3)
            for (int64 xx = -6; xx <= 6; xx += 3) {
              if (zz == 0 && yy == 0 && xx == 0)
                continue;  // skipp the middle point
              neighOffset[countW++] = xx + imgDims[0] * (yy + imgDims[1] * zz);
            }
        if (countW != 74) {
          cout << "ERROR: at watershedPersistanceAgglomeration: Window size "
                  "structure has not been completed correctly"
               << endl;
          return NULL;
        }
        boundarySize[0] = 6;
        boundarySize[1] = 6;
        boundarySize[2] = 1;
        break;
      }

    default:
      cout << "ERROR: at supervoxel::buildNeighboorhoodConnectivity Code not "
              "ready for these connected components"
           << endl;
      return NULL;
  }

  return neighOffset;
}

//====================================================================================
void supervoxel::mergeSupervoxels(vector<supervoxel*>& src) {
  if (src.empty() == true) return;

  float w = (float)(PixelIdxList.size());
  for (int ii = 0; ii < dimsImage; ii++) centroid[ii] *= w;
  float aux;
  for (vector<supervoxel*>::iterator iterS = src.begin(); iterS != src.end();
       ++iterS) {
    if ((this) == (*iterS))  // we are trying to merge teh same supervoxel
      continue;

    if (TM != (*iterS)->TM) {
      cout << "ERROR: at mergeSupervoxel: TM are different. We cannot merge "
              "supervoxels"
           << endl;
      exit(3);
    }

    if (dataPtr != (*iterS)->dataPtr) {
      cout << "ERROR: at mergeSupervoxel: dataPtr. We cannot merge supervoxels"
           << endl;
      exit(3);
    }

    // merge Pixeld Idx List
    PixelIdxList.insert(PixelIdxList.end(), (*iterS)->PixelIdxList.begin(),
                        (*iterS)->PixelIdxList.end());

    // add centroid by weighted size (imprecise since we do not use intensity)
    aux = ((float)((*iterS)->PixelIdxList.size()));
    w += aux;
    for (int ii = 0; ii < dimsImage; ii++)
      centroid[ii] += (aux * (*iterS)->centroid[ii]);
  }
  // calculate weighted mean
  for (int ii = 0; ii < dimsImage; ii++) {
    centroid[ii] /= w;
  }
  // sort merged list. TODO: I could create a merge sort function (merge
  // multiple sorted lists) to gain a little bit of performance.
  sort(PixelIdxList.begin(), PixelIdxList.end());
}

//===================================================================================================================================
// calculates nearest neighbors from supervoxelA (query points) into supervoxelB
// (reference points). It stores results in
// supervoxelA[ii].nearestNeighborsInTimeForward
int supervoxel::nearestNeighbors(
    list<supervoxel>& superVoxelA, list<supervoxel>& superVoxelB,
    unsigned int KmaxNumNN, float KmaxDistKNN, int devCUDA,
    vector<vector<SibilingTypeSupervoxel> >& nearestNeighborVec,
    vector<vector<float> >* nearestNeighborDist2Vec)  // you need to setup scale
                                                      // to calculate this
                                                      // property
{
  supervoxel::setKmaxDistKNN(KmaxDistKNN);
  supervoxel::setKmaxNumNN(KmaxNumNN);

  if (superVoxelA.empty() == true) {
    nearestNeighborVec.clear();
    return 0;
  }

  if (KmaxNumNN > maxKNN) {
    cout << "ERROR: at supervoxelNearestNeighborsInTimeForward: maximum number "
            "of NN "
         << KmaxNumNN << " is superior to maxKNN " << maxKNN
         << ". Please recompile knnCUDA.h code with a larger constant" << endl;
    return 2;
  }

  // preallocate memory for all the centroids from supervoxels
  int ref_nb = superVoxelB.size();
  long long int query_nb = superVoxelA.size();  // we find kNN for each query_nb
                                                // point with respect to ref_nb

  int* ind = new int[KmaxNumNN * query_nb];
  float* dist = new float[KmaxNumNN * query_nb];
  float* query_xyz = new float[dimsImage * query_nb];  // stores centroids
  float* ref_xyz = new float[dimsImage * ref_nb];      // stores centroids

  // copy centroids to temporary array
  long long int count = 0, offset;
  for (list<supervoxel>::iterator iter = superVoxelA.begin();
       iter != superVoxelA.end(); ++iter) {
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
  for (list<supervoxel>::iterator iter = superVoxelB.begin();
       iter != superVoxelB.end(); ++iter) {
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
  nearestNeighborVec.resize(superVoxelA.size());
  if (nearestNeighborDist2Vec != NULL)  // save also the distance (square!!!)
    (*nearestNeighborDist2Vec).resize(superVoxelA.size());
  for (list<supervoxel>::iterator iter = superVoxelA.begin();
       iter != superVoxelA.end(); ++iter) {
    if (nearestNeighborDist2Vec != NULL)  // save also the distance (square!!!)
    {
      (*nearestNeighborDist2Vec)[count].reserve(KmaxNumNN);
      (*nearestNeighborDist2Vec)[count].clear();
    }
    nearestNeighborVec[count].reserve(KmaxNumNN);
    nearestNeighborVec[count].clear();
    offset = count;
    for (long long int ii = 0; ii < KmaxNumNN; ii++) {
      // cout<<offset<<" "<<dist[offset]<<" "<<ind[offset]<<"; Supervoxel:
      // "<<(*(vecIter[ ind[ offset] ]))<<endl;
      if (dist[offset] < auxMaxDist) {
        nearestNeighborVec[count].push_back(vecIter[ind[offset]]);
        if (nearestNeighborDist2Vec !=
            NULL)  // save also the distance (square!!!)
          (*nearestNeighborDist2Vec)[count].push_back(dist[offset]);
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

//===================================================================================================================================
// calculates nearest neighbors from supervoxelA (query points) into supervoxelB
// (reference points).
int supervoxel::nearestNeighbors(
    list<supervoxel>& superVoxelA, vector<supervoxel>& superVoxelB,
    unsigned int KmaxNumNN, float KmaxDistKNN, int devCUDA,
    vector<vector<vector<supervoxel>::iterator> >& nearestNeighborVec,
    vector<vector<float> >* nearestNeighborDist2Vec)  // you need to setup scale
                                                      // to calculate this
                                                      // property
{
  supervoxel::setKmaxDistKNN(KmaxDistKNN);
  supervoxel::setKmaxNumNN(KmaxNumNN);

  if (superVoxelA.empty() == true) {
    nearestNeighborVec.clear();
    return 0;
  }

  if (KmaxNumNN > maxKNN) {
    cout << "ERROR: at supervoxelNearestNeighborsInTimeForward: maximum number "
            "of NN "
         << KmaxNumNN << " is superior to maxKNN " << maxKNN
         << ". Please recompile knnCUDA.h code with a larger constant" << endl;
    return 2;
  }

  // preallocate memory for all the centroids from supervoxels
  int ref_nb = superVoxelB.size();
  long long int query_nb = superVoxelA.size();  // we find kNN for each query_nb
                                                // point with respect to ref_nb

  int* ind = new int[KmaxNumNN * query_nb];
  float* dist = new float[KmaxNumNN * query_nb];
  float* query_xyz = new float[dimsImage * query_nb];  // stores centroids
  float* ref_xyz = new float[dimsImage * ref_nb];      // stores centroids

  // copy centroids to temporary array
  long long int count = 0, offset;
  for (list<supervoxel>::iterator iter = superVoxelA.begin();
       iter != superVoxelA.end(); ++iter) {
    offset = count;
    for (int ii = 0; ii < dimsImage; ii++) {
      query_xyz[offset] = iter->centroid[ii];
      offset += query_nb;
    }
    count++;
  }

  count = 0, offset;
  vector<vector<supervoxel>::iterator> vecIter(
      ref_nb);  // we will need it later to assign ind to nearest neighbor
  for (vector<supervoxel>::iterator iter = superVoxelB.begin();
       iter != superVoxelB.end(); ++iter) {
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
  nearestNeighborVec.resize(superVoxelA.size());
  if (nearestNeighborDist2Vec != NULL)  // save also the distance (square!!!)
    (*nearestNeighborDist2Vec).resize(superVoxelA.size());
  for (list<supervoxel>::iterator iter = superVoxelA.begin();
       iter != superVoxelA.end(); ++iter) {
    if (nearestNeighborDist2Vec != NULL)  // save also the distance (square!!!)
    {
      (*nearestNeighborDist2Vec)[count].reserve(KmaxNumNN);
      (*nearestNeighborDist2Vec)[count].clear();
    }
    nearestNeighborVec[count].reserve(KmaxNumNN);
    nearestNeighborVec[count].clear();
    offset = count;
    for (long long int ii = 0; ii < KmaxNumNN; ii++) {
      // cout<<offset<<" "<<dist[offset]<<" "<<ind[offset]<<"; Supervoxel:
      // "<<(*(vecIter[ ind[ offset] ]))<<endl;
      if (dist[offset] < auxMaxDist) {
        nearestNeighborVec[count].push_back(vecIter[ind[offset]]);
        if (nearestNeighborDist2Vec !=
            NULL)  // save also the distance (square!!!)
          (*nearestNeighborDist2Vec)[count].push_back(dist[offset]);
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

int supervoxel::getMaxKnnCUDA()  // maximum number of KNN allowed
{
  return maxKNN;
}

//===================================================================================================================================
// calculates nearest neighbors from supervoxelA (query points) into supervoxelB
// (reference points). x_i = supervoxelB[0][i]
int supervoxel::nearestNeighbors(
    vector<float> superVoxelA[dimsImage], vector<float> superVoxelB[dimsImage],
    unsigned int KmaxNumNN, float KmaxDistKNN, int devCUDA,
    vector<vector<int> >& nearestNeighborVec,
    vector<vector<float> >* nearestNeighborDist2Vec)  // you need to setup scale
                                                      // to calculate this
                                                      // property
{
  supervoxel::setKmaxDistKNN(KmaxDistKNN);
  supervoxel::setKmaxNumNN(KmaxNumNN);

  if (superVoxelA[0].empty() == true) {
    nearestNeighborVec.clear();
    return 0;
  }

  if (KmaxNumNN > maxKNN) {
    cout << "ERROR: at supervoxelNearestNeighborsInTimeForward: maximum number "
            "of NN "
         << KmaxNumNN << " is superior to maxKNN " << maxKNN
         << ". Please recompile knnCUDA.h code with a larger constant" << endl;
    return 2;
  }

  // preallocate memory for all the centroids from supervoxels
  int ref_nb = superVoxelB[0].size();
  long long int query_nb = superVoxelA[0].size();  // we find kNN for each
                                                   // query_nb point with
                                                   // respect to ref_nb

  int* ind = new int[KmaxNumNN * query_nb];
  float* dist = new float[KmaxNumNN * query_nb];
  float* query_xyz = new float[dimsImage * query_nb];  // stores centroids
  float* ref_xyz = new float[dimsImage * ref_nb];      // stores centroids

  // copy centroids to temporary array
  long long int count = 0;
  for (int ii = 0; ii < dimsImage; ii++) {
    for (vector<float>::iterator iter = superVoxelA[ii].begin();
         iter != superVoxelA[ii].end(); ++iter) {
      ref_xyz[count] = *iter;
      count++;
    }
  }

  count = 0;
  for (int ii = 0; ii < dimsImage; ii++) {
    for (vector<float>::iterator iter = superVoxelB[ii].begin();
         iter != superVoxelB[ii].end(); ++iter) {
      ref_xyz[count] = *iter;
      count++;
    }
  }
  // calculate nearest neighbors
  int err = knnCUDA_(ind, dist, query_xyz, ref_xyz, query_nb, ref_nb, KmaxNumNN,
                     supervoxel::getScale(), devCUDA);
  if (err > 0) return err;

  // parse results to supervoxel structure
  count = 0;
  long long int offset;
  float auxMaxDist =
      KmaxDistKNN * KmaxDistKNN;  // knNCUDA returns squared distance
  nearestNeighborVec.resize(superVoxelA[0].size());
  if (nearestNeighborDist2Vec != NULL)  // save also the distance (square!!!)
    (*nearestNeighborDist2Vec).resize(superVoxelA[0].size());

  for (count = 0; count < superVoxelA[0].size(); count++) {
    if (nearestNeighborDist2Vec != NULL)  // save also the distance (square!!!)
    {
      (*nearestNeighborDist2Vec)[count].reserve(KmaxNumNN);
      (*nearestNeighborDist2Vec)[count].clear();
    }
    nearestNeighborVec[count].reserve(KmaxNumNN);
    nearestNeighborVec[count].clear();
    offset = count;
    for (long long int ii = 0; ii < KmaxNumNN; ii++) {
      // cout<<offset<<" "<<dist[offset]<<" "<<ind[offset]<<"; Supervoxel:
      // "<<(*(vecIter[ ind[ offset] ]))<<endl;
      if (dist[offset] < auxMaxDist) {
        nearestNeighborVec[count].push_back(ind[offset]);
        if (nearestNeighborDist2Vec !=
            NULL)  // save also the distance (square!!!)
          (*nearestNeighborDist2Vec)[count].push_back(dist[offset]);
      }
      offset += query_nb;
    }
  }

  // release memory
  delete[] ind;
  delete[] dist;
  delete[] query_xyz;
  delete[] ref_xyz;

  return 0;
}
//===================================================================================================================================
// calculates nearest neighbors from supervoxelA (query points) into supervoxelB
// (reference points). x_i = supervoxelB[0][i]
int supervoxel::nearestNeighbors(
    vector<supervoxel*>& superVoxelA, vector<float> superVoxelB[dimsImage],
    unsigned int KmaxNumNN, float KmaxDistKNN, int devCUDA,
    vector<vector<int> >& nearestNeighborVec,
    vector<vector<float> >* nearestNeighborDist2Vec)  // you need to setup scale
                                                      // to calculate this
                                                      // property
{
  supervoxel::setKmaxDistKNN(KmaxDistKNN);
  supervoxel::setKmaxNumNN(KmaxNumNN);

  if (superVoxelA.empty() == true) {
    nearestNeighborVec.clear();
    return 0;
  }

  if (KmaxNumNN > maxKNN) {
    cout << "ERROR: at supervoxelNearestNeighborsInTimeForward: maximum number "
            "of NN "
         << KmaxNumNN << " is superior to maxKNN " << maxKNN
         << ". Please recompile knnCUDA.h code with a larger constant" << endl;
    return 2;
  }

  // preallocate memory for all the centroids from supervoxels
  int ref_nb = superVoxelB[0].size();
  long long int query_nb = superVoxelA.size();  // we find kNN for each query_nb
                                                // point with respect to ref_nb

  int* ind = new int[KmaxNumNN * query_nb];
  float* dist = new float[KmaxNumNN * query_nb];
  float* query_xyz = new float[dimsImage * query_nb];  // stores centroids
  float* ref_xyz = new float[dimsImage * ref_nb];      // stores centroids

  // copy centroids to temporary array
  long long int count = 0, offset;
  for (vector<supervoxel*>::iterator iter = superVoxelA.begin();
       iter != superVoxelA.end(); ++iter) {
    offset = count;
    for (int ii = 0; ii < dimsImage; ii++) {
      query_xyz[offset] = (*iter)->centroid[ii];
      offset += query_nb;
    }
    count++;
  }

  count = 0;
  for (int ii = 0; ii < dimsImage; ii++) {
    for (vector<float>::iterator iter = superVoxelB[ii].begin();
         iter != superVoxelB[ii].end(); ++iter) {
      ref_xyz[count] = *iter;
      count++;
    }
  }
  // calculate nearest neighbors
  int err = knnCUDA_(ind, dist, query_xyz, ref_xyz, query_nb, ref_nb, KmaxNumNN,
                     supervoxel::getScale(), devCUDA);
  if (err > 0) return err;

  // parse results to supervoxel structure
  count = 0;
  float auxMaxDist =
      KmaxDistKNN * KmaxDistKNN;  // knNCUDA returns squared distance
  nearestNeighborVec.resize(superVoxelA.size());
  if (nearestNeighborDist2Vec != NULL)  // save also the distance (square!!!)
    (*nearestNeighborDist2Vec).resize(superVoxelA.size());

  for (count = 0; count < superVoxelA.size(); count++) {
    if (nearestNeighborDist2Vec != NULL)  // save also the distance (square!!!)
    {
      (*nearestNeighborDist2Vec)[count].reserve(KmaxNumNN);
      (*nearestNeighborDist2Vec)[count].clear();
    }
    nearestNeighborVec[count].reserve(KmaxNumNN);
    nearestNeighborVec[count].clear();
    offset = count;
    for (long long int ii = 0; ii < KmaxNumNN; ii++) {
      // cout<<offset<<" "<<dist[offset]<<" "<<ind[offset]<<"; Supervoxel:
      // "<<(*(vecIter[ ind[ offset] ]))<<endl;
      if (dist[offset] < auxMaxDist) {
        nearestNeighborVec[count].push_back(ind[offset]);
        if (nearestNeighborDist2Vec !=
            NULL)  // save also the distance (square!!!)
          (*nearestNeighborDist2Vec)[count].push_back(dist[offset]);
      }
      offset += query_nb;
    }
  }

  // release memory
  delete[] ind;
  delete[] dist;
  delete[] query_xyz;
  delete[] ref_xyz;

  return 0;
}

//===================================================================================================================================
// calculates nearest neighbors from supervoxelA (query points) into supervoxelB
// (reference points).
int supervoxel::nearestNeighbors(
    vector<supervoxel>& superVoxelA, vector<supervoxel>& superVoxelB,
    unsigned int KmaxNumNN, float KmaxDistKNN, int devCUDA,
    vector<vector<vector<supervoxel>::iterator> >& nearestNeighborVec,
    vector<vector<float> >* nearestNeighborDist2Vec)  // you need to setup scale
                                                      // to calculate this
                                                      // property
{
  supervoxel::setKmaxDistKNN(KmaxDistKNN);
  supervoxel::setKmaxNumNN(KmaxNumNN);

  if (superVoxelA.empty() == true) {
    nearestNeighborVec.clear();
    return 0;
  }

  if (KmaxNumNN > maxKNN) {
    cout << "ERROR: at supervoxelNearestNeighborsInTimeForward: maximum number "
            "of NN "
         << KmaxNumNN << " is superior to maxKNN " << maxKNN
         << ". Please recompile knnCUDA.h code with a larger constant" << endl;
    return 2;
  }

  // preallocate memory for all the centroids from supervoxels
  int ref_nb = superVoxelB.size();
  long long int query_nb = superVoxelA.size();  // we find kNN for each query_nb
                                                // point with respect to ref_nb

  int* ind = new int[KmaxNumNN * query_nb];
  float* dist = new float[KmaxNumNN * query_nb];
  float* query_xyz = new float[dimsImage * query_nb];  // stores centroids
  float* ref_xyz = new float[dimsImage * ref_nb];      // stores centroids

  // copy centroids to temporary array
  long long int count = 0, offset;
  for (vector<supervoxel>::iterator iter = superVoxelA.begin();
       iter != superVoxelA.end(); ++iter) {
    offset = count;
    for (int ii = 0; ii < dimsImage; ii++) {
      query_xyz[offset] = iter->centroid[ii];
      offset += query_nb;
    }
    count++;
  }

  count = 0, offset;
  vector<vector<supervoxel>::iterator> vecIter(
      ref_nb);  // we will need it later to assign ind to nearest neighbor
  for (vector<supervoxel>::iterator iter = superVoxelB.begin();
       iter != superVoxelB.end(); ++iter) {
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
  nearestNeighborVec.resize(superVoxelA.size());
  if (nearestNeighborDist2Vec != NULL)  // save also the distance (square!!!)
    (*nearestNeighborDist2Vec).resize(superVoxelA.size());
  for (vector<supervoxel>::iterator iter = superVoxelA.begin();
       iter != superVoxelA.end(); ++iter) {
    if (nearestNeighborDist2Vec != NULL)  // save also the distance (square!!!)
    {
      (*nearestNeighborDist2Vec)[count].reserve(KmaxNumNN);
      (*nearestNeighborDist2Vec)[count].clear();
    }
    nearestNeighborVec[count].reserve(KmaxNumNN);
    nearestNeighborVec[count].clear();
    offset = count;
    for (long long int ii = 0; ii < KmaxNumNN; ii++) {
      // cout<<offset<<" "<<dist[offset]<<" "<<ind[offset]<<"; Supervoxel:
      // "<<(*(vecIter[ ind[ offset] ]))<<endl;
      if (dist[offset] < auxMaxDist) {
        nearestNeighborVec[count].push_back(vecIter[ind[offset]]);
        if (nearestNeighborDist2Vec !=
            NULL)  // save also the distance (square!!!)
          (*nearestNeighborDist2Vec)[count].push_back(dist[offset]);
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

//===================================================================================================================================
// calculates nearest neighbors from supervoxelA (query points) into supervoxelB
// (reference points).
int supervoxel::nearestNeighbors(
    vector<supervoxel*>& superVoxelA, vector<supervoxel*>& superVoxelB,
    unsigned int KmaxNumNN, float KmaxDistKNN, int devCUDA,
    vector<vector<vector<supervoxel*>::iterator> >& nearestNeighborVec,
    vector<vector<float> >* nearestNeighborDist2Vec)  // you need to setup scale
                                                      // to calculate this
                                                      // property
{
  supervoxel::setKmaxDistKNN(KmaxDistKNN);
  supervoxel::setKmaxNumNN(KmaxNumNN);

  if (superVoxelA.empty() == true) {
    nearestNeighborVec.clear();
    return 0;
  }

  if (KmaxNumNN > maxKNN) {
    cout << "ERROR: at supervoxelNearestNeighborsInTimeForward: maximum number "
            "of NN "
         << KmaxNumNN << " is superior to maxKNN " << maxKNN
         << ". Please recompile knnCUDA.h code with a larger constant" << endl;
    return 2;
  }

  // preallocate memory for all the centroids from supervoxels
  int ref_nb = superVoxelB.size();
  long long int query_nb = superVoxelA.size();  // we find kNN for each query_nb
                                                // point with respect to ref_nb

  int* ind = new int[KmaxNumNN * query_nb];
  float* dist = new float[KmaxNumNN * query_nb];
  float* query_xyz = new float[dimsImage * query_nb];  // stores centroids
  float* ref_xyz = new float[dimsImage * ref_nb];      // stores centroids

  // copy centroids to temporary array
  long long int count = 0, offset;
  for (vector<supervoxel*>::iterator iter = superVoxelA.begin();
       iter != superVoxelA.end(); ++iter) {
    offset = count;
    for (int ii = 0; ii < dimsImage; ii++) {
      query_xyz[offset] = (*iter)->centroid[ii];
      offset += query_nb;
    }
    count++;
  }

  count = 0, offset;
  vector<vector<supervoxel*>::iterator> vecIter(
      ref_nb);  // we will need it later to assign ind to nearest neighbor
  for (vector<supervoxel*>::iterator iter = superVoxelB.begin();
       iter != superVoxelB.end(); ++iter) {
    offset = count;
    for (int ii = 0; ii < dimsImage; ii++) {
      ref_xyz[offset] = (*iter)->centroid[ii];
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
  nearestNeighborVec.resize(superVoxelA.size());
  if (nearestNeighborDist2Vec != NULL)  // save also the distance (square!!!)
    (*nearestNeighborDist2Vec).resize(superVoxelA.size());
  for (vector<supervoxel*>::iterator iter = superVoxelA.begin();
       iter != superVoxelA.end(); ++iter) {
    if (nearestNeighborDist2Vec != NULL)  // save also the distance (square!!!)
    {
      (*nearestNeighborDist2Vec)[count].reserve(KmaxNumNN);
      (*nearestNeighborDist2Vec)[count].clear();
    }
    nearestNeighborVec[count].reserve(KmaxNumNN);
    nearestNeighborVec[count].clear();
    offset = count;
    for (long long int ii = 0; ii < KmaxNumNN; ii++) {
      // cout<<offset<<" "<<dist[offset]<<" "<<ind[offset]<<"; Supervoxel:
      // "<<(*(vecIter[ ind[ offset] ]))<<endl;
      if (dist[offset] < auxMaxDist) {
        nearestNeighborVec[count].push_back(vecIter[ind[offset]]);
        if (nearestNeighborDist2Vec !=
            NULL)  // save also the distance (square!!!)
          (*nearestNeighborDist2Vec)[count].push_back(dist[offset]);
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

//===================================================================================================
template <class imgTypeC>
int supervoxel::weightedGaussianStatistics(double* m, double* W,
                                           float* intensity, bool regularizeW) {
  imgTypeC* imgPtr = (imgTypeC*)(dataPtr);
  float imgVal, N_k = 0.0f;
  int64 coordAux;
  int64 coord[dimsImage];
  int count;

  memset(m, 0, sizeof(double) * dimsImage);

  if (W == NULL)  // calculate only centroid
  {
    for (vector<uint64>::const_iterator iterP = PixelIdxList.begin();
         iterP != PixelIdxList.end(); ++iterP) {
      coordAux = (*iterP);
      imgVal = imgPtr[coordAux];
      for (int aa = 0; aa < dimsImage - 1; aa++) {
        coord[aa] = coordAux % (supervoxel::dataDims[aa]);
        coordAux -= coord[aa];
        coordAux /= (supervoxel::dataDims[aa]);
      }
      coord[dimsImage - 1] = coordAux;

      N_k += imgVal;
      for (int ii = 0; ii < dimsImage; ii++) {
        m[ii] += imgVal * coord[ii];
      }
    }
  } else  // calculate W as well
  {
    memset(W, 0, sizeof(double) * dimsImage * (1 + dimsImage) / 2);
    for (vector<uint64>::const_iterator iterP = PixelIdxList.begin();
         iterP != PixelIdxList.end(); ++iterP) {
      coordAux = (*iterP);
      imgVal = imgPtr[coordAux];
      for (int aa = 0; aa < dimsImage - 1; aa++) {
        coord[aa] = coordAux % (supervoxel::dataDims[aa]);
        coordAux -= coord[aa];
        coordAux /= (supervoxel::dataDims[aa]);
      }
      coord[dimsImage - 1] = coordAux;

      N_k += imgVal;
      count = 0;
      for (int ii = 0; ii < dimsImage; ii++) {
        m[ii] += imgVal * coord[ii];
        for (int jj = ii; jj < dimsImage; jj++) {
          W[count] += imgVal * coord[ii] * coord[jj];
          count++;
        }
      }
    }
  }
  // finish calculating sufficient statistics
  if (intensity != NULL) *intensity = N_k;

  if (N_k < 1e-5)  // special case
  {
    for (int ii = 0; ii < dimsImage; ii++) m[ii] = 0.0;
  } else {
    for (int ii = 0; ii < dimsImage; ii++) m[ii] /= N_k;
  }

  if (W != NULL) {
    count = 0;

    if (N_k < 1e-5)  // special case to avoid segmentation fault
    {
      for (int ii = 0; ii < dimsImage; ii++) {
        W[count] = 1.0;  // diagonal
        count++;
        for (int jj = ii + 1; jj < dimsImage; jj++) {
          W[count] = 0.0;
          count++;
        }
      }
    } else {
      for (int ii = 0; ii < dimsImage; ii++) {
        for (int jj = ii; jj < dimsImage; jj++) {
          W[count] = W[count] / N_k - m[ii] * m[jj];
          count++;
        }
      }
    }
    // inverse matrix
    double Waux[dimsImage * (1 + dimsImage) / 2];
    memcpy(Waux, W, sizeof(double) * dimsImage * (1 + dimsImage) / 2);

    if (Waux[5] < 1e-8)  // 2D case, to avoid NaN
    {
      Waux[5] = 0.5 * (Waux[0] + Waux[3]);
    }
    utilsAmatf_inverseSymmetricW_3D(Waux, W);

    // regularize matrix
    if (regularizeW == true)
      utilsAmatf_regularizePrecisionMatrix(W, scale, false);
  }

  return 0;
}

//===================================================================================================
template <class imgTypeC>
int supervoxel::weightedGaussianStatistics(bool regularizeW) {
  imgTypeC* imgPtr = (imgTypeC*)(dataPtr);
  float imgVal, N_k = 0.0f;
  int64 coordAux;
  int64 coord[dimsImage];
  int count;

  memset(centroid, 0, sizeof(float) * dimsImage);
  memset(precisionW, 0, sizeof(float) * dimsImage * (1 + dimsImage) / 2);
  for (vector<uint64>::const_iterator iterP = PixelIdxList.begin();
       iterP != PixelIdxList.end(); ++iterP) {
    coordAux = (*iterP);
    imgVal = imgPtr[coordAux];
    for (int aa = 0; aa < dimsImage - 1; aa++) {
      coord[aa] = coordAux % (supervoxel::dataDims[aa]);
      coordAux -= coord[aa];
      coordAux /= (supervoxel::dataDims[aa]);
    }
    coord[dimsImage - 1] = coordAux;

    N_k += imgVal;
    count = 0;
    for (int ii = 0; ii < dimsImage; ii++) {
      centroid[ii] += imgVal * coord[ii];
      for (int jj = ii; jj < dimsImage; jj++) {
        precisionW[count] += imgVal * coord[ii] * coord[jj];
        count++;
      }
    }
  }

  // finish calculating sufficient statistics
  intensity = N_k;

  if (N_k < 1e-5)  // special case
  {
    for (int ii = 0; ii < dimsImage; ii++) centroid[ii] = 0.0f;
  } else {
    for (int ii = 0; ii < dimsImage; ii++) centroid[ii] /= N_k;
  }

  if (precisionW != NULL) {
    count = 0;

    if (N_k < 1e-5)  // special case to avoid segmentation fault
    {
      for (int ii = 0; ii < dimsImage; ii++) {
        precisionW[count] = 1.0;  // diagonal
        count++;
        for (int jj = ii + 1; jj < dimsImage; jj++) {
          precisionW[count] = 0.0;
          count++;
        }
      }
    } else {
      for (int ii = 0; ii < dimsImage; ii++) {
        for (int jj = ii; jj < dimsImage; jj++) {
          precisionW[count] =
              precisionW[count] / N_k - centroid[ii] * centroid[jj];
          count++;
        }
      }
    }
    // inverse matrix
    float Waux[dimsImage * (1 + dimsImage) / 2];
    memcpy(Waux, precisionW, sizeof(float) * dimsImage * (1 + dimsImage) / 2);
    if (Waux[5] < 1e-6)  // 2D case, to avoid NaN
    {
      Waux[5] = 0.5 * (Waux[0] + Waux[3]);
    }
    utilsAmatf_inverseSymmetricW_3D(Waux, precisionW);

    // regularize matrix
    if (regularizeW == true)
      utilsAmatf_regularizePrecisionMatrix(precisionW, scale, false);
  }

  return 0;
}

//===================================================================================================
template <class imgTypeC>
int supervoxel::weightedCentroid() {
  imgTypeC* imgPtr = (imgTypeC*)(dataPtr);
  float imgVal, N_k = 0.0f;
  int64 coordAux;
  int64 coord[dimsImage];

  memset(centroid, 0, sizeof(float) * dimsImage);
  for (vector<uint64>::const_iterator iterP = PixelIdxList.begin();
       iterP != PixelIdxList.end(); ++iterP) {
    coordAux = (*iterP);
    imgVal = (float)(imgPtr[coordAux]);
    for (int aa = 0; aa < dimsImage - 1; aa++) {
      coord[aa] = coordAux % (supervoxel::dataDims[aa]);
      coordAux -= coord[aa];
      coordAux /= (supervoxel::dataDims[aa]);
    }
    coord[dimsImage - 1] = coordAux;

    N_k += imgVal;
    for (int ii = 0; ii < dimsImage; ii++) {
      centroid[ii] += imgVal * coord[ii];
    }
  }

  if (N_k < 1e-5)  // special case
  {
    for (int ii = 0; ii < dimsImage; ii++) centroid[ii] = 0.0f;
  } else {
    for (int ii = 0; ii < dimsImage; ii++) centroid[ii] /= N_k;
  }

  return 0;
}

//=====================================================================
template <class imgTypeC>
int supervoxel::robustGaussianStatistics(double* m, double* W, int maxSize,
                                         float maxPercentile) {
  cout << "ERROR: robustGaussianStatistics code not ready" << endl;

  imgTypeC* imgPtr = (imgTypeC*)(dataPtr);

  size_t thrPos = PixelIdxList.size();

  // 1.-calculate optimal threshold for voxels based on three criteria: maxSize,
  // maxPercentile and Otzu threshold

  // sort elements based on intensity
  cout << "ERROR:robustGaussianStatistics: not finished yet!!!" << endl;
  exit(3);

  // 2.-calculate Gaussian statistics with the reduced subset of PixelIdxList
  supervoxel sv(this->TM);
  sv.dataPtr = this->dataPtr;

  cout << "TODO: copy pixidx list;" << endl;

  float I;
  sv.weightedGaussianStatistics<imgTypeC>(
      m, W, &I, false);  // we should not need to regularize
}

//====================================================================
template <class imgTypeC>
imgTypeC supervoxel::intensityThreshold(int maxSize, float maxPercentile,
                                        unsigned int* maxPosPixelIdxList,
                                        bool localBackgroundSubtracted) {
  imgTypeC* imgPtr = (imgTypeC*)(dataPtr);
  unsigned int N = PixelIdxList.size();
  *maxPosPixelIdxList = 0;

  if (N == 0) return 0;

  vector<float> imVal(N);
  float maxVal = (float)(imgPtr[PixelIdxList[0]]);

  for (unsigned int ii = 0; ii < N; ii++) {
    imVal[ii] = (float)(imgPtr[PixelIdxList[ii]]);

    if (imVal[ii] > maxVal) {
      *maxPosPixelIdxList = ii;
      maxVal = imVal[ii];
    }
  }

  std::sort(imVal.begin(), imVal.end());  // ascending order

  // calculate threshold
  float thr = imVal[max(0, (int)(N)-maxSize)];  // size of the supervoxel

  if (localBackgroundSubtracted == false)  // we need other thresholds
  {
    float thr2 = imVal[(int)((1.0f - maxPercentile) * N)];  // percentile
    thr = max(thr, thr2);
    float thr3 = otsuThreshold(&(imVal[0]), N);
    thr = max(thr, thr3);
  }

  return (imgTypeC)(thr);
}

//===========================================================================
// inspired in code from:
// http://www.labbookpages.co.uk/software/imgProc/otsuThreshold.html
// Note: you can calculate what is called the between class variance, which is
// far quicker to calculate. Luckily, the threshold with the maximum between
// class variance also has the minimum within class variance. So it can also be
// used for finding the best threshold and therefore due to being simpler is a
// much better approach to use.
float supervoxel::otsuThreshold(float* arrayValues, int N) {
  int hist[256];
  // convert to [0,255] values

  float minVal = std::numeric_limits<float>::max();
  float maxVal = -minVal;

  for (int ii = 0; ii < N; ii++) {
    minVal = min(minVal, arrayValues[ii]);
    maxVal = max(maxVal, arrayValues[ii]);
  }

  // compute histogram
  memset(hist, 0, sizeof(int) * 256);
  float maxMin = 255.0f / (maxVal - minVal);

#if defined(_WIN32) || defined(_WIN64)
  if (_finite(maxMin) == false)  // degenerated cases
#else
  if (isfinite(maxMin) == 0)  // degenerated cases
#endif
    return -std::numeric_limits<float>::max();  // no threshold

  for (int ii = 0; ii < N; ii++) {
    hist[(int)((arrayValues[ii] - minVal) * maxMin)]++;
  }

  // compute between calss variance
  float sum = 0;
  for (int t = 0; t < 256; t++) sum += t * hist[t];

  float sumB = 0;
  int wB = 0;
  int wF = 0;

  float varMax = 0;
  int threshold = 0;

  float mB, mF, varBetween;
  for (int t = 0; t < 256; t++) {
    wB += hist[t];  // Weight Background
    if (wB == 0) continue;

    wF = N - wB;  // Weight Foreground
    if (wF == 0) break;

    sumB += (float)(t * hist[t]);

    mB = sumB / wB;          // Mean Background
    mF = (sum - sumB) / wF;  // Mean Foreground

    // Calculate Between Class Variance
    varBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);

    // Check if new maximum found
    if (varBetween > varMax) {
      varMax = varBetween;
      threshold = t;
    }
  }

  return (((float)(threshold) / maxMin) + minVal);
}

//====================================================================
template <class imgTypeC>
imgTypeC supervoxel::trimSupervoxel(int maxSize, float maxPercentile,
                                    int conn3D, int64* neighOffset) {
  imgTypeC* imgPtr = (imgTypeC*)(dataPtr);
  unsigned int maxPosPixelIdxList = 0;
  imgTypeC thr = 0;

  if (localBackgroundSubtracted == true) {
    removeZeros<imgTypeC>();  // remove all background elements
  }

  if (PixelIdxList.empty() == true)  // degenerate case
    return 0;

  // calculate threshold
  thr = intensityThreshold<imgTypeC>(
      maxSize, maxPercentile, &maxPosPixelIdxList, localBackgroundSubtracted);

  // stores if pixel has been selected or visited for conenctivity purposes
  unsigned char* visited = new unsigned char[PixelIdxList.size()];
  memset(visited, 0, sizeof(unsigned char) * PixelIdxList.size());

  queue<int> qPos;  // stores indexes in PixelIdxList that remain to be visited
  qPos.push(maxPosPixelIdxList);
  visited[maxPosPixelIdxList]++;  // it has already been added to be looked at

  int pos, posNeigh;
  vector<uint64>::iterator iterP, iterPos;
  int64 idxNeigh;
  int newSize = 0;

  while (qPos.empty() == false) {
    pos = qPos.front();
    qPos.pop();

    iterP = PixelIdxList.begin() + pos;

    if (imgPtr[*iterP] > thr)  // element selected
    {
      visited[pos]++;
      newSize++;
    } else
      continue;  // we are not interested in its neighbors

    // add neighbors within teh supervoxel that have not been visited
    for (int jj = 0; jj < conn3D; jj++) {
      idxNeigh = (*iterP) + neighOffset[jj];
      if (neighOffset[jj] > 0)
        iterPos = lower_bound(iterP, PixelIdxList.end(), idxNeigh);
      else
        iterPos = lower_bound(PixelIdxList.begin(), iterP, idxNeigh);

      if (iterPos != PixelIdxList.end() &&
          (*iterPos) == idxNeigh)  // element found
      {
        posNeigh = iterPos - PixelIdxList.begin();
        if (visited[posNeigh] == 0) {
          qPos.push(posNeigh);
          visited[posNeigh]++;
        }
      }
    }
  }

  // redo supervoxels
  vector<uint64> PixelIdxListAux(newSize);
  newSize = 0;
  for (size_t ii = 0; ii < PixelIdxList.size(); ii++) {
    if (visited[ii] > 1) {
      PixelIdxListAux[newSize++] =
          PixelIdxList[ii];  // we keep them alreayd sorted
    }
  }

  PixelIdxList = PixelIdxListAux;

  delete[] visited;

  return thr;
}

//====================================================================
template <class imgTypeC>
float supervoxel::mergePriorityFunction(supervoxel& svCh1, supervoxel& svCh2) {
  // int64 boundarySizeIsNeigh[dimsImage];
  // int conn3DIsNeigh = 74;
  // int64* neighOffsetIsNeigh =
  // supervoxel::buildNeighboorhoodConnectivity(conn3DIsNeigh + 1,
  // boundarySizeIsNeigh);//using the special neighborhood for coarse sampling

  float score = 0.5f;

  // simple test: are they touching?check if both supervoxels are actually
  // touching (otherwise we do not merge)
  if (svCh1.isNeighboring(svCh2, pMergeSplit.conn3DIsNeigh,
                          pMergeSplit.neighOffsetIsNeigh) == false) {
    // cout<<"mergePriority 0.0f because they are not neighbors"<<endl;
    return 0.0f;
  }

  // simple test: deltaZ
  if (getDeltaZ() > pMergeSplit.deltaZthr) {
    // cout<<"mergePriority 0.0f because deltaZ =
    // "<<getDeltaZ()<<">"<<pMergeSplit.deltaZthr<<endl;
    return 0.0f;
  }

  // cout<<"WARNING: supervoxel::mergePriorityFunction NOT DONE YET!!!!!"<<endl;
  // //TODO

  return score;
}

//============================================================
template <class imgTypeC>
void supervoxel::removeZeros() {
  imgTypeC* ptr = (imgTypeC*)(dataPtr);
  int pos = 0;
  for (int ii = 0; ii < PixelIdxList.size(); ii++) {
    if (ptr[PixelIdxList[ii]] > 0) {
      // PixelIdxList.erase( PixelIdxList.begin() + ii );//I don't do swap and
      // resize to preserve sorting. BUT this is ridiculously slow

      PixelIdxList[pos++] = PixelIdxList[ii];
    }
  }

  PixelIdxList.resize(pos);
}

//=============================================================
float supervoxel::Mahalanobis2Distance(const supervoxel& s) const {
  float aux[dimsImage];
  for (int ii = 0; ii < dimsImage; ii++) {
    aux[ii] = (centroid[ii] - s.centroid[ii]);  // I do not need to scale
                                                // anything because W already
                                                // scales
  }
  return utilsAmatf_MahalanobisDistance_3D<float>(aux, precisionW);
}

//=========================================================
// template declaration
template float supervoxel::trimSupervoxel<float>(int maxSize,
                                                 float maxPercentile,
                                                 int conn3D,
                                                 int64* neighOffset);
template unsigned short int supervoxel::trimSupervoxel<unsigned short int>(
    int maxSize, float maxPercentile, int conn3D, int64* neighOffset);
template unsigned char supervoxel::trimSupervoxel<unsigned char>(
    int maxSize, float maxPercentile, int conn3D, int64* neighOffset);

template float supervoxel::trimSupervoxel<float>();
template unsigned short int supervoxel::trimSupervoxel<unsigned short int>();
template unsigned char supervoxel::trimSupervoxel<unsigned char>();

template void supervoxel::removeZeros<float>();
template void supervoxel::removeZeros<unsigned short int>();
template void supervoxel::removeZeros<unsigned char>();

template float supervoxel::intensityThreshold<float>(
    int maxSize, float maxPercentile, unsigned int* maxPosPixelIdxList,
    bool localBackgroundSubtracted);
template unsigned short int supervoxel::intensityThreshold<unsigned short int>(
    int maxSize, float maxPercentile, unsigned int* maxPosPixelIdxList,
    bool localBackgroundSubtracted);
template unsigned char supervoxel::intensityThreshold<unsigned char>(
    int maxSize, float maxPercentile, unsigned int* maxPosPixelIdxList,
    bool localBackgroundSubtracted);

template int supervoxel::weightedGaussianStatistics<float>(double* m, double* W,
                                                           float* intensity,
                                                           bool regularizeW);
template int supervoxel::weightedGaussianStatistics<unsigned short int>(
    double* m, double* W, float* intensity, bool regularizeW);
template int supervoxel::weightedGaussianStatistics<unsigned char>(
    double* m, double* W, float* intensity, bool regularizeW);

template int supervoxel::weightedGaussianStatistics<float>(bool regularizeW);
template int supervoxel::weightedGaussianStatistics<unsigned short int>(
    bool regularizeW);
template int supervoxel::weightedGaussianStatistics<unsigned char>(
    bool regularizeW);

template int supervoxel::weightedCentroid<float>();
template int supervoxel::weightedCentroid<unsigned short int>();
template int supervoxel::weightedCentroid<unsigned char>();

template int supervoxel::robustGaussianStatistics<float>(double* m, double* W,
                                                         int maxSize,
                                                         float maxPercentile);
template int supervoxel::robustGaussianStatistics<unsigned short int>(
    double* m, double* W, int maxSize, float maxPercentile);
template int supervoxel::robustGaussianStatistics<unsigned char>(
    double* m, double* W, int maxSize, float maxPercentile);

template float supervoxel::mergePriorityFunction<float>(supervoxel& svCh1,
                                                        supervoxel& svCh2);
template float supervoxel::mergePriorityFunction<unsigned short int>(
    supervoxel& svCh1, supervoxel& svCh2);
template float supervoxel::mergePriorityFunction<unsigned char>(
    supervoxel& svCh1, supervoxel& svCh2);
