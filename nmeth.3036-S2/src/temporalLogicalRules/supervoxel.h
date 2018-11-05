/*
 * See license.txt for full license and copyright notice.
 *
 * \brief Stores supervoxel from image
 *
 */

#ifndef __SUPERVOXEL_LINEAGE_H__
#define __SUPERVOXEL_LINEAGE_H__

#include <stdio.h>
#include <string.h>
#include <cstring>
#include <iostream>
#include <list>
#include <vector>
#include "../constants.h"
#include "edgeTree.h"
#include "localGeometricDescriptor.h"

#if defined(_MSC_VER) || defined(__BORLANDC__)
typedef unsigned __int64 uint64;
typedef signed __int64 int64;
#else
typedef unsigned long long uint64;
typedef signed long long int64;
#endif

using namespace std;

//=====================================================
// parameters to decide on merge/split
struct paramMergeSplit {
  int conn3DIsNeigh;
  int64* neighOffsetIsNeigh;
  uint64 deltaZthr;

  paramMergeSplit() { neighOffsetIsNeigh = NULL; }
  ~paramMergeSplit() {
    if (neighOffsetIsNeigh != NULL) {
      delete[] neighOffsetIsNeigh;
      neighOffsetIsNeigh = NULL;
    }
  }

  void setParam(int conn3DIsNeigh_, int64* neighOffsetIsNeigh_,
                uint64 deltaZthr_) {
    conn3DIsNeigh = conn3DIsNeigh_;
    if (neighOffsetIsNeigh != NULL) delete[] neighOffsetIsNeigh;

    neighOffsetIsNeigh = new int64[conn3DIsNeigh];
    memcpy(neighOffsetIsNeigh, neighOffsetIsNeigh_,
           sizeof(int64) * conn3DIsNeigh);
    deltaZthr = deltaZthr_;
  }
};

// typedef declarations to build hierarchical tree
class nucleus;                        // forward declaration
class supervoxel;                     /// forward declaration
struct nodeHierarchicalSegmentation;  // forward declaration
template <class ItemType>
struct TreeNode;  // forward declaration

typedef list<nucleus>::iterator ParentTypeSupervoxel;
typedef list<supervoxel>::iterator SibilingTypeSupervoxel;
typedef char ChildrenTypeSupervoxel;  // bogus definition

// simple class to store supervoxels
class supervoxel {
 public:
  vector<uint64> PixelIdxList;  // ordered indexes of the pixels belonging to
                                // teh supervoxel
  void* dataPtr;  // pointer to the image data: super voxel does not take care
                  // of freeing this pointer
  static uint64 dataDims[dimsImage];  // size of the data in each dimension in
                                      // case we need to convert linear index to
                                      // x,y,z
  static uint64 dataSizeInBytes;  // so we can deallocated dataPtr and we can
                                  // calculate the size of the data

  // sufficient statistics
  float centroid[dimsImage];
  float precisionW[dimsImage * (1 + dimsImage) / 2];  // inverse of covariance
                                                      // metric of the
                                                      // equivalent ellipsoid
  float intensity;  // total intensity over PixelIdxList

  localGeometricDescriptor<dimsImage>
      gDescriptor;  // stores a geometric descriptor

  int TM;  // time point that the supervxoel belongs to

  float tempWildcard;  // temporary storage space for different purposes. IT IS
                       // NOT COPIED WITH THE ASSIGNMENT OPERATOR OR THE COPY
                       // CONSTRUCTOR!!!

  float probClassifier;  // stores probability (or score) of some sort of
                         // classifier (split, background, etc) to be used later

  bool localBackgroundSubtracted;  // true->we have subtracted local bacground
                                   // to zero->trimming is also removing zeros;
                                   // false->otzu threshold; (default is false)

  edgeTree<ParentTypeSupervoxel, ChildrenTypeSupervoxel> treeNode;
  vector<SibilingTypeSupervoxel> nearestNeighborsInSpace;
  vector<SibilingTypeSupervoxel>
      nearestNeighborsInTimeForward;  // all these are based on centroids
  vector<SibilingTypeSupervoxel> nearestNeighborsInTimeBackward;

  // parameters for merge / split
  static paramMergeSplit pMergeSplit;

  TreeNode<nodeHierarchicalSegmentation>*
      nodeHSptr;  // pointer to a node in the hierarchical segmentation that
                  // generated this supervoxel. TODO: use this in
                  // hierarchicalSegmentation class to not use
                  // currentSegmentationNode variable vector anymore

  // constructor / destructors
  supervoxel();
  supervoxel(int TM_);
  supervoxel(const supervoxel& p);
  ~supervoxel();

  supervoxel(istream& is);          // create supervoxel from binary file
  void writeToBinary(ostream& os);  // write ot binary file

  // operators
  supervoxel& operator=(const supervoxel& p);
  friend ostream& operator<<(ostream& os, supervoxel m);

  // calculates the number of pixels that two superpixel overlap. For Jaccard
  // distance this is faster than keeping the specific pixelIdx of teh
  // intersection
  // VIP: we assume PixelIdxList are sorted
  uint64 intersectionSize(const supervoxel& s) const;
  float intersectionCost(const supervoxel& s) const;  // exactly the same as
                                                      // above. It just returns
                                                      // a float so it has the
                                                      // same structure as other
                                                      // cost/distance methods
  float intersectionCostRelativeToCandidate(const supervoxel& s)
      const;  // exactly the same as above but it is divided by the number of
              // voxels in this->PixelIdxList.size() to make it relative. Thus,
              // maximum value returned is 1.0
  float JaccardDistance(const supervoxel& s) const;
  float JaccardIndex(const supervoxel& s) const;      // 1.0-jaccardDistance
  float JaccardIndexWithOffset(const supervoxel& s);  // 1.0-jaccardDistance
                                                      // after subtracting the
                                                      // centroid difference
                                                      // between supervoxels. It
                                                      // is a shape measure
  float Euclidean2Distance(
      const supervoxel& s) const;  // Euclidean^2 distance (to avoid sqrt)
  float Euclidean2Distance(
      const float p[dimsImage]) const;  // Euclidean^2 distance (to avoid sqrt)
  float Mahalanobis2Distance(const supervoxel& s) const;  // Distance using W
                                                          // (ellipsoid) from
                                                          // *this. You need to
                                                          // make sure that
                                                          // centroid and W have
                                                          // been calculated
  int neighboringVoxels(const supervoxel& s, int conn3D, int64* neighOffset,
                        vector<uint64>& PixelIdxListBorderThis,
                        vector<uint64>& PixelIdxListBorderS);  // returns a list
                                                               // of neighboring
                                                               // voxels beween
                                                               // two
                                                               // supervoxels
  int neighboringVoxels(const supervoxel& s, int conn3D, int64* neighOffset,
                        int satNeighboringVoxels);  // returns the number of
                                                    // neighboring supervoxels
                                                    // (but not its
                                                    // position)->faster.
                                                    // satNeighboringVoxels is a
                                                    // threshold: if neighboring
                                                    // supervoxels reaches that
                                                    // number->exit ( set to Inf
                                                    // if you don't need it )
  bool isNeighboring(const supervoxel& s, int conn3D,
                     int64* neighOffset);  // check only if they are neighbors.
                                           // Faster version of
                                           // neighboringVoxels
  static int64* buildNeighboorhoodConnectivity(int conn3D, int64* boundarySize);
  float neighboringVoxelsSelf(int conn3D,
                              int64* neighOffset);  // average number of
                                                    // neighboring voxels that
                                                    // belong to the this
                                                    // supervoxel (it is some
                                                    // sort of dual of the
                                                    // number of "empty" hoels
                                                    // within a supervoxel)
  int numHoles();  // number of empty supervoxels inside supervoxel
                   // (approximate. There are some singular cases where approach
                   // fails)

  bool idxBelongsToPixelIdxListLinearSearch(
      uint64 idx, size_t iniPosSearch = 0);  // returns if idx belongs to
                                             // pixelIdxList or not. If you have
                                             // a clue of the position in the
                                             // list (for example, looking at
                                             // neighboring elements), search
                                             // would be faster. We use a simple
                                             // linear search since the list
                                             // tends to be small.
  void mergeSupervoxels(vector<supervoxel*>& src);  // adds components from src
                                                    // into *this. HOWEVER, the
                                                    // nearest neighbors would
                                                    // have to be computed again
                                                    // as well as the centroid.
  // calculates nearest neighbors from supervoxelA (query points) into
  // supervoxelB (reference points). It stores results in nearestNeighborVec.
  // nearestNeighborVec.size() = superVoxelA.size() and contains iterators to
  // superVoxelB
  static int nearestNeighbors(
      list<supervoxel>& superVoxelA, list<supervoxel>& superVoxelB,
      unsigned int KmaxNumNN, float KmaxDistKNN, int devCUDA,
      vector<vector<SibilingTypeSupervoxel> >& nearestNeighborVec,
      vector<vector<float> >* nearestNeighborDist2Vec = NULL);
  static int nearestNeighbors(
      vector<supervoxel>& superVoxelA, vector<supervoxel>& superVoxelB,
      unsigned int KmaxNumNN, float KmaxDistKNN, int devCUDA,
      vector<vector<vector<supervoxel>::iterator> >& nearestNeighborVec,
      vector<vector<float> >* nearestNeighborDist2Vec = NULL);
  static int nearestNeighbors(
      vector<supervoxel*>& superVoxelA, vector<supervoxel*>& superVoxelB,
      unsigned int KmaxNumNN, float KmaxDistKNN, int devCUDA,
      vector<vector<vector<supervoxel*>::iterator> >& nearestNeighborVec,
      vector<vector<float> >* nearestNeighborDist2Vec = NULL);
  static int nearestNeighbors(
      list<supervoxel>& superVoxelA, vector<supervoxel>& superVoxelB,
      unsigned int KmaxNumNN, float KmaxDistKNN, int devCUDA,
      vector<vector<vector<supervoxel>::iterator> >& nearestNeighborVec,
      vector<vector<float> >* nearestNeighborDist2Vec = NULL);
  static int nearestNeighbors(
      vector<float> superVoxelA[dimsImage],
      vector<float> superVoxelB[dimsImage], unsigned int KmaxNumNN,
      float KmaxDistKNN, int devCUDA, vector<vector<int> >& nearestNeighborVec,
      vector<vector<float> >* nearestNeighborDist2Vec = NULL);
  static int nearestNeighbors(
      vector<supervoxel*>& superVoxelA, vector<float> superVoxelB[dimsImage],
      unsigned int KmaxNumNN, float KmaxDistKNN, int devCUDA,
      vector<vector<int> >& nearestNeighborVec,
      vector<vector<float> >* nearestNeighborDist2Vec = NULL);

  // simple set/get operations
  static void setDataDims(uint64 p[dimsImage]) {
    memcpy(dataDims, p, sizeof(uint64) * dimsImage);
  };
  static uint64 getDataSize();
  static void setScale(float p[dimsImage]) {
    memcpy(scale, p, sizeof(float) * dimsImage);
  };
  static float* getScale() { return scale; };
  static void getScale(float sc[3]) {
    for (int ii = 0; ii < dimsImage; ii++) sc[ii] = scale[ii];
  };
  static float getKmaxDistKNN() { return KmaxDistKNN; };
  static unsigned int getKmaxNumNN() { return KmaxNumNN; };
  static unsigned int getmaxKNNCUDA();
  static void setKmaxDistKNN(float p) { KmaxDistKNN = p; };
  static void setKmaxNumNN(unsigned int p) { KmaxNumNN = p; };
  static int getDataType();  // return an int that corresponds to enum in mylib
                             // for XXX_TYPE
  static void getCoordinates(
      uint64 p,
      uint64 coord[dimsImage]);  // parse PixelIdx to x,y,z coordinates

  static void setTrimParameters(int maxSize, float maxPercentile, int conn3D);
  static void freeTrimParameters();

  static int getMaxKnnCUDA();  // maximum number of KNN allowed

  uint64 getDeltaZ();  // calculates the difference between top z coord and
                       // bottom z coord

  template <class imgTypeC>
  int weightedGaussianStatistics(double* m, double* W, float* intensity,
                                 bool regularizeW);  // calculate mean (m) and
                                                     // precision matrix( W )
                                                     // from supervoxel (so we
                                                     // have an equivalent
                                                     // Gaussian). double
                                                     // W[dimsImage *
                                                     // (1+dimsImage) /2]
                                                     // (symmetric matrix). We
                                                     // weight based on the
                                                     // intensity
  template <class imgTypeC>
  int weightedGaussianStatistics(bool regularizeW);  // calculate mean (m) and
                                                     // precision matrix( W )
                                                     // from supervoxel (so we
                                                     // have an equivalent
                                                     // Gaussian). double
                                                     // W[dimsImage *
                                                     // (1+dimsImage) /2]
                                                     // (symmetric matrix). We
                                                     // weight based on the
                                                     // intensity
  template <class imgTypeC>
  int weightedCentroid();  // only the centroid (in case we do not need the
                           // other stats)
  template <class imgTypeC>
  int robustGaussianStatistics(double* m, double* W, int maxSize,
                               float maxPercentile);  // calculate mean (m) and
                                                      // precision matrix( W )
                                                      // from supervoxel (so we
                                                      // have an equivalent
                                                      // Gaussian). double
                                                      // W[dimsImage *
                                                      // (1+dimsImage) /2]
                                                      // (symmetric matrix). We
                                                      // "prefilter" elements
                                                      // before calculating
                                                      // Gaussian statistics

  template <class imgTypeC>
  imgTypeC intensityThreshold(
      int maxSize, float maxPercentile, unsigned int* maxPosPixelIdxList,
      bool localBackgroundSubtracted);  // calculates intensity threshold based
                                        // on 3 criteria: maximum size,
                                        // maxPrecentile and otzu's threshold.
                                        // Returns also the position of the
                                        // maximum in PixelIdxList
  template <class imgTypeC>
  imgTypeC trimSupervoxel(int maxSize, float maxPercentile, int conn3D,
                          int64* neighOffset);  // reduces the number of
                                                // PixelIdxList based on
                                                // intensityThreshold() and
                                                // connectivity starting from
                                                // the maximum (an attempt to
                                                // segment nucleus after image
                                                // partition with supervoxels)

  template <class imgTypeC>
  imgTypeC trimSupervoxel() {
    return trimSupervoxel<imgTypeC>(maxSizeTrim, maxPrecentileTrim, conn3Dtrim,
                                    neighOffsetTrim);
  };  // it uses static values within teh class supervoxel

  template <class imgTypeC>
  void removeZeros();  // if we have already done local backgroudn subtraction,
                       // we just need to subtract zeros instead of trimming

  /*
  \brief returns the probability that this and sv should be merged
  \warning This is the proposed supervoxel that comes from teh merge of svCh1
  and svCH2

  \warning it used the pMergeSplit values to make decisions

  \param in	svCh1	children 1 to be merged with svCh2
  \param in	svCh2	children 2 to be merged with svCh1
  */
  template <class imgTypeC>
  float mergePriorityFunction(supervoxel& svCh1, supervoxel& svCh2);

  // other functions
  static float otsuThreshold(
      float* arrayValues,
      int N);  // calculate otzu's threshold for an array of length N

  // debugging functions
  int debugPixelIdxListIsSorted();

 protected:
 private:
  static float scale[dimsImage];
  static unsigned int KmaxNumNN;  // maximum nearest neighbors
  static float KmaxDistKNN;  // maximum distance to select nearest neighbors

  // parameters used for trimming supervoxels
  static int maxSizeTrim;
  static float maxPrecentileTrim;
  static int conn3Dtrim;
  static int64* neighOffsetTrim;
  static int64 boundarySizeTrim[dimsImage];
};

//===================================================
// we assume PixelIdxList is sorted
inline uint64 supervoxel::intersectionSize(const supervoxel& s) const {
  uint64 count = 0;

  vector<uint64>::const_iterator iter = PixelIdxList.begin();
  vector<uint64>::const_iterator iterS = s.PixelIdxList.begin();

  while ((iter != PixelIdxList.end()) && (iterS != s.PixelIdxList.end())) {
    if ((*iter) == (*iterS)) {
      count++;
      iter++;
      iterS++;
    } else if ((*iter) < (*iterS)) {
      iter++;
    } else {  //(*iter)>(*iterS)
      iterS++;
    }
  }

  return count;
}

inline float supervoxel::intersectionCost(const supervoxel& s) const {
  uint64 count = 0;

  vector<uint64>::const_iterator iter = PixelIdxList.begin();
  vector<uint64>::const_iterator iterS = s.PixelIdxList.begin();

  while ((iter != PixelIdxList.end()) && (iterS != s.PixelIdxList.end())) {
    if ((*iter) == (*iterS)) {
      count++;
      iter++;
      iterS++;
    } else if ((*iter) < (*iterS)) {
      iter++;
    } else {  //(*iter)>(*iterS)
      iterS++;
    }
  }

  return ((float)count);
}

inline float supervoxel::intersectionCostRelativeToCandidate(
    const supervoxel& s) const {
  uint64 count = 0;

  vector<uint64>::const_iterator iter = PixelIdxList.begin();
  vector<uint64>::const_iterator iterS = s.PixelIdxList.begin();

  while ((iter != PixelIdxList.end()) && (iterS != s.PixelIdxList.end())) {
    if ((*iter) == (*iterS)) {
      count++;
      iter++;
      iterS++;
    } else if ((*iter) < (*iterS)) {
      iter++;
    } else {  //(*iter)>(*iterS)
      iterS++;
    }
  }

  return ((float)count) / ((float)PixelIdxList.size());
}

inline float supervoxel::JaccardDistance(const supervoxel& s) const {
  float aux = (float)(intersectionSize(s));
  return 1.0f - (aux / ((float(PixelIdxList.size())) +
                        (float(s.PixelIdxList.size())) - aux));
}

inline float supervoxel::JaccardIndex(const supervoxel& s) const {
  float aux = (float)(intersectionSize(s));
  return (aux / ((float(PixelIdxList.size())) + (float(s.PixelIdxList.size())) -
                 aux));
}

inline float supervoxel::JaccardIndexWithOffset(const supervoxel& s) {
  if (PixelIdxList.empty() == true) return 0;

  // calculate offset
  int64 offset = 0;
  int64 offsetVol = 1;
  for (int ii = 0; ii < dimsImage; ii++) {
    offset += int64(centroid[ii] - s.centroid[ii]) * offsetVol;
    offsetVol *= dataDims[ii];
  }

  // subtract offset from supervoxel PixelIdxList
  if (int64(PixelIdxList[0]) - offset < 0) return 0;

  for (size_t ii = 0; ii < PixelIdxList.size(); ii++)
    PixelIdxList[ii] -= offset;

  // calculate Jaccard index
  float aux = (float)(intersectionSize(s));
  aux = (aux /
         ((float(PixelIdxList.size())) + (float(s.PixelIdxList.size())) - aux));

  // unsubtract offset
  for (size_t ii = 0; ii < PixelIdxList.size(); ii++)
    PixelIdxList[ii] += offset;

  return aux;
}

inline float supervoxel::Euclidean2Distance(const supervoxel& s) const {
  float dist = 0;
  float aux = 0;
  for (int ii = 0; ii < dimsImage; ii++) {
    aux = (centroid[ii] - s.centroid[ii]) * scale[ii];
    dist += aux * aux;
  }
  return dist;
}

inline float supervoxel::Euclidean2Distance(const float p[dimsImage]) const {
  float dist = 0;
  float aux = 0;
  for (int ii = 0; ii < dimsImage; ii++) {
    aux = (centroid[ii] - p[ii]) * scale[ii];
    dist += aux * aux;
  }
  return dist;
}

//===================================================
inline int supervoxel::getDataType() {
  uint64 dataSize = 1;
  for (int ii = 0; ii < dimsImage; ii++) dataSize *= dataDims[ii];

  dataSize = dataSizeInBytes / dataSize;

  switch (dataSize)  // we cannot cover all the cases
  {
    case 1:
      return 0;
      break;
    case 2:
      return 1;
      break;
    case 4:
      return 8;
      break;
    case 8:
      return 9;
      break;
    default:
      return -1;
  }

  /*mylib.h
  UINT8_TYPE   = 0,
UINT16_TYPE  = 1,
UINT32_TYPE  = 2,
UINT64_TYPE  = 3,
INT8_TYPE    = 4,
INT16_TYPE   = 5,
INT32_TYPE   = 6,
INT64_TYPE   = 7,
FLOAT32_TYPE = 8,
FLOAT64_TYPE = 9
  */
}
//======================================
inline void supervoxel::getCoordinates(
    uint64 p, uint64 coord[dimsImage])  // parse PixelIdx to x,y,z coordinates
{
  for (int ii = 0; ii < dimsImage; ii++) {
    coord[ii] = p % dataDims[ii];
    p -= coord[ii];
    p /= dataDims[ii];
  }
}
//====================================================
inline void supervoxel::setTrimParameters(int maxSize, float maxPercentile,
                                          int conn3D) {
  maxSizeTrim = maxSize;
  maxPrecentileTrim = maxPercentile;
  conn3Dtrim = conn3D;
  neighOffsetTrim =
      buildNeighboorhoodConnectivity(conn3Dtrim, boundarySizeTrim);
}

inline void supervoxel::freeTrimParameters() { delete[] neighOffsetTrim; }

//=========================================================
inline uint64 supervoxel::getDeltaZ() {
  if (PixelIdxList.empty() == true) return 0;

  float alpha = 0.05;  // percentile to make it robust estimation
  uint64 zTop, zBottom;
  uint64 coord[dimsImage];

  int sizeSv = PixelIdxList.size();

  uint64 pos = PixelIdxList[(int)(alpha * sizeSv)];
  supervoxel::getCoordinates(pos, coord);
  zBottom = coord[dimsImage - 1];

  pos = PixelIdxList[(int)((1.0f - alpha) * sizeSv)];
  supervoxel::getCoordinates(pos, coord);
  zTop = coord[dimsImage - 1];

  return (zTop - zBottom);
}

#endif
