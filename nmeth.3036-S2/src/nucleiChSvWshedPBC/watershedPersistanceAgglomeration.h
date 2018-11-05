/*
 *
 * \brief Implements main functionalities to segment nuclei channel using
 * watershed + agglomeration based on persistance methods
 *
 */

#ifndef __WATERSHED_PERSISTANCE_AGGLOMERATION_H__
#define __WATERSHED_PERSISTANCE_AGGLOMERATION_H__

#include <vector>
#include "constants.h"
#include "hierarchicalSegmentation.h"
#include "set_union.h"

struct ForegroundVoxel;  // forward declaration

int64* buildNeighboorhoodConnectivity(int conn3D, const int64* imgDims,
                                      int64* boundarySize);
void selectForegroundElements(const imgVoxelType* img, const int64* imgDims,
                              imgVoxelType backgroundThr, int conn3D,
                              std::vector<ForegroundVoxel>& foregroundVec,
                              imgLabelType* imgL,
                              int64* numForegroundPerElementsSlice = NULL);

int watershedPersistanceAgglomeration(
    const imgVoxelType* img, const int64* imgDims, imgVoxelType backgroundThr,
    int conn3D, imgVoxelType tau, imgLabelType* imgL, imgLabelType* numLabels,
    set_unionC* L = NULL);  // L only needed for multithreaded watershed

int watershedPersistanceAgglomerationMultithread(
    const imgVoxelType* img, const int64* imgDims, imgVoxelType backgroundThr,
    int conn3D, imgVoxelType tau, imgLabelType* imgL, imgLabelType* numLabels,
    int numThreads = -1);  // if numtrheads is not specified it selects all the
                           // the cores availables
// int threadWatershedCallback(const imgVoxelType* img, const int64*
// imgDimsThread, uint64 offset,imgVoxelType backgroundThr, int conn3D,
// imgVoxelType tau, set_unionC* L, imgLabelType *imgL);//watershed executed per
// blocks on each thread
void* threadWatershedCallback(void* p);  // adapted for pthreads now
int checkRegionMerge(
    const imgVoxelType* img, const std::vector<ForegroundVoxel>& foregroundVec,
    set_unionC* L, const int64* imgDims, int conn3D, imgVoxelType tau,
    imgLabelType* imgL, imgLabelType* numLabels,
    std::vector<labelType>&
        mergePairs);  // routine to check which regions need to be merged

// it can be useful to detect background super-voxels after local background
// subtraction
int averageNumberOfNeighborsPerRegion(const imgLabelType* imgL,
                                      const int64* imgDims, int conn3D,
                                      std::vector<float>& avgNumberOfNeighbors);

// functions to build a hierarchical segmentation
/*
\brief builds hierarchical segmentation dendrogram

\return NULL if there was an error. A newly allocated object if succesfull.

\param in minTau  minimum tau to build the basic regions (bottom of the
dendrogram). Any two regions below that tau will be merged and will not be
considered in the hierarchy

\param in TODO

*/
hierarchicalSegmentation* buildHierarchicalSegmentation(
    imgVoxelType* img, const int64* imgDims, imgVoxelType backgroundThr,
    int conn3D, imgVoxelType minTau, int numThreads);

#endif  //__WATERSHED_PERSISTANCE_AGGLOMERATION_H__
