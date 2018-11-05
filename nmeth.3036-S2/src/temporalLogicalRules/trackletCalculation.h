/*
 * See license.txt for full license and copyright notice.
 *
 *  trackletCalculation.h
 *
 * \brief Methods to calculate tracklets from a supervoxel decomposition
 *
 */

#ifndef __TRACKLET_CALCULATION_LINEAGE_H__
#define __TRACKLET_CALCULATION_LINEAGE_H__

#include "lineageHyperTree.h"  //each tracklet is represented by a lineageHyperTree
#include "sparseHungarianAlgorithm/sparseHungarianAlgorithm.h"

/*
\brief calculate tracklets using Hungarian algorithm between super-voxels in two
time points T0 and T1

\param in svList0 list of input super-voxels. We use
supervoxel.nearestNeighborsInTimeForward to calculate candidates and
correspondence
\param in distanceMethod. 0->Jaccard distance between supervoxels; 1->Scaled
Euclidean distance between centroids
\param in thrCost cost assigned to an edge that goes to the garbage potential

\param out assignmentId resized to length svListT0.size() to hold solution.
assignmentId[jj] holds an iterator pointing to the candidate supervoxel within a
list<supervoxel>.
\param in If no assignment has been made (garbage potential wins), then
assignmentId[jj] = nullAssignment;

*/
int calculateTrackletsWithSparseHungarianAlgorithm(
    list<supervoxel> &svListT0, int distanceMethod, double thrCost,
    vector<SibilingTypeSupervoxel> &assignmentId,
    SibilingTypeSupervoxel *nullAssignment);

/*
\brief given a lineage, calculates tracklets based on
calculateTrackletsWithSparseHungarianAlgorithm. lht needs to have temporal
neighbors calculated for each supervxoel before calling this function

\warning All elements in the hypergraph are modified except supervoxels

\param in out lht Contains the supervoxels and it will contain the solution at
the end. a Tracklet solution is still a set of lineages
*/
int calculateTrackletsWithSparseHungarianAlgorithm(lineageHyperTree &lht,
                                                   int distanceMethod,
                                                   double thrCost,
                                                   unsigned int numThreads);

/*
\brief TODO


calculateSupervoxelMatchingOneToManyWithSparseHungarianAlgorithm(superSuperVoxelList,
lht.supervoxelsList[frame], costMethod, thrCost, assignmentId);
*/
int calculateSupervoxelMatchingOneToManyWithSparseHungarianAlgorithm(
    list<supervoxel> &superVoxelA, list<supervoxel> &superVoxelB,
    vector<vector<SibilingTypeSupervoxel> > &nearestNeighborVec, int costMethod,
    double thrCost, assignmentOneToMany *assignmentId);

/*
\brief Tries to entend elements that have not been assigned yet. THEREFORE,
assignmentId IS NOT INITIALIZED. You have to run
calculateSupervoxelMatchingOneToManyWithSparseHungarianAlgorithm first

*/
int extendMatchingWithClearOneToOneAssignments(
    list<supervoxel> &superVoxelA, list<supervoxel> &superVoxelB,
    vector<vector<SibilingTypeSupervoxel> > &nearestNeighborVecAtoB,
    vector<vector<SibilingTypeSupervoxel> > &nearestNeighborVecBtoA,
    assignmentOneToMany *assignmentId);
int extendMatchingWithClearOneToOneAssignmentsEuclidean(
    list<supervoxel> &superVoxelA, list<supervoxel> &superVoxelB,
    vector<vector<SibilingTypeSupervoxel> > &nearestNeighborVecAtoB,
    vector<vector<SibilingTypeSupervoxel> > &nearestNeighborVecBtoA,
    assignmentOneToMany *assignmentId);

#endif  // __TRACKLET_CALCULATION_LINEAGE_H__;
