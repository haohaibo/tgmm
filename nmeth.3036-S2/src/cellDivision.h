/*
 *
 * \brief Contains routines dealing with cell division for lineage hypertree
 *
 */
#ifndef __CELL_DVISION_ROUTINES_H__
#define __CELL_DVISION_ROUTINES_H__

#include "UtilsCUDA/3DEllipticalHaarFeatures/EllipticalHaarFeatures.h"
#include "lineageHyperTree.h"

/*
\brief very simple routine to check for cell division. For all the nuclei in
time TM that have more than 1 supervoxel, it checks if it can partition the
group of supervoxels into two with max flow / min-cut (i.e, in two sets where
supervoxels are not touching)

*/
int cellDivisionMinFlowTouchingSupervoxels(lineageHyperTree& lht, int TM,
                                           int conn3D,
                                           size_t minNeighboringVoxels,
                                           int& numCellDivisions);
int cellDivisionMinFlowTouchingSupervoxels(
    lineageHyperTree& lht, int TM, int conn3D, size_t minNeighboringVoxels,
    int& numCellDivisions,
    int& numBirths);  // considers the possibility of split in more than 2 ways

/*
\brief calculate classifier score for background only in short lineages
*/
int backgroundCheckWithClassifierShortLineages(lineageHyperTree& lht, int TM,
                                               int maxLengthBackgroundCheck,
                                               const imageType* imgPtr,
                                               const long long int* imgDims);

/*
\brief To decide best cell division pair
*/
float MahalanobisDistanceMotherAlongDaughtersAxis(float centroidPar[dimsImage],
                                                  float centroidCh1[dimsImage],
                                                  float centroidCh2[dimsImage],
                                                  float scale[dimsImage]);

/*
\brief alculate distance of motehr cell to cell division plane between daughters
        */
float DistanceCellDivisionPlane(float centroidPar[dimsImage],
                                float centroidCh1[dimsImage],
                                float centroidCh2[dimsImage],
                                float scale[dimsImage]);

//------------------------debug-------------------------
// test KullbackLeibler method for ISBI2013 paper in comparison with
// EllipticalHaar3Dfeatures
int testKullbackLeiblerMethod();

#endif  //__CELL_DVISION_ROUTINES_H__
