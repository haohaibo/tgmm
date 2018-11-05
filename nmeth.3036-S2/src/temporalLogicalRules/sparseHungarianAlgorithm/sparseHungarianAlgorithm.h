/*
 * See license.txt for full license and copyright notice.
 *
 * \brief Solves bipartite matching assignment using Hungarian algorithm. We
 * want to MINIMIZE the sume of teh weights in the assignment. Check notebook
 * September 28th 2012 for more details.
 *        It is dessigned for sparse graphs to be efficient in memory
 *
 */
#ifndef __SPARSE_HUNGARIAN_ALGORITHM__
#define __SPARSE_HUNGARIAN_ALGORITHM__

// main structure to store information about our problem
typedef struct {
  int N;  // number of input objects
  int M;  // number of candidate objects (we want to match N to M objects)
  int E;  // number of edges
  double thrCost;  // cost assigned to an edge that goes to the garbage
                   // potential
  int* edgesId;    // for edge_i,j this is j. Array of size E
  double* edgesW;  // weight (the lower the better) of each edge. Array of size
                   // E
  int* edgesPtr;   // edgesId[edgesPtr[ i]: edgesPtr[i+1] ) belong to edge i.
                   // Array of size N+1, with edgesPtr[0] and edgesPtr[N] = E.
} workspaceSparseHungarian;

void workspaceSpraseHungarian_init(workspaceSparseHungarian* W, int N_, int M_,
                                   int E_, double thrCost_);
void workspaceSpraseHungarian_destroy(workspaceSparseHungarian* W);

// structure to hold "one to many" assignments
#define MAX_NUMBER_ASSINGMENTS_SHA (30)
typedef struct {
  int numAssignments;
  int assignmentId[MAX_NUMBER_ASSINGMENTS_SHA];
  float assignmentCost[MAX_NUMBER_ASSINGMENTS_SHA];
} assignmentOneToMany;

/*
\brief Solves the bipartite mathcing problem for unique assignment that
MINIMIZES the sum of the assignment
*/
int solveSparseHungarianAlgorithm(const workspaceSparseHungarian* W,
                                  int* assignmentId);

int mainTestSparseHungarianAlgorithm(int argc, const char** argv);

#endif  //__SPARSE_HUNGARIAN_ALGORITHM__
