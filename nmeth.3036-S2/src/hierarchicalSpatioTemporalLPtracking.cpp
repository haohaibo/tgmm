/*
 * See license.txt for full license and copyright notice.
 *
 *  hierarchicalSpatioTemporalLPtracking.cpp
 *
 * \brief routines to implement techniques to track cells using hierarhiccal
 * segmentations + LP tracking
 *
 *
 */

// for memory usage debugging
#if defined(_WIN32) || defined(_WIN64)

#ifdef _DEBUG
#define DEBUG_CLIENTBLOCK new (_CLIENT_BLOCK, __FILE__, __LINE__)
#else
#define DEBUG_CLIENTBLOCK
#endif  // _DEBUG

#define NOMINMAX
#include <Psapi.h>
#include <Windows.h>
#include <crtdbg.h>
#include <stdio.h>
#ifdef _DEBUG
#define new DEBUG_CLIENTBLOCK
#endif
#pragma comment(linker, "/DEFAULTLIB:psapi.lib")
#endif

#include <math.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <unordered_map>
#include <unordered_set>  //hash_table implementation (it should be faster than set (they are trees) )
#include "binaryTree.h"
#include "external/Nathan/tictoc.h"
#include "hierarchicalSpatioTemporalLPtracking.h"
#include "hierarchicalSpatioTemporalLPtrackingCost.h"
#include "lineageHyperTree.h"
#include "utilsAmatf.h"

//#define DEBUG_MEMORY_SL_LP

using namespace std;

bool cplexWorkspaceSpatioTemporalGraph::useUniformSampling = false;
size_t cplexWorkspaceSpatioTemporalGraph::numInsertDebug = 0;
//=============================================================================================================================
// simple struct to store possible cell division edges
struct cellDivisionEdge {
  double cost;
  // vector<int> indAnc;
  int indCh1;
  int indCh2;
  // bool isBoundaryEdge;
  vector<float> f;  // temporalCostFeatures for linear average

  void setValues(double cost_, int indCh1_, int indCh2_,
                 const vector<float>& f_) {
    cost = cost_;
    // indAnc = indAnc_;
    indCh1 = indCh1_;
    indCh2 = indCh2_;
    // isBoundaryEdge = isBoundaryEdge;
    f = f_;
  }

  bool operator<(cellDivisionEdge const& other) const;
};

bool cellDivisionEdge::operator<(cellDivisionEdge const& other) const {
  return (this->cost < other.cost);
};
//==================================================================================================================================
/* This simple routine frees up the pointer *ptr, and sets *ptr to NULL */

void free_and_null(char** ptr) {
  if (*ptr != NULL) {
    free(*ptr);
    *ptr = NULL;
  }
} /* END free_and_null */

//====================================================================================================
void buildHierarchicalSpatioTemporalLPtrackingWithBoundaryConditions(
    vector<hierarchicalSegmentation*>& hsVec,
    vector<vector<supervoxel*> >& svIniVec,
    vector<BinaryTree<supervoxel> >& vecSvLPSolution, bool anchorIniFrame) {
  // TODO: add these parameters in some config file
  unsigned int KmaxNumNN = 10;
  float KmaxDistKNN = 50;
  int devCUDA = 0;
  // only used right now for training
  double thrCost[2] = {0.0, 0.0};  // thrCost[0] -> for cell displacement;
                                   // thrCost[1] -> for cell division. thrCost
                                   // DOES NOT AFFECT GARBAGE POTENTIAL. the
                                   // higher the cost, the better the match
                                   // (think of probabilities). Although with
                                   // structural learning weights, cost can also
                                   // be negative.

  for (int ii = 0; ii < temporalCost::getNumFeaturesCellDisplacement(); ii++)
    thrCost[0] += 0.1 * temporalCost::getWeightsPointer()[ii];
  for (int ii = temporalCost::getNumFeaturesCellDisplacement();
       ii < temporalCost::getNumFeaturesCellDisplacement() +
                temporalCost::getNumFeaturesCellDivision();
       ii++)
    thrCost[1] += 0.1 * temporalCost::getWeightsPointer()[ii];

// cout<<"WARNING:buildHierarchicalSpatioTemporalLPtracking: you need to double
// check thrCost and if you have set the weights for structure learning
// cost!!!"<<endl;

//==============================================
#ifdef DEBUG_MEMORY_SL_LP
  {
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    cout << "DEBUGGING: MEMORY: buildHierarchicalSpatioTemporalLPtracking:: "
            "entry point. Total memory used = "
         << pmc.WorkingSetSize / pow(2.0, 20) << "MB" << endl;
  }
#endif
  //===========================================

  //--------------------------------------------------------------

  int numTM = hsVec.size();  // we assume they are consecutive
  TicTocTimer tt = tic();
  //------------------------------------------------------------------------------
  //------------build spatial graph (a set of hierarchical
  // segmentations)---------
  vector<vector<hierarchicalSegmentation*> > spatialLinksVec(numTM);
  size_t totalNumSv = 0;

  vector<supervoxel*> mapNodeId2Sv;
  mapNodeId2Sv.reserve(10000);  // to have the reverse map: nodeId to supervoxel
  vector<TreeNode<nodeHierarchicalSegmentation>*> mapNode2HSorig;  // to be able
                                                                   // to map
                                                                   // tehfinal
                                                                   // solution
                                                                   // to the
                                                                   // original
                                                                   // node in
                                                                   // the
                                                                   // dendrogram
  mapNode2HSorig.reserve(10000);
  if (svIniVec.empty() == true) {
    cout << "ERROR:buildHierarchicalSpatioTemporalLPtracking: this option is "
            "not implemented yet!!"
         << endl;
    exit(3);
  } else {
    if (anchorIniFrame == true) {
      totalNumSv += constructSpatialLinksAnchored<float>(
          hsVec[0], svIniVec[0], spatialLinksVec[0], mapNodeId2Sv,
          mapNode2HSorig);  // it is important to add nodes from root to leaves
                            // in hsVec so later we can impose HS constraints in
                            // LP easily
    } else {
      totalNumSv += constructSpatialLinks<float>(
          hsVec[0], svIniVec[0], spatialLinksVec[0], mapNodeId2Sv,
          mapNode2HSorig, true);  // it is important to add nodes from root to
                                  // leaves in hsVec so later we can impose HS
                                  // constraints in LP easily
    }
    for (int ii = 1; ii < numTM; ii++) {
      totalNumSv += constructSpatialLinks<float>(
          hsVec[ii], svIniVec[ii], spatialLinksVec[ii], mapNodeId2Sv,
          mapNode2HSorig, true);  // it is important to add nodes from root to
                                  // leaves in hsVec so later we can impose HS
                                  // constraints in LP easily
//==============================================
#ifdef DEBUG_MEMORY_SL_LP
      {
        PROCESS_MEMORY_COUNTERS pmc;
        GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
        cout << "DEBUGGING: MEMORY: "
                "buildHierarchicalSpatioTemporalLPtracking:: after "
                "constructSpatialLInks "
             << ii
             << ". Total memory used = " << pmc.WorkingSetSize / pow(2.0, 20)
             << "MB" << endl;
      }
#endif
      //===========================================
    }
  }

  //-------------------------------------------------------------------
  //-------------build temporal links (including cell
  // division)---------------------------------

  //--------------------debugging--------------------
  /*
  cout<<"====================DEBUGGING: list of supervoxels for
  LP===================="<<endl;
  int countD = 0;
  for(vector<supervoxel*>::iterator iterS = mapNodeId2Sv.begin(); iterS !=
  mapNodeId2Sv.end(); ++iterS, countD++)
  {
          (*iterS)->weightedCentroid<float>();
          int parent = -1;
          if( (*iterS)->nodeHSptr->parent != NULL )
                  parent = (*iterS)->nodeHSptr->parent->nodeId;
          cout<<"Index ="<<countD<<";sv: "<<(*(*iterS))<<"; parentHSidx =
  "<<parent<<endl;
  }
  */
  //------------------------------------------------

  // if ( debugCheckSupervoxelListUniqueness(mapNodeId2Sv, devCUDA) > 0 )
  //	return ;

  // add garbage potential
  mapNodeId2Sv.push_back(NULL);
  totalNumSv++;

  cplexWorkspaceSpatioTemporalGraph temporalLinksW(totalNumSv);
  temporalLinksW.reserve(numTM);  // reserve mmeory for all the elements
  temporalLinksW.temporalWindowTMini = hsVec[0]->basicRegionsVec[0].TM;

  // add edges to the graph (pairwise between adjacent time points)
  cout << toc(&tt) << " secs. "
       << "DEBUGGING SEG FAULT: building temporal graph. Total number of "
          "supervoxels = "
       << totalNumSv << ". svIniVec[0][0] = " << *(svIniVec[0][0]) << endl;
  constructTemporalLinksPairwise(spatialLinksVec[0], spatialLinksVec[1],
                                 temporalLinksW, KmaxNumNN, KmaxDistKNN,
                                 devCUDA, thrCost, true);
  for (int ii = 1; ii < numTM - 1; ii++) {
// cout<<"DEBUGGING SEG FAULT: building temporal graph. ii = "<<ii<<endl;
//==============================================
#ifdef DEBUG_MEMORY_SL_LP
    {
      PROCESS_MEMORY_COUNTERS pmc;
      GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
      cout << "DEBUGGING: MEMORY: buildHierarchicalSpatioTemporalLPtracking:: "
              "constructTemporalLinksPairwise at ii = "
           << ii
           << ". Total memory used = " << pmc.WorkingSetSize / pow(2.0, 20)
           << "MB" << endl;
      temporalLinksW.debugPrintProblemSize(cout);
    }
#endif
    //===========================================
    constructTemporalLinksPairwise(spatialLinksVec[ii], spatialLinksVec[ii + 1],
                                   temporalLinksW, KmaxNumNN, KmaxDistKNN,
                                   devCUDA, thrCost, false);
  }
  // add one edge to garbage potential for each supervoxel in the last time
  // point in order to be able to satisfy constraints without extra boundary
  // conditions
  size_t garbageEdgeId = temporalLinksW.getNumSupervoxels() - 1;
  vector<int> indDesc;
  indDesc.reserve(100);
  queue<TreeNode<nodeHierarchicalSegmentation>*> qDesc;
  for (vector<hierarchicalSegmentation*>::iterator iter =
           spatialLinksVec[numTM - 1].begin();
       iter != spatialLinksVec[numTM - 1].end(); ++iter) {
    for (unsigned int jj = 0; jj < (*iter)->getNumberOfBasicRegions(); jj++) {
      // find a list of descendant for svA (to impose HS constraints). Including
      // svA itself
      TreeNode<nodeHierarchicalSegmentation>* auxNode;
      indDesc.clear();
      qDesc.push((*iter)->basicRegionsVec[jj].nodeHSptr);
      while (qDesc.empty() == false) {
        auxNode = qDesc.front();
        qDesc.pop();
        indDesc.push_back(auxNode->nodeId);
        if (auxNode->left != NULL) qDesc.push(auxNode->left);
        if (auxNode->right != NULL) qDesc.push(auxNode->right);
      }
      temporalLinksW.insertEdge(0.0, indDesc, garbageEdgeId, -1, false,
                                NULL);  // garbage potential is the last one. In
                                        // this case cost = 0 since it is just a
                                        // "dummy" edge
    }
  }

  // release all what is left from cellDivisionEdgePathId to not run into memory
  // problems
  for (vector<hierarchicalSegmentation*>::iterator iter =
           spatialLinksVec[numTM - 1].begin();
       iter != spatialLinksVec[numTM - 1].end(); ++iter) {
    for (unsigned int jj = 0; jj < (*iter)->getNumberOfBasicRegions(); jj++) {
      temporalLinksW
          .cellDivisionEdgePathId
              [(*iter)->basicRegionsVec[jj].nodeHSptr->nodeId]
          .clear();
    }
  }

//==============================================
#ifdef DEBUG_MEMORY_SL_LP
  {
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    cout << "DEBUGGING: MEMORY: buildHierarchicalSpatioTemporalLPtracking:: "
            "after constructTemporalLInks. Total memory used = "
         << pmc.WorkingSetSize / pow(2.0, 20) << "MB" << endl;
    temporalLinksW.debugPrintProblemSize(cout);
  }
#endif
  //===========================================
  //-----------------------debug----------------------------
  /*
  cout<<"===============DEBUGGING: writing out
  edges==========================="<<endl;
  vector<edgesLP> edgesVec;
  getEdgesFromConstraintsLPMatrix(temporalLinksW, mapNodeId2Sv, edgesVec);
  for( size_t ii = 0; ii < edgesVec.size(); ii++)
  {
          int aux = -1;
          if( edgesVec[ii].ch[1] != NULL )
                  aux = edgesVec[ii].ch[1]->nodeHSptr->nodeId;
          int aux0 = -1;
          if( edgesVec[ii].ch[0] != NULL )
                  aux0 = edgesVec[ii].ch[0]->nodeHSptr->nodeId;
          cout<<"Edge "<<ii<<":
  "<<edgesVec[ii].parent->nodeHSptr->nodeId<<";"<<aux0<<";"<<aux<<". Cost =
  "<<edgesVec[ii].cost<<endl;
  }
  */
  cout << toc(&tt) << " secs. "
       << "DEBUGGING: number of Sv=" << temporalLinksW.getNumSupervoxels()
       << ";numedges = " << temporalLinksW.getNumEdges()
       << ";numCellDivEdges =" << temporalLinksW.getNumCellDivisionEdges()
       << endl;
  //--------------------------------------------------------

  //-------------------------------------------------------------------
  //-------------build and solve LP problem---------------------------------
  vector<bool> edgeAssignmentId;
  int err = solveSpatioTemporalLP(temporalLinksW, edgeAssignmentId);
  if (err > 0) return;

  cout << toc(&tt) << " secs. "
       << "DEBUGGING: after solvign LP" << endl;
  // construct lineages from solution
  parseSpatioTemporalLP2Lineages(edgeAssignmentId, temporalLinksW, mapNodeId2Sv,
                                 mapNode2HSorig, vecSvLPSolution);

  //------------------------------------------------------------------------
  // release memory
  for (int ii = 0; ii < numTM; ii++) {
    for (size_t jj = 0; jj < spatialLinksVec[ii].size(); jj++) {
      delete spatialLinksVec[ii][jj];
    }
    spatialLinksVec[ii].clear();
  }
}

//====================================================================================================
cplexWorkspaceSpatioTemporalGraph*
buildHierarchicalSpatioTemporalLPtrackingForStructuralLearningTraining(
    vector<hierarchicalSegmentation*>& hsVec,
    vector<vector<supervoxel*> >& svIniVec, vector<bool>& edgeAssignmentId,
    vector<supervoxel>* mapNodeId2SvToPrint, bool debug) {
  if (svIniVec[0].empty() == true)  // we do not even know how to start
    return NULL;

  // TODO: add these parameters in some config file
  unsigned int KmaxNumNN = 10;
  float KmaxDistKNN = 50;
  int devCUDA = 0;

  //--------------------------------------------------------------

  int numTM = hsVec.size();  // we assume they are consecutive
  //------------------------------------------------------------------------------
  //------------build spatial graph (a set of hierarchical
  // segmentations)---------
  vector<vector<hierarchicalSegmentation*> > spatialLinksVec(numTM);
  size_t totalNumSv = 0;

  vector<supervoxel*> mapNodeId2Sv;
  mapNodeId2Sv.reserve(10000);  // to have the reverse map: nodeId to supervoxel
  vector<TreeNode<nodeHierarchicalSegmentation>*>
      mapNode2HSorig;  // to be able to map tehfinal solution to the original
                       // node in the dendrogram. We do not need this right now
                       // but teh function constructSpatialLinks() needs it
  mapNode2HSorig.reserve(10000);
  if (svIniVec.empty() == true) {
    cout << "ERROR:buildHierarchicalSpatioTemporalLPtracking: this option is "
            "not implemented yet!!"
         << endl;
    exit(3);
  } else {
    // if ground truth has death at TM = ii->svIniVec[ii].empty() = true and I
    // do not generate extension hypothesis for that time point -> I need to
    // make a copy and generate those hypothesis
    vector<vector<supervoxel*> > svIniVecCopy(svIniVec.size());
    for (size_t aa = 0; aa < svIniVec.size(); aa++) {
      if (svIniVec[aa].empty() == false)
        svIniVecCopy[aa] = svIniVec[aa];
      else {
        cout << "DEBUGGING: "
                "buildHierarchicalSpatioTemporalLPtrackingForStructuralLearning"
                "Training: svVecIni empty at ii = "
             << aa << ". Filling extension hypothesis with knn" << endl;
        for (size_t bb = 0; bb < svIniVecCopy[aa - 1].size(); bb++) {
          for (vector<SibilingTypeSupervoxel>::iterator iterS =
                   svIniVecCopy[aa - 1]
                               [bb]->nearestNeighborsInTimeForward.begin();
               iterS !=
               svIniVecCopy[aa - 1][bb]->nearestNeighborsInTimeForward.end();
               ++iterS) {
            svIniVecCopy[aa].push_back(&(*(*iterS)));
          }
        }
      }
    }

    for (int ii = 0; ii < numTM; ii++) {
      totalNumSv += constructSpatialLinks<unsigned short int>(
          hsVec[ii], svIniVecCopy[ii], spatialLinksVec[ii], mapNodeId2Sv,
          mapNode2HSorig, false);  // it is important to add nodes from root to
                                   // leaves in hsVec so later we can impose HS
                                   // constraints in LP easily
    }
  }

  //-------------------------------------------------------------------
  //-------------build temporal links (including cell
  // division)---------------------------------

  if (debug)
    cout << "DEBUGGING: "
            "buildHierarchicalSpatioTemporalLPtrackingForStructuralLearningTrai"
            "ning: build temporal links (including cell division) "
         << endl;
  // add garbage potential
  vector<supervoxel*> mapNodeId2SvCopy(mapNodeId2Sv);  // copy without the NULL
                                                       // temrination (needed
                                                       // for some KNN bussiness
                                                       // later on)
  mapNodeId2Sv.push_back(NULL);
  totalNumSv++;

  cplexWorkspaceSpatioTemporalGraph* temporalLinksW =
      new cplexWorkspaceSpatioTemporalGraph(
          totalNumSv, true);  // we will save temporal features to be able to
                              // use later for structural learning
  temporalLinksW->reserve(numTM);  // reserve mmeory for all the elements
  temporalLinksW->temporalWindowTMini = hsVec[0]->basicRegionsVec[0].TM;

  // add edges to the graph (pairwise between adjacent time points)
  double thrCostMax[2] = {-std::numeric_limits<double>::max(),
                          -std::numeric_limits<double>::max()};
  constructTemporalLinksPairwise(
      spatialLinksVec[0], spatialLinksVec[1], *temporalLinksW, KmaxNumNN,
      KmaxDistKNN, devCUDA, thrCostMax,
      true);  // thrCOst = -1e32 so all costs edges are accepted
  for (int ii = 1; ii < numTM - 1; ii++) {
    constructTemporalLinksPairwise(spatialLinksVec[ii], spatialLinksVec[ii + 1],
                                   *temporalLinksW, KmaxNumNN, KmaxDistKNN,
                                   devCUDA, thrCostMax, false);
  }
  // add one edge to garbage potential for each supervoxel in the last time
  // point in order to be able to satisfy constraints without extra boundary
  // conditions
  size_t garbageEdgeId = temporalLinksW->getNumSupervoxels() - 1;
  vector<int> indDesc;
  indDesc.reserve(100);
  queue<TreeNode<nodeHierarchicalSegmentation>*> qDesc;
  temporalCost f;
  for (vector<hierarchicalSegmentation*>::iterator iter =
           spatialLinksVec[numTM - 1].begin();
       iter != spatialLinksVec[numTM - 1].end(); ++iter) {
    for (unsigned int jj = 0; jj < (*iter)->getNumberOfBasicRegions(); jj++) {
      // find a list of descendant for svA (to impose HS constraints). Including
      // svA itself
      TreeNode<nodeHierarchicalSegmentation>* auxNode;
      indDesc.clear();
      qDesc.push((*iter)->basicRegionsVec[jj].nodeHSptr);
      while (qDesc.empty() == false) {
        auxNode = qDesc.front();
        qDesc.pop();
        indDesc.push_back(auxNode->nodeId);
        if (auxNode->left != NULL) qDesc.push(auxNode->left);
        if (auxNode->right != NULL) qDesc.push(auxNode->right);
      }

      double cost = calculateTemporalCost(f);
      temporalLinksW->insertEdge(
          0.0, indDesc, garbageEdgeId, -std::numeric_limits<double>::max(),
          false, &(f.f[0]));  // garbage potential is the last one. In this case
                              // cost = 0 since it is just a "dummy" edge
    }
  }

  //-------------------------------------------------------------------
  //-------------assign ground truth solution to
  // edgeAssignmentId----------------------------
  if (debug)
    cout << "DEBUGGING: "
            "buildHierarchicalSpatioTemporalLPtrackingForStructuralLearningTrai"
            "ning: assign ground truth solution to edgeAssignmentId "
         << endl;
  // find the correct assignment
  edgeAssignmentId.resize(temporalLinksW->getNumColumns(), false);

  // reset counter so I can identify neighbors
  for (size_t countS = 0; countS < mapNodeId2SvCopy.size();
       countS++)  // we want without the last element beign NULL
    mapNodeId2SvCopy[countS]->tempWildcard = (float)countS;

  vector<supervoxel*> mapNodeId2SvIniVec(
      mapNodeId2Sv.size(), NULL);  // NULL indicates that a supervoxel does not
                                   // correspond to any of the svIniVec elements
  for (size_t ii = 0; ii < svIniVec.size(); ii++) {
    if (svIniVec[ii].empty() == true)
      continue;  // cell death in ground truth, and so we do not have
                 // supervoxels to assign in time point ii

    // if( debug ) cout<<"DEBUGGING:
    // buildHierarchicalSpatioTemporalLPtrackingForStructuralLearningTraining:
    // entering svIniVec for loop ii = "<<ii<<endl;
    // calculate centroid
    for (size_t jj = 0; jj < svIniVec[ii].size(); jj++)
      svIniVec[ii][jj]->weightedCentroid<unsigned short int>();

    // if( debug ) cout<<"DEBUGGING:
    // buildHierarchicalSpatioTemporalLPtrackingForStructuralLearningTraining:
    // kNN calculation. svIniVec[ii].size() = "<<svIniVec[ii].size()<<";
    // mapNodeId2SvCopy.size="<<mapNodeId2SvCopy.size()<<endl;
    vector<vector<vector<supervoxel*>::iterator> > knnVec;
    supervoxel::nearestNeighbors(svIniVec[ii], mapNodeId2SvCopy, 1, 1.0f,
                                 devCUDA, knnVec);

    for (size_t jj = 0; jj < svIniVec[ii].size(); jj++) {
      if (knnVec[jj].empty() == true) {
        cout << "EXCEPTION: "
                "buildHierarchicalSpatioTemporalLPtrackingForStructuralLearning"
                "Training: could not find a match for svIniVec"
             << endl;
        cout << "svIniVec[ii][jj]:" << *(svIniVec[ii][jj]) << endl;

        // TODO: how to delete all elements without using gopto statements in
        // C++ (throw exceptions). Read more here
        // http://en.wikipedia.org/wiki/Resource_Acquisition_Is_Initialization
        // release memory
        for (int ii = 0; ii < numTM; ii++) {
          for (size_t jj = 0; jj < spatialLinksVec[ii].size(); jj++) {
            delete spatialLinksVec[ii][jj];
          }
          spatialLinksVec[ii].clear();
        }
        delete temporalLinksW;
        return NULL;
      }
      // if( debug ) cout<<"DEBUGGING:
      // buildHierarchicalSpatioTemporalLPtrackingForStructuralLearningTraining:
      // entering svIniVec for loop ii = "<<ii<<";jj = "<<jj<<";kNN index =
      // "<<(int)((*(knnVec[jj][0]))->tempWildcard )<<";size =
      // "<<mapNodeId2SvIniVec.size()<<endl;
      mapNodeId2SvIniVec[(int)((*(knnVec[jj][0]))->tempWildcard)] =
          svIniVec[ii][jj];
    }
  }

  // find which edges are correct solutions
  if (debug)
    cout << "DEBUGGING: "
            "buildHierarchicalSpatioTemporalLPtrackingForStructuralLearningTrai"
            "ning:find which edges are correct solutions"
         << endl;
  int parIdx, ch1Idx, ch2Idx;
  int endTM = svIniVec[0].back()->TM;
  for (size_t aa = 0; aa < svIniVec.size(); aa++)  // for times when svIniVec
                                                   // ends prematurely (death in
                                                   // groudn truth)
    if (svIniVec[aa].empty() == false) endTM = svIniVec[aa].back()->TM;

  for (size_t ii = 0; ii < edgeAssignmentId.size(); ii++) {
    temporalLinksW->getEdgeIndex(ii, parIdx, ch1Idx, ch2Idx);
    // if( debug ) cout<<"DEBUGGING:
    // buildHierarchicalSpatioTemporalLPtrackingForStructuralLearningTraining:
    // inisde edgeAssignmentId loop. parIdx = "<<parIdx<<";
    // mapNodeId2SvIniVec.size() = "<<mapNodeId2SvIniVec.size()<<endl;
    // if( debug ) cout<<"DEBUGGING:
    // buildHierarchicalSpatioTemporalLPtrackingForStructuralLearningTraining:
    // inisde edgeAssignmentId loop. ptrAddress =
    // "<<mapNodeId2SvIniVec[parIdx]<<endl;
    if (mapNodeId2SvIniVec[parIdx] != NULL) {
      int matches = 0;
      // find children from tree node
      TreeNode<ChildrenTypeLineage>* ch1 =
          mapNodeId2SvIniVec[parIdx]->treeNode.getParent()->treeNodePtr->left;
      TreeNode<ChildrenTypeLineage>* ch2 =
          mapNodeId2SvIniVec[parIdx]->treeNode.getParent()->treeNodePtr->right;

      if (ch1 == NULL && ch2 != NULL)  // to establish order of comparison
      {
        ch1 = ch2;
        ch2 = NULL;
      }

      if (ch1 == NULL) {  // death (since ch2Idx == NULL also
        if (ch1Idx == (temporalLinksW->getNumSupervoxels() -
                       1))  // indicates death (garbage potential)
        {
          matches++;
        }
      } else {
        if (ch1Idx >= 0 && mapNodeId2SvIniVec[ch1Idx] != NULL) {
          if (ch1 ==
              mapNodeId2SvIniVec[ch1Idx]->treeNode.getParent()->treeNodePtr) {
            matches++;
          }
        }
        if (ch2Idx > 0 && mapNodeId2SvIniVec[ch2Idx] != NULL) {
          if (ch1 ==
              mapNodeId2SvIniVec[ch2Idx]->treeNode.getParent()->treeNodePtr) {
            matches++;
          }
        }
      }

      if (ch2 == NULL) {
        // simple displacement
        if (ch2Idx < 0) {
          matches++;
        }
      } else {
        // cell division, since ch1 != NULL
        if (ch1Idx >= 0 && mapNodeId2SvIniVec[ch1Idx] != NULL) {
          if (ch2 ==
              mapNodeId2SvIniVec[ch1Idx]->treeNode.getParent()->treeNodePtr) {
            matches++;
          }
        }

        if (ch2Idx > 0 && mapNodeId2SvIniVec[ch2Idx] != NULL) {
          if (ch2 ==
              mapNodeId2SvIniVec[ch2Idx]->treeNode.getParent()->treeNodePtr) {
            matches++;
          }
        }
      }

      // to add final edges (last time point to garbage potential )
      if (matches < 2 && mapNodeId2SvIniVec[parIdx]->TM == endTM) matches = 2;

      if (matches == 2)  // we have found solution
      {
        /*
        cout<<"Match found!!"<<endl;
        cout<<*(mapNodeId2SvIniVec[ parIdx ])<<endl;
        if(ch1Idx >= 0 && mapNodeId2SvIniVec[ ch1Idx ] != NULL)
                cout<<*(mapNodeId2SvIniVec[ ch1Idx ])<<endl;
        else
                cout<<"NULL"<<endl;
        if(ch2Idx >= 0 && mapNodeId2SvIniVec[ ch2Idx ] != NULL)
                cout<<*(mapNodeId2SvIniVec[ ch2Idx ])<<endl;
        else
                cout<<"NULL"<<endl;
        */
        edgeAssignmentId[ii] = true;
        mapNodeId2SvIniVec[parIdx] = NULL;  // we have already found the match
      }
    }
  }

  if (debug)
    cout << "DEBUGGING: "
            "buildHierarchicalSpatioTemporalLPtrackingForStructuralLearningTrai"
            "ning: different asserts to make sure everything makes sense"
         << endl;

  // check if solution satisfies constraints
  double c = temporalLinksW->solutionSatisfyConstraints(edgeAssignmentId);
  if (c == numeric_limits<double>::max()) {
    cout << "EXCEPTION: "
            "buildHierarchicalSpatioTemporalLPtrackingForStructuralLearningTrai"
            "ning: mapped ground truth solution does not satisfy constraints"
         << endl;
    delete temporalLinksW;
    temporalLinksW = NULL;
  }

  // all the lements should be null
  for (vector<supervoxel*>::const_iterator iterS = mapNodeId2SvIniVec.begin();
       iterS != mapNodeId2SvIniVec.end(); ++iterS) {
    if ((*iterS) != NULL) {
      cout << "EXCEPTION: "
              "buildHierarchicalSpatioTemporalLPtrackingForStructuralLearningTr"
              "aining: not all elements of ground truth solution were assigned"
           << endl;
      delete temporalLinksW;
      temporalLinksW = NULL;
    }
  }

  // copy supervoxels if we need them to visualize solutions outside this
  // function
  if (mapNodeId2SvToPrint != NULL) {
    mapNodeId2SvToPrint->resize(mapNodeId2Sv.size() - 1);
    for (size_t aa = 0; aa < mapNodeId2Sv.size() - 1;
         aa++)  // last one is NULL for garbage potential
      (*mapNodeId2SvToPrint)[aa] = (*(mapNodeId2Sv[aa]));
  }

  //------------------------------------------------------------------------
  // release memory
  for (int ii = 0; ii < numTM; ii++) {
    for (size_t jj = 0; jj < spatialLinksVec[ii].size(); jj++) {
      delete spatialLinksVec[ii][jj];
    }
    spatialLinksVec[ii].clear();
  }

  return temporalLinksW;
}

//===============================================================================
void cplexWorkspaceSpatioTemporalGraph::insertEdge(double cost,
                                                   vector<int> indAnc,
                                                   int indCh1, int indCh2,
                                                   bool isBoundaryEdge,
                                                   const float* f) {
  //==============================================
  /*
  size_t memStart;
  {
  PROCESS_MEMORY_COUNTERS pmc;
  GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
  memStart = pmc.WorkingSetSize;
  }
  */
  //===========================================

  zobj.push_back(cost);
  zmatbeg.push_back(zmatind.size());

  int cnt = 0;
  for (vector<int>::const_iterator iter = indAnc.begin(); iter != indAnc.end();
       ++iter) {
    zmatind.push_back(*iter);  // first constraint: sum_i e_{i,*} + sum_{a \in
                               // ancestor(i)} e_{a,*} <= 1
    zmatval.push_back(1.0);
    cnt++;
  }
  if (isBoundaryEdge == false)  // it is not a boundary (t != 0) point. See
                                // notebook April 30th 2013 for mroe details
  {
    zmatind.push_back(
        indAnc[0] +
        numSupervoxels);  // second constraint sum_i e_{i,*} = sum_i e_{*,i}
    zmatval.push_back(1.0);
    cnt++;
  } else  // we do not know the second constraint at time 0->we let the LP
          // choose the solution. If you have anchor points, then those points
          // should be the only ones having edges at time 0
  {
    // mark this index as part of time t = 0 (we need this for structural
    // learning training)
    iniTMedgesIdx.push_back(zobj.size() - 1);
    // initialize pathId
    if (cellDivisionEdgePathId[indAnc[0]].empty() == true) {
      cellDivisionEdgePathId[indAnc[0]].insert(numPaths++);
      // numInsertDebug++;
      numRows++;
    }
  }

  zmatind.push_back(indCh1 + numSupervoxels);
  zmatval.push_back(-1.0);
  cnt++;
  // propagate path information
  for (unordered_set<int>::iterator iter =
           cellDivisionEdgePathId[indAnc[0]].begin();
       iter != cellDivisionEdgePathId[indAnc[0]].end(); ++iter) {
    cellDivisionEdgePathId[indCh1].insert(*iter);
    // numInsertDebug++;
  }
  if (indCh2 >= 0)  // cell division edge
  {
    zmatind.push_back(indCh2 + numSupervoxels);
    zmatval.push_back(-1.0);
    cnt++;

    // write path constraint ( 3rd constraint )
    for (unordered_set<int>::iterator iter =
             cellDivisionEdgePathId[indAnc[0]].begin();
         iter != cellDivisionEdgePathId[indAnc[0]].end(); ++iter) {
      zmatind.push_back(numSupervoxels + numSupervoxels + (*iter));
      zmatval.push_back(1.0);
      cnt++;
    }

    numCellDivisionEdges++;

    // propagate path information
    for (unordered_set<int>::iterator iter =
             cellDivisionEdgePathId[indAnc[0]].begin();
         iter != cellDivisionEdgePathId[indAnc[0]].end(); ++iter) {
      cellDivisionEdgePathId[indCh2].insert(*iter);
      // numInsertDebug++;
    }
  }

  zmatcnt.push_back(cnt);

  if (saveTemporalCostFeatures == true) {
    for (long long int ii = 0; ii < temporalCost::getNumFeaturesTotal(); ii++)
      temporalCostFeatures.push_back(f[ii]);
  }

  //==============================================
  /*
#ifdef DEBUG_MEMORY_SL_LP
  size_t memEnd;
  {
  PROCESS_MEMORY_COUNTERS pmc;
  GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
  memEnd = pmc.WorkingSetSize;
  if( memEnd >= memStart )
          cout<<"DEBUGGING: MEMORY: insertEdge:: . indCh2 = "<<indCh2<<";Total
memory allocated = "<<( memEnd - memStart)<<" "<<endl;
  else
          cout<<"DEBUGGING: MEMORY: insertEdge:: . indCh2 = "<<indCh2<<";Total
memory allocated = -"<<( memStart - memEnd)<<" "<<endl;
  }
  #endif
  */
  //===========================================
}

//==================================================================================
void parseSpatioTemporalLP2Lineages(
    const vector<bool>& edgeAssignmentId,
    const cplexWorkspaceSpatioTemporalGraph& W,
    const vector<supervoxel*>& mapNodeId2Sv,
    const vector<TreeNode<nodeHierarchicalSegmentation>*>& mapNode2HSorig,
    vector<BinaryTree<supervoxel> >& vecSvLPSolution) {
  vecSvLPSolution.clear();
  vecSvLPSolution.reserve(W.getNumRows() -
                          2 * W.getNumSupervoxels());  // maximum number of
                                                       // possible paths from
                                                       // time point zero

  int parIdx, ch1Idx, ch2Idx;
  unordered_map<int, TreeNode<supervoxel>*>
      edgeMap;  // to be able to reconstruct pointer
  edgeMap.rehash(
      ceil(0.2 * W.getNumEdges() / edgeMap.max_load_factor()));  // to avoid
  // rehash.reserve(n) is the
  // same as a.rehash(ceil(n /
  // a.max_load_factor()))

  size_t ii = 0;
  int numLineages = 0;
  TreeNode<supervoxel> *auxNode, *parNode;
  for (vector<bool>::const_iterator iter = edgeAssignmentId.begin();
       iter != edgeAssignmentId.end(); ++iter, ii++) {
    if ((*iter) == false) continue;  // this edge is not part of the solution
    W.getEdgeIndex(ii, parIdx, ch1Idx, ch2Idx);

    // check if it is part of a lineage or we need to create a new one
    if (edgeMap.find(parIdx) == edgeMap.end())  // new lineage
    {
      parNode = NULL;
      if (vecSvLPSolution.capacity() ==
          vecSvLPSolution.size())  // THIS CODE DOES NOT WORK IF THERE IS
                                   // REALLOCATION SINCE edgeMap KEEPS POINTERS
                                   // IN THE TREE THAT ARE REALLOCATED IF WE
                                   // NEED TO INCREASE CAPACITY
      {
        cout << "ERROR: at parseSpatioTemporalLP2Lineages: you need to "
                "preallocate more memory"
             << endl;
        exit(3);
      }
      vecSvLPSolution.push_back(BinaryTree<supervoxel>());
      auxNode = vecSvLPSolution.back().insert(*(mapNodeId2Sv[parIdx]));
      auxNode->nodeId = numLineages;  // to know which lineage to look at
      numLineages++;
    } else {  // existing lineage
      parNode = edgeMap[parIdx];
      vecSvLPSolution[parNode->nodeId].SetCurrent(parNode);
      auxNode =
          vecSvLPSolution[parNode->nodeId].insert(*(mapNodeId2Sv[parIdx]));
    }
    if (auxNode == NULL) {
      cout
          << "ERROR: parseSpatioTemporalLP2Lineages: node could not be inserted"
          << endl;
      exit(3);
    }
    auxNode->data.nodeHSptr = mapNode2HSorig[parIdx];  // so supervoxel points
                                                       // to original dendrogram
                                                       // in HS
    // insert node in hash map
    if (ch1Idx >= 0) edgeMap[ch1Idx] = auxNode;
    if (ch2Idx >= 0) edgeMap[ch2Idx] = auxNode;
    if (parNode == NULL)
      auxNode->nodeId = numLineages - 1;
    else
      auxNode->nodeId = parNode->nodeId;
  }

  /*
  //-------------debuging: write out edges--------------------
  vector<edgesLP> edgesVec;
  getEdgesFromConstraintsLPMatrix(W, mapNodeId2Sv, edgesVec);
  ii = 0;
  for(vector<bool>::const_iterator iter = edgeAssignmentId.begin(); iter !=
  edgeAssignmentId.end(); ++iter, ii++)
  {
          if( *iter == true )
          {
                  int aux = -1;
                  if( edgesVec[ii].ch[1] != NULL )
                          aux = edgesVec[ii].ch[1]->nodeHSptr->nodeId;
                  int aux0 = -1;
                  if( edgesVec[ii].ch[0] != NULL )
                          aux0 = edgesVec[ii].ch[0]->nodeHSptr->nodeId;
                  cout<<"Edge "<<ii<<" selected:
  "<<edgesVec[ii].parent->nodeHSptr->nodeId<<";"<<aux0<<";"<<aux<<". Cost =
  "<<edgesVec[ii].cost<<endl;
          }
  }
   */
}

//=====================================================================================
template <class imgTypeC>
size_t constructSpatialLinks(
    hierarchicalSegmentation* hs, vector<supervoxel*>& svIni,
    vector<hierarchicalSegmentation*>& spatialLinks,
    vector<supervoxel*>& mapNodeId2Sv,
    vector<TreeNode<nodeHierarchicalSegmentation>*>& mapNode2HSorig,
    bool selfContained, bool debug) {
  spatialLinks.clear();
  spatialLinks.reserve(svIni.size());

  // find unique root nodes through merging (each unique root node will be a
  // hierarchical segmentation)
  unordered_map<TreeNode<nodeHierarchicalSegmentation>*, float>
      mapMerge;  // to avoid recomputing merge suggestions. TODO: change this to
                 // unordered_map
  mapMerge.rehash(
      ceil(5 * svIni.size() / mapMerge.max_load_factor()));  // to avoid
  // rehash.reserve(n) is
  // the same as
  // a.rehash(ceil(n /
  // a.max_load_factor()))

  float score;
  supervoxel rootSv, rootMergeSv;
  TreeNode<nodeHierarchicalSegmentation> *root, *rootMerge;
  unordered_set<TreeNode<nodeHierarchicalSegmentation>*>
      rootMergeHash;  // checks if an element is unique or not (hash table look
                      // up in average is constant time isntead of logarithmic)
  rootMergeHash.rehash(
      ceil(svIni.size() /
           rootMergeHash.max_load_factor()));  // to avoid rehash.reserve(n) is
                                               // the same as a.rehash(ceil(n /
                                               // a.max_load_factor()))

  // if( debug ) cout<<"DEBUGGING:constructSpatialLinks: entering supervoxel
  // iteration for rootMergeHash"<<endl;

  for (vector<supervoxel*>::const_iterator iterS = svIni.begin();
       iterS != svIni.end(); ++iterS) {
    rootMerge = (*iterS)->nodeHSptr;
    rootMergeSv = *(*iterS);
    score = 1.0f;
    while (score > 0.0f) {
      root = rootMerge;
      rootSv = rootMergeSv;

      if (mapMerge.find(root) !=
          mapMerge.end())  // element already pre-computed
      {
        score = mapMerge[root];
        rootMerge = root->parent;
        hs->supervoxelAtTreeNode(rootMerge, rootMergeSv);
        rootMergeSv.trimSupervoxel<imgTypeC>();
      } else {  // we need to precompute element
        score =
            hs->suggestMerge<imgTypeC>(root, rootSv, &rootMerge, rootMergeSv);

        if (selfContained == true && score > 0.0f)  // check that he proposed
                                                    // merge is done with a
                                                    // supervoxel contained in
                                                    // svIni
        {
          size_t intersectionSize = 0;
          for (vector<supervoxel*>::const_iterator iterSS = svIni.begin();
               iterSS != svIni.end();
               ++iterSS)  // TODO: do it faster by some KNN method
          {
            intersectionSize += rootMergeSv.intersectionSize(*(*iterSS));
          }
          if (((float)intersectionSize) /
                  ((float)rootMergeSv.PixelIdxList.size()) <
              0.9)  // it is not composed by svIni vectors->not allowed to merge
          {
            score = 0.0f;
            rootMergeSv.PixelIdxList.clear();
            rootMerge = NULL;
          }
        }
        mapMerge.insert(
            pair<TreeNode<nodeHierarchicalSegmentation>*, float>(root, score));
      }
    }

    // insert the root of the merging (if it is repeated, it won't be inserted)
    rootMergeHash.insert(root);
  }

  //---------------------------------------------------------------------------------------------
  //---------------------------for each unique root merge node generate the
  // hierarchical tree-----------
  // if( debug ) cout<<"DEBUGGING:constructSpatialLinks: entering for loop to
  // generate hierarchical tree for each root. Number of
  // roots="<<rootMergeHash.size()<<endl;
  TreeNode<nodeHierarchicalSegmentation>* rootSplit[2];
  supervoxel rootSplitSv[2];
  size_t totalN = 0;
  for (unordered_set<TreeNode<nodeHierarchicalSegmentation>*>::iterator iter =
           rootMergeHash.begin();
       iter != rootMergeHash.end(); ++iter) {
    queue<TreeNode<nodeHierarchicalSegmentation> *> q,
        qLocal;  // qHs stores pointers to nodes in local HS; q stores pointers
                 // to nodes in global (or stack wise) HS
    q.push(*iter);

    int maxBasicRegions = min(
        128, 2 *
                 hs->getNumberOfDescendants(
                     *iter));  // maximum number of subnodes that one can have
    hierarchicalSegmentation* hsAux =
        new hierarchicalSegmentation(maxBasicRegions);
    int numNodes = 0;  // to keep count

    TreeNode<nodeHierarchicalSegmentation> *auxNode, *auxLocal;
    nodeHierarchicalSegmentation nodeHS;
    nodeHS.thrTau = numeric_limits<imgVoxelType>::max();  // this will contain
                                                          // the merging
                                                          // probabilities later

    // insert root node into HS
    hs->supervoxelAtTreeNode(
        *iter, hsAux->basicRegionsVec[numNodes]);  // generate supervoxel
    hsAux->basicRegionsVec[numNodes].trimSupervoxel<imgTypeC>();
    nodeHS.svPtr = &(
        hsAux->basicRegionsVec[numNodes]);  // add pointer in node to supervoxel
    hsAux->basicRegionsVec[numNodes].nodeHSptr =
        hsAux->dendrogram.insert(nodeHS);  // add pointer in supervoxel to node
    qLocal.push(hsAux->basicRegionsVec[numNodes].nodeHSptr);
    hsAux->basicRegionsVec[numNodes].weightedGaussianStatistics<imgTypeC>(
        true);  // we need it for nearest neighbors later and for structural
                // learning features

    hsAux->basicRegionsVec[numNodes].nodeHSptr->nodeId = mapNodeId2Sv.size();
    mapNodeId2Sv.push_back(&(hsAux->basicRegionsVec[numNodes]));
    mapNode2HSorig.push_back(*iter);

    numNodes++;

    // if( debug ) cout<<"DEBUGGING:constructSpatialLinks: traversing
    // hierarchical tree for root "<<endl;

    // traverse the HS using split function
    while (q.empty() == false) {
      auxNode = q.front();
      q.pop();
      auxLocal = qLocal.front();
      qLocal.pop();

      // if( debug ) cout<<"DEBUGGING:constructSpatialLinks: check if we can
      // split "<<endl;
      // if( debug ) cout<<"DEBUGGING:constructSpatialLinks: supervoxel "<<
      // *(auxLocal->data.svPtr) <<endl;

      // check if we can split
      score = hs->suggestSplit<imgTypeC>(auxNode, *(auxLocal->data.svPtr),
                                         rootSplit, rootSplitSv);

      if (score > 0) {
        for (int aa = 0; aa < 2; aa++) {
          // if( debug ) cout<<"DEBUGGING:constructSpatialLinks: numNodes =
          // "<<numNodes<<"; numBasicRegions =
          // "<<hsAux->getNumberOfBasicRegions()<<endl;
          if (numNodes >=
              hsAux->getNumberOfBasicRegions())  // allocate more memory
          {
            cout << "WARNING: constructSpatialLinks: trying to save beyond num "
                    "regions. Increase maxBasicRegions ="
                 << maxBasicRegions << endl;
            break;
          }
          // insert child into queue
          q.push(rootSplit[aa]);
          // insert root node into HS
          hsAux->basicRegionsVec[numNodes] = rootSplitSv[aa];
          nodeHS.svPtr = &(hsAux->basicRegionsVec[numNodes]);  // add pointer in
                                                               // node to
                                                               // supervoxel
          hsAux->dendrogram.SetCurrent(auxLocal);
          hsAux->basicRegionsVec[numNodes].nodeHSptr = hsAux->dendrogram.insert(
              nodeHS);  // add pointer in supervoxel to node
          auxLocal->data.thrTau =
              100 - (imgVoxelType)(min(score, 0.95f) *
                                   100);  // here we use it as the merging
                                          // probability. if A = par(B) =
                                          // par(C). Then prob( merge(B,C) ) =
                                          // A->data.thrTau \in [0,100]. 0.95f
                                          // max in order to avoid true zeros
                                          // than can distort cost
          qLocal.push(hsAux->basicRegionsVec[numNodes].nodeHSptr);
          hsAux->basicRegionsVec[numNodes].weightedGaussianStatistics<imgTypeC>(
              true);  // we need it for nearest neighbors later and for
                      // structural learning features

          hsAux->basicRegionsVec[numNodes].nodeHSptr->nodeId =
              mapNodeId2Sv.size();
          mapNodeId2Sv.push_back(&(hsAux->basicRegionsVec[numNodes]));
          mapNode2HSorig.push_back(rootSplit[aa]);

          numNodes++;
        }
      }
      if (numNodes >= hsAux->getNumberOfBasicRegions() &&
          q.empty() == false)  // allocate more memory
      {
        cout << "WARNING: constructSpatialLinks: trying to save beyond num "
                "regions. Increase maxBasicRegions ="
             << maxBasicRegions << endl;
        break;
      }
    }
    hsAux->shrinkBasicRegionsVec(numNodes);

    spatialLinks.push_back(hsAux);  // add local HS to vector of spatial links
    totalN += numNodes;
  }  // end of for(unordered_set< TreeNode< nodeHierarchicalSegmentation >*
     // >::iterator iter = rootMergeHash.begin(); iter != rootMergeHash.end();
     // ++iter )

  return totalN;
}

//=====================================================================================
template <class imgTypeC>
size_t constructSpatialLinksAnchored(
    hierarchicalSegmentation* hs, vector<supervoxel*>& svIni,
    vector<hierarchicalSegmentation*>& spatialLinks,
    vector<supervoxel*>& mapNodeId2Sv,
    vector<TreeNode<nodeHierarchicalSegmentation>*>& mapNode2HSorig) {
  spatialLinks.clear();
  spatialLinks.reserve(svIni.size());

  // in this case each supervoxel at svINi will be a unique root node in the
  // hierarchical segmentation
  unordered_set<TreeNode<nodeHierarchicalSegmentation>*>
      rootMergeHash;  // checks if an element is unique or not (hash table look
                      // up in average is constant time isntead of logarithmic)
  rootMergeHash.rehash(
      ceil(svIni.size() /
           rootMergeHash.max_load_factor()));  // to avoid rehash.reserve(n) is
                                               // the same as a.rehash(ceil(n /
                                               // a.max_load_factor()))

  for (vector<supervoxel*>::const_iterator iterS = svIni.begin();
       iterS != svIni.end(); ++iterS) {
    // insert the root of the merging (if it is repeated, it won't be inserted)
    rootMergeHash.insert((*iterS)->nodeHSptr);
  }

  //---------------------------------------------------------------------------------------------
  //---------------------------for each unique root merge node generate the
  // hierarchical tree-----------
  // if( debug ) cout<<"DEBUGGING:constructSpatialLinks: entering for loop to
  // generate hierarchical tree for each root. Number of
  // roots="<<rootMergeHash.size()<<endl;
  TreeNode<nodeHierarchicalSegmentation>* rootSplit[2];
  supervoxel rootSplitSv[2];
  size_t totalN = 0;
  for (unordered_set<TreeNode<nodeHierarchicalSegmentation>*>::iterator iter =
           rootMergeHash.begin();
       iter != rootMergeHash.end(); ++iter) {
    int maxBasicRegions =
        1;  // in this case we only have one subregion per HS object
    hierarchicalSegmentation* hsAux =
        new hierarchicalSegmentation(maxBasicRegions);
    int numNodes = 0;  // to keep count. Although in this case we only need 1
                       // element( it made copy and paste easier)

    nodeHierarchicalSegmentation nodeHS;
    nodeHS.thrTau = numeric_limits<imgVoxelType>::max();  // this will contain
                                                          // the merging
                                                          // probabilities later

    // insert root node into HS
    hs->supervoxelAtTreeNode(
        *iter, hsAux->basicRegionsVec[numNodes]);  // generate supervoxel
    hsAux->basicRegionsVec[numNodes].trimSupervoxel<imgTypeC>();
    nodeHS.svPtr = &(
        hsAux->basicRegionsVec[numNodes]);  // add pointer in node to supervoxel
    hsAux->basicRegionsVec[numNodes].nodeHSptr =
        hsAux->dendrogram.insert(nodeHS);  // add pointer in supervoxel to node

    hsAux->basicRegionsVec[numNodes].weightedGaussianStatistics<imgTypeC>(
        true);  // we need it for nearest neighbors later and for structural
                // learning features

    hsAux->basicRegionsVec[numNodes].nodeHSptr->nodeId = mapNodeId2Sv.size();
    mapNodeId2Sv.push_back(&(hsAux->basicRegionsVec[numNodes]));
    mapNode2HSorig.push_back(*iter);

    numNodes++;

    spatialLinks.push_back(hsAux);  // add local HS to vector of spatial links
    totalN += numNodes;
  }  // end of for(unordered_set< TreeNode< nodeHierarchicalSegmentation >*
     // >::iterator iter = rootMergeHash.begin(); iter != rootMergeHash.end();
     // ++iter )

  return totalN;
}

//==================================================================================================
// thrCost[0] -> for cell displacement; thrCost[1] -> for cell division
void constructTemporalLinksPairwise(
    vector<hierarchicalSegmentation*>& spatialLinksA,
    vector<hierarchicalSegmentation*>& spatialLinksB,
    cplexWorkspaceSpatioTemporalGraph& temporalLinksW, unsigned int KmaxNumNN,
    float KmaxDistKNN, int devCUDA, double thrCost[2], bool isBoundary) {
//==============================================
#ifdef DEBUG_MEMORY_SL_LP
  {
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    cout << "DEBUGGING: MEMORY: constructTemporalLinksPairwise:: entry point. "
            "Total memory used = "
         << pmc.WorkingSetSize / pow(2.0, 20) << "MB" << endl;
  }
#endif
  //===========================================

  // parse local HS to single vector
  unsigned int numSvA = 0;
  for (vector<hierarchicalSegmentation*>::const_iterator iter =
           spatialLinksA.begin();
       iter != spatialLinksA.end(); ++iter) {
    numSvA += (*iter)->getNumberOfBasicRegions();
  }
  vector<supervoxel*> supervoxelA(numSvA);

  unsigned int numSvB = 0;
  for (vector<hierarchicalSegmentation*>::const_iterator iter =
           spatialLinksB.begin();
       iter != spatialLinksB.end(); ++iter) {
    numSvB += (*iter)->getNumberOfBasicRegions();
  }
  vector<supervoxel*> supervoxelB(numSvB);

  if (numSvA == 0 || numSvB == 0) return;  // no temporal links can be build

  // cout<<"DEBUGGING SEG FAULT: constructTemporalLinksPairwise: building
  // ancestor cost"<<endl;
  unsigned int count = 0;
  double costHS;  // to account for HS tree
  TreeNode<nodeHierarchicalSegmentation>* auxNode;
  vector<double> costHSvec(numSvA);
  for (vector<hierarchicalSegmentation*>::const_iterator iter =
           spatialLinksA.begin();
       iter != spatialLinksA.end(); ++iter) {
    for (unsigned int ii = 0; ii < (*iter)->getNumberOfBasicRegions();
         ii++, count++) {
      supervoxelA[count] = &((*iter)->basicRegionsVec[ii]);

      // calculate ancestor HS cost for each supervoxel (to avoid favoring
      // larger supervoxels)
      costHS = 1.0;
      auxNode =
          supervoxelA[count]->nodeHSptr->parent;  // add the cost depending on
                                                  // the number of ancestors to
                                                  // not promote always the root
                                                  // in the hierarhical
                                                  // segmentation
      while (auxNode != NULL) {
        costHS *= ((double)(auxNode->data.thrTau)) / 100.0;
        auxNode = auxNode->parent;
      }
      costHSvec[count] = costHS;
    }
  }

//==============================================
#ifdef DEBUG_MEMORY_SL_LP
  {
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    cout << "DEBUGGING: MEMORY: constructTemporalLinksPairwise:: before knn. "
            "Total memory used = "
         << pmc.WorkingSetSize / pow(2.0, 20) << "MB" << endl;
  }
#endif
  //===========================================

  count = 0;
  for (vector<hierarchicalSegmentation*>::const_iterator iter =
           spatialLinksB.begin();
       iter != spatialLinksB.end(); ++iter) {
    for (unsigned int ii = 0; ii < (*iter)->getNumberOfBasicRegions(); ii++)
      supervoxelB[count++] = &((*iter)->basicRegionsVec[ii]);
  }

  // calculate nearest neighbors
  // cout<<"DEBUGGING SEG FAULT: constructTemporalLinksPairwise: calculate
  // nearest neighbor"<<endl;
  vector<vector<vector<supervoxel*>::iterator> > nearestNeighborVec;
  if (supervoxel::nearestNeighbors(supervoxelA, supervoxelB, KmaxNumNN,
                                   KmaxDistKNN, devCUDA,
                                   nearestNeighborVec) > 0) {
    exit(3);
  }

//==============================================
#ifdef DEBUG_MEMORY_SL_LP
  {
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    cout << "DEBUGGING: MEMORY: constructTemporalLinksPairwise:: after knn. "
            "Total memory used = "
         << pmc.WorkingSetSize / pow(2.0, 20) << "MB" << endl;
  }
#endif
  //===========================================

  // calculate local geometric descriptors (for both supervoxelA and
  // supervoxelB)
  // cout<<"DEBUGGING SEG FAULT: constructTemporalLinksPairwise: calculate local
  // geometric descriptor"<<endl;
  vector<vector<int> > knnIdxVec;
  vector<float>* refPtsVec =
      localGeometricDescriptor<3>::getRefPts(supervoxelA[0]->TM);
  supervoxel::nearestNeighbors(
      supervoxelA, refPtsVec, supervoxel::getMaxKnnCUDA(),
      localGeometricDescriptor<3>::getNeighRadius(), devCUDA, knnIdxVec);
  float auxPts[dimsImage];
  count = 0;
  for (vector<supervoxel *>::iterator iter = supervoxelA.begin();
       iter != supervoxelA.end(); ++iter, count++) {
    // cout<<"DEBUGGING SEG FAULT:constructTemporalLinksPairwise: calculate
    // local geometric descriptor A. Ptr = "<<*iter<<endl;
    // cout<<"DEBUGGING SEG FAULT:constructTemporalLinksPairwise:
    // cout<<"sv"<<*(*iter)<<".Local desciprtor
    // size="<<(*iter)->gDescriptor.size()<<endl;

    if ((*iter)->gDescriptor.empty() ==
        false)  // we have already calculated this descriptor
      continue;
    (*iter)->gDescriptor.reserveNeighPts(knnIdxVec[count].size());
    for (size_t jj = 0; jj < knnIdxVec[count].size(); jj++) {
      for (int ii = 0; ii < dimsImage; ii++)
        auxPts[ii] = refPtsVec[ii][knnIdxVec[count][jj]];
      (*iter)->gDescriptor.addNeighPts((*iter)->centroid, auxPts);
    }
  }

  // cout<<"DEBUGGING SEG FAULT:constructTemporalLinksPairwise: entering
  // supervoxel B. Vector size = "<<supervoxelB.size()<<endl;
  // for supervoxelB
  refPtsVec = localGeometricDescriptor<3>::getRefPts(supervoxelB[0]->TM);
  supervoxel::nearestNeighbors(
      supervoxelB, refPtsVec, supervoxel::getMaxKnnCUDA(),
      localGeometricDescriptor<3>::getNeighRadius(), devCUDA, knnIdxVec);
  count = 0;
  // cout<<"DEBUGGING SEG FAULT:constructTemporalLinksPairwise: after nearest
  // neighbors"<<endl;
  for (vector<supervoxel *>::iterator iter = supervoxelB.begin();
       iter != supervoxelB.end(); ++iter, count++) {
    // cout<<"DEBUGGING SEG FAULT:constructTemporalLinksPairwise: calculate
    // local geometric descriptor B. Ptr = "<<*iter<<endl;
    // cout<<"DEBUGGING SEG FAULT:constructTemporalLinksPairwise:
    // cout<<"sv"<<*(*iter)<<".Local desciprtor
    // size="<<(*iter)->gDescriptor.size()<<endl;

    if ((*iter)->gDescriptor.empty() ==
        false)  // we have already calculated this descriptor
      continue;
    (*iter)->gDescriptor.reserveNeighPts(knnIdxVec[count].size());
    for (size_t jj = 0; jj < knnIdxVec[count].size(); jj++) {
      for (int ii = 0; ii < dimsImage; ii++)
        auxPts[ii] = refPtsVec[ii][knnIdxVec[count][jj]];
      (*iter)->gDescriptor.addNeighPts((*iter)->centroid, auxPts);
    }
  }

//==============================================
#ifdef DEBUG_MEMORY_SL_LP
  _CrtMemState s1;
  {
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    cout << "DEBUGGING: MEMORY: constructTemporalLinksPairwise:: after local "
            "geometric descriptors. Total memory used = "
         << pmc.WorkingSetSize / pow(2.0, 20) << "MB" << endl;
    // Store a memory checkpoint in the s1 memory-state structure
    _CrtMemCheckpoint(&s1);
  }
#endif
  //===========================================

  // calculate cost of each temporal link and add it to the graph
  // cout<<"DEBUGGING SEG FAULT: constructTemporalLinksPairwise: calculate cost
  // of temporal link"<<endl;
  double cost;
  supervoxel* svA;
  size_t garbageEdgeId = temporalLinksW.getNumSupervoxels() - 1;
  vector<int> indDesc;
  indDesc.reserve(100);
  queue<TreeNode<nodeHierarchicalSegmentation>*> qDesc;
  vector<cellDivisionEdge> vecCellDivisionEdge;
  vecCellDivisionEdge.reserve(KmaxNumNN * KmaxNumNN);
  cellDivisionEdge auxCellDivisionEdge;
  vector<cellDivisionEdge> vecCellDisplacementEdge;
  vecCellDisplacementEdge.reserve(KmaxNumNN);
  cellDivisionEdge auxCellDisplacementEdge;

  temporalCost f;
  for (size_t ii = 0; ii < nearestNeighborVec.size(); ii++) {
    svA = supervoxelA[ii];

    // find a list of descendants for svA (to impose HS constraints). Including
    // svA itself
    TreeNode<nodeHierarchicalSegmentation>* auxNode = svA->nodeHSptr;
    indDesc.clear();
    qDesc.push(svA->nodeHSptr);
    while (qDesc.empty() == false) {
      auxNode = qDesc.front();
      qDesc.pop();
      indDesc.push_back(auxNode->nodeId);
      if (auxNode->left != NULL) qDesc.push(auxNode->left);
      if (auxNode->right != NULL) qDesc.push(auxNode->right);
    }

    vecCellDivisionEdge.clear();
    vecCellDisplacementEdge.clear();
    for (vector<vector<supervoxel*>::iterator>::iterator iter =
             nearestNeighborVec[ii].begin();
         iter != nearestNeighborVec[ii].end();
         ++iter)  // TODO:parallelize this for loop using pollthreads since we
                  // sort elements afterwards
    {
      cost = calculateTemporalCost(
          f, svA, *(*iter));  // TODO:pass a functor double myfunc(supervoxel*,
                              // supervoxel* ) so we can test different cost
                              // functions easily
      int indCh1 = (*(*iter))->nodeHSptr->nodeId;
      if (cost > thrCost[0])  // add edge
      {
        auxCellDisplacementEdge.setValues(-cost * costHSvec[ii], indCh1, -1,
                                          f.f);
        vecCellDisplacementEdge.push_back(auxCellDisplacementEdge);
        // temporalLinksW.insertEdge( -cost * costHSvec[ii], indDesc, indCh1,
        // -1, isBoundary, &(f.f[0]) );//negative cost since we calculate arg
        // min
      }

      // calculate cell division cost: we precompute them first and only select
      // the top candidates (to avoid too many edges)
      for (vector<vector<supervoxel*>::iterator>::iterator iter2 = (iter + 1);
           iter2 != nearestNeighborVec[ii].end(); ++iter2) {
        cost = calculateTemporalCost(
            f, svA, *(*iter), *(*iter2),
            temporalLinksW.temporalWindowTMini);  // TODO:pass a functor double
                                                  // myfunc(supervoxel*,
                                                  // supervoxel* ) so we can
                                                  // test different cost
                                                  // functions easily

        if (cost > thrCost[1])  // add edge
        {
          auxCellDivisionEdge.setValues(-cost * costHSvec[ii], indCh1,
                                        (*(*iter2))->nodeHSptr->nodeId, f.f);
          vecCellDivisionEdge.push_back(auxCellDivisionEdge);
          // temporalLinksW.insertEdge( -cost * costHSvec[ii], indAnc, indCh1,
          // (*(*iter2))->nodeHSptr->nodeId, isBoundary );
        }
      }
    }

    // define maximum number of edges per node(ther eis always one garbage)
    int maxCellDisplacementHypothesis = 5;
    int maxCellDivisiionHypthesis = 3;
    double costBest = numeric_limits<double>::max();
    double thrRatioCostBest =
        0.25;  // to impose a threshold with respect to best cost

    // sort candidates to edges
    sort(vecCellDivisionEdge.begin(), vecCellDivisionEdge.end());
    sort(vecCellDisplacementEdge.begin(), vecCellDisplacementEdge.end());
    // decide best candidate in order to threshodl based on that (this does not
    // affect training samples)

    if (vecCellDisplacementEdge.empty() == false)
      costBest = vecCellDisplacementEdge[0].cost;
    if (vecCellDivisionEdge.empty() == false)
      costBest = min(costBest, vecCellDivisionEdge[0].cost);

    // insert elements for cell displacement
    if (temporalLinksW.saveTemporalCostFeatures ==
        true)  // this is training->I should save all possible cell divisions to
               // make sure ground truth is included
    {
      for (size_t aa = 0; aa < vecCellDisplacementEdge.size(); aa++) {
        // indAnc and indCh1 and isBoundary are constant within thsi loop
        temporalLinksW.insertEdge(vecCellDisplacementEdge[aa].cost, indDesc,
                                  vecCellDisplacementEdge[aa].indCh1,
                                  vecCellDisplacementEdge[aa].indCh2,
                                  isBoundary,
                                  &(vecCellDisplacementEdge[aa].f[0]));
      }
    } else {  // not for training
      for (size_t aa = 0; aa < min(maxCellDisplacementHypothesis,
                                   (int)(vecCellDisplacementEdge.size()));
           aa++) {
        if (vecCellDisplacementEdge[aa].cost / costBest <
            thrRatioCostBest)  // refuse edges that are obviously wrong
          break;
        // indDesc and isBoundary are constant within thsi loop
        temporalLinksW.insertEdge(vecCellDisplacementEdge[aa].cost, indDesc,
                                  vecCellDisplacementEdge[aa].indCh1,
                                  vecCellDisplacementEdge[aa].indCh2,
                                  isBoundary,
                                  &(vecCellDisplacementEdge[aa].f[0]));
      }
    }

    // insert elements for cell division
    if (temporalLinksW.saveTemporalCostFeatures ==
        true)  // this is training->I should save all possible cell divisions to
               // make sure ground truth is included
    {
      for (size_t aa = 0; aa < vecCellDivisionEdge.size(); aa++) {
        // indDesc and isBoundary are constant within thsi loop
        temporalLinksW.insertEdge(vecCellDivisionEdge[aa].cost, indDesc,
                                  vecCellDivisionEdge[aa].indCh1,
                                  vecCellDivisionEdge[aa].indCh2, isBoundary,
                                  &(vecCellDivisionEdge[aa].f[0]));
      }
    } else {  // not training
      for (size_t aa = 0; aa < min(maxCellDivisiionHypthesis,
                                   (int)(vecCellDivisionEdge.size()));
           aa++) {
        if (vecCellDivisionEdge[aa].cost / costBest <
            thrRatioCostBest)  // refuse edges that are obviously wrong
          break;
        // indDesc and isBoundary are constant within thsi loop
        temporalLinksW.insertEdge(vecCellDivisionEdge[aa].cost, indDesc,
                                  vecCellDivisionEdge[aa].indCh1,
                                  vecCellDivisionEdge[aa].indCh2, isBoundary,
                                  &(vecCellDivisionEdge[aa].f[0]));
      }
    }

    // garbage potential: so flow is preserved
    cost = calculateTemporalCost(f);
    temporalLinksW.insertEdge(-cost * costHSvec[ii], indDesc, garbageEdgeId, -1,
                              isBoundary,
                              &(f.f[0]));  // garbage potential is the last one

    // clear temporalLinksW.cellDivisionEdgePathId[ii] because it takes a lot of
    // memory (number of possible paths grows exponentially)
    temporalLinksW.cellDivisionEdgePathId[svA->nodeHSptr->nodeId].clear();
    // temporalLinksW.cellDivisionEdgePathId[ svA->nodeHSptr->nodeId ].insert(
    // 1000000 + svA->nodeHSptr->nodeId);//for debugging purposes
  }

//==============================================
#ifdef DEBUG_MEMORY_SL_LP
  {
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    cout << "DEBUGGING: MEMORY: constructTemporalLinksPairwise:: after "
            "temporal link. Total memory used = "
         << pmc.WorkingSetSize / pow(2.0, 20) << "MB" << endl;
    /*
    _CrtMemState s2;
    _CrtMemCheckpoint( &s2 );
    // Send all reports to STDOUT
_CrtSetReportMode( _CRT_WARN, _CRTDBG_MODE_FILE );
_CrtSetReportFile( _CRT_WARN, _CRTDBG_FILE_STDOUT );
_CrtSetReportMode( _CRT_ERROR, _CRTDBG_MODE_FILE );
_CrtSetReportFile( _CRT_ERROR, _CRTDBG_FILE_STDOUT );
_CrtSetReportMode( _CRT_ASSERT, _CRTDBG_MODE_FILE );
_CrtSetReportFile( _CRT_ASSERT, _CRTDBG_FILE_STDOUT );
    cout<<"DEBUGGING: MEMORY: constructTemporalLinksPairwise:: difference in
heap between two last checks"<<endl;
    _CrtMemState s3;
    int diff = _CrtMemDifference( &s3, &s1, &s2 );
//  _CrtMemDumpStatistics( &s3 );
    _CrtMemDumpAllObjectsSince( &s1 );
    */
  }
#endif
  //===========================================
}

//=====================================================
int debugMainTestHierarchicalSpatioTemporalLPtracking() {
  //------------------------synthetic
  // data-----------------------------------------------
  const int numFrames = 3;
  string fileImage[numFrames] = {
      "E:/syntheticData2/rawData/TM00000/test_00000.tif",
      "E:/syntheticData2/rawData/TM00001/test_00001.tif",
      "E:/syntheticData2/rawData/TM00002/test_00002.tif"};
  string fileHS[numFrames] = {
      "E:/syntheticData2/rawData/TM00000/"
      "test_00000_hierarchicalSegmentation_conn3D74_medFilRad1.bin",
      "E:/syntheticData2/rawData/TM00001/"
      "test_00001_hierarchicalSegmentation_conn3D74_medFilRad1.bin",
      "E:/syntheticData2/rawData/TM00002/"
      "test_00002_hierarchicalSegmentation_conn3D74_medFilRad1.bin"};
  imgVoxelType tau = 1;
  imgVoxelType tauMax = 40;  // to trim hierarchical segmentation
  //------------------------small size test (real
  // data)--------------------------------------
  /*
  const int numFrames = 4;
  string fileImage[numFrames] =
  {"G:/12-08-28/Results/TimeFused.Blending_croppedTest/TM00000_timeFused_blending/SPC0_CM0_CM1_CHN00_CHN01.fusedStack_00000.tif",
          "G:/12-08-28/Results/TimeFused.Blending_croppedTest/TM00001_timeFused_blending/SPC0_CM0_CM1_CHN00_CHN01.fusedStack_00001.tif",
          "G:/12-08-28/Results/TimeFused.Blending_croppedTest/TM00002_timeFused_blending/SPC0_CM0_CM1_CHN00_CHN01.fusedStack_00002.tif",
          "G:/12-08-28/Results/TimeFused.Blending_croppedTest/TM00003_timeFused_blending/SPC0_CM0_CM1_CHN00_CHN01.fusedStack_00003.tif"};

  string fileHS[numFrames] =
  {"G:/12-08-28/Results/TimeFused.Blending_croppedTest/TM00000_timeFused_blending/SPC0_CM0_CM1_CHN00_CHN01.fusedStack_00000_hierarchicalSegmentation_conn3D74_medFilRad2.bin",
          "G:/12-08-28/Results/TimeFused.Blending_croppedTest/TM00001_timeFused_blending/SPC0_CM0_CM1_CHN00_CHN01.fusedStack_00001_hierarchicalSegmentation_conn3D74_medFilRad2.bin",
          "G:/12-08-28/Results/TimeFused.Blending_croppedTest/TM00002_timeFused_blending/SPC0_CM0_CM1_CHN00_CHN01.fusedStack_00002_hierarchicalSegmentation_conn3D74_medFilRad2.bin",
          "G:/12-08-28/Results/TimeFused.Blending_croppedTest/TM00003_timeFused_blending/SPC0_CM0_CM1_CHN00_CHN01.fusedStack_00003_hierarchicalSegmentation_conn3D74_medFilRad2.bin"};

  imgVoxelType tau = 14;
  imgVoxelType tauMax = 200;//to trim hierarchical segmentation
  */
  //---------------------------------------------------------
  // parameters
  float scale[3] = {1.0f, 1.0f, 5.0f};
  supervoxel::setScale(scale);
  int minNucleiSize = 5;  // in voxels
  int maxNucleiSize = 3000;

  float maxPercentileTrimSV = 0.4;  // percentile to trim supervoxel
  int conn3Dsv = 6;                 // connectivity to trim supervoxel

  // for merge split
  const int mergeSplitDeltaZ = 11;
  int64 boundarySizeIsNeigh[dimsImage];
  int conn3DIsNeigh = 74;
  int64* neighOffsetIsNeigh = supervoxel::buildNeighboorhoodConnectivity(
      conn3DIsNeigh + 1, boundarySizeIsNeigh);  // using the special
                                                // neighborhood for coarse
                                                // sampling

  supervoxel::pMergeSplit.setParam(conn3DIsNeigh, neighOffsetIsNeigh,
                                   mergeSplitDeltaZ);
  delete[] neighOffsetIsNeigh;
  //--------------------------------------------------------
  // load images
  mylib::Array* imVec[numFrames];
  for (int ii = 0; ii < numFrames; ii++) {
    imVec[ii] = mylib::Read_Image((char*)(fileImage[ii].c_str()), 0);

    if (imVec[ii] == NULL) {
      std::cout << "ERROR: reading image file " << endl;
      exit(3);
    }
    // convert to float
    mylib::Convert_Array_Inplace(imVec[ii], mylib::PLAIN_KIND,
                                 mylib::FLOAT32_TYPE, 1, 1.0);
    mylib::Value v1a, v0a;
    v1a.fval = 1.0;
    v0a.fval = 0.0;
    mylib::Scale_Array_To_Range(imVec[ii], v0a, v1a);
  }

  // load hierarchical segmentation
  vector<hierarchicalSegmentation*> hsVec(numFrames);
  for (int ii = 0; ii < numFrames; ii++) {
    ifstream fin(fileHS[ii].c_str(), ios::binary | ios::in);
    if (fin.is_open() == false) {
      cout << "ERROR: opening HS files" << endl;
      exit(3);
    }
    hsVec[ii] = new hierarchicalSegmentation(fin);
    fin.close();

    for (unsigned int jj = 0; jj < hsVec[ii]->getNumberOfBasicRegions(); jj++) {
      hsVec[ii]->basicRegionsVec[jj].dataPtr = imVec[ii]->data;
      hsVec[ii]->basicRegionsVec[jj].dataSizeInBytes =
          sizeof(float) * imVec[ii]->size;
      hsVec[ii]->basicRegionsVec[jj].TM = ii;
    }
    hsVec[ii]->segmentationAtTau(tau);
    hsVec[ii]->setMaxTau(tauMax);
  }

  // setup trimming parameters
  supervoxel::setTrimParameters(maxNucleiSize, maxPercentileTrimSV, conn3Dsv);

  // we add all the supervoxels
  vector<vector<supervoxel*> > svVecIni(numFrames);
  for (int ii = 0; ii < numFrames; ii++) {
    for (unsigned int jj = 0;
         jj < hsVec[ii]->currentSegmentatioSupervoxel.size(); jj++) {
      hsVec[ii]->currentSegmentatioSupervoxel[jj].trimSupervoxel<float>();
      svVecIni[ii].push_back(&(hsVec[ii]->currentSegmentatioSupervoxel[jj]));
    }
  }

  // call main LP function
  vector<BinaryTree<supervoxel> > vecLPsolution;
  buildHierarchicalSpatioTemporalLPtracking(hsVec, svVecIni, vecLPsolution);

  // release memory
  for (int ii = 0; ii < numFrames; ii++) {
    mylib::Free_Array(imVec[ii]);
    delete hsVec[ii];
  }
}

//=================================================================
int debugCheckSupervoxelListUniqueness(vector<supervoxel*>& svVec,
                                       int devCUDA) {
  cout << "=======DEBUGGING:debugCheckSupervoxelListUniqueness: make sure "
          "centroids are present====================="
       << endl;
  unsigned int KmaxNumNN = 5;
  float KmaxDistKNN = 2.0f;

  // calculate nearest neighbors
  vector<vector<vector<supervoxel*>::iterator> > nearestNeighborVec;
  if (supervoxel::nearestNeighbors(svVec, svVec, KmaxNumNN, KmaxDistKNN,
                                   devCUDA, nearestNeighborVec) > 0) {
    cout << "ERROR: calculating nearest neighbors" << endl;
    return 3;
  }

  // check for uniqueness
  for (size_t ii = 0; ii < svVec.size(); ii++) {
    if (nearestNeighborVec[ii].size() <= 1) continue;
    int numEq = 0;
    for (vector<vector<supervoxel*>::iterator>::iterator iter =
             nearestNeighborVec[ii].begin();
         iter != nearestNeighborVec[ii].end(); ++iter) {
      if (svVec[ii]->TM == (*(*iter))->TM &&
          svVec[ii]->PixelIdxList.size() == (*(*iter))->PixelIdxList.size() &&
          svVec[ii]->Euclidean2Distance(*(*(*iter))) < 0.5)
        numEq++;
    }

    if (numEq > 1) {
      cout << "ERROR: found repeat : " << (*(svVec[ii])) << endl;
      return numEq;
    }
  }

  return 0;
}

//===================================================================================
void getEdgesFromConstraintsLPMatrix(const cplexWorkspaceSpatioTemporalGraph& W,
                                     const vector<supervoxel*>& mapSvIdx,
                                     vector<edgesLP>& edgesVec) {
  edgesVec.resize(W.getNumEdges());

  int parIdx, ch1Idx, ch2Idx;
  for (size_t ii = 0; ii < edgesVec.size(); ii++) {
    edgesVec[ii].cost = W.zobj[ii];
    // VIP: zmatind.size() = number of edges (= number of random variables). if
    // m-th edge is e_{i,j} -> zmatind[ zmatbeg[m] ] = i; zmatind[ zmatbeg[m] +
    // zmatcnt[m] - 2 ] = j - numSupervoxels; zmatind[ zmatbeg[m] + zmatcnt[m] -
    // 1 ] = k - numSupervoxels(for cell division edges)

    W.getEdgeIndex(ii, parIdx, ch1Idx, ch2Idx);

    edgesVec[ii].parent = mapSvIdx[parIdx];
    edgesVec[ii].ch[0] = mapSvIdx[ch1Idx];
    if (ch2Idx >= 0)
      edgesVec[ii].ch[1] = mapSvIdx[ch2Idx];
    else
      edgesVec[ii].ch[1] = NULL;
  }
}

//==============================================================================
void cplexWorkspaceSpatioTemporalGraph::
    addConstraintsToAvoidMultipleCellDivisionsInOneBranch() {
  cout << "ERROR: addConstraintsToAvoidMultipleCellDivisionsInOneBranch: added "
          "function not implemented yet!!!!"
       << endl;
  exit(3);

  vector<bool> edgeVisited(getNumEdges(), false);
  size_t numVisitedCellDivisionEdges = 0;

  size_t pp = 0;
  int parIdx, ch1Idx, ch2Idx;
  while (numVisitedCellDivisionEdges < numCellDivisionEdges) {
    if (edgeVisited[pp] == true) {
      pp++;
      continue;
    }

    getEdgeIndex(pp, parIdx, ch1Idx, ch2Idx);
    if (ch2Idx < 0)  // not a cell division edge
    {
      edgeVisited[pp] = true;
      pp++;
      continue;
    }

    // start a new row containing all paths that start at this new cell division
    // edge

    // update number of rows
    numRows++;
  }

  // cout<<"DEBUGGING: addConstraintsToAvoidMultipleCellDivisionsInOneBranch:
  // added "<< numRows - 2* numSupervoxels<<" rows for constraint 3"<<endl;
}

//==========================================================================
double cplexWorkspaceSpatioTemporalGraph::solutionSatisfyConstraints(
    const vector<bool>& e, vector<double>& Ae) const {
  if (e.size() != getNumEdges()) {
    cout << "WARNING: "
            "cplexWorkspaceSpatioTemporalGraph::solutionSatisfyConstraints: "
            "proposed solution has different size than current problem!!"
         << endl;
    return numeric_limits<double>::max();  // not feasible
  }

  Ae.resize(getNumRows());
  memset(&(Ae[0]), 0, sizeof(double) * Ae.size());

  double c = 0;  // cost of the solution
  int row;

  for (size_t col = 0; col < getNumColumns(); col++) {
    if (e[col] == true) {
      c += zobj[col];

      for (int ii = zmatbeg[col]; ii < zmatbeg[col] + zmatcnt[col]; ii++) {
        row = zmatind[ii];
        Ae[row] +=
            zmatval[ii];  // zmatval[ii] * e[col]  (but we know e[col] == 1)
        if (Ae[row] > 1.5)
          return numeric_limits<double>::max();  // not feasible. All
                                                 // constraints are below or
                                                 // equal to 1 so we can already
                                                 // tell this is not feasible
      }
    }
  }

  // check if constraints are satisfied (this has ot match setProblemData)
  int ii;
  for (ii = 0; ii < getNumSupervoxels() - 1; ii++) {
    // all the N input objects need to be assigned (even if it is just to teh
    // garbage potential)
    // zsense[ii]='L';//look
    // http://www.rpi.edu/dept/math/math-programming/cplex66/sun4x_58/doc/refman/html/copylp.html#1001201
    // for the different possible values
    // zrhs[ii]=1.0;
    if (Ae[ii] > 1.0) return numeric_limits<double>::max();  // not feasible
  }
  // zrhs[ii-1] = NUMCOLS;//garbage potential
  ii++;

  for (; ii < 2 * getNumSupervoxels() - 1; ii++) {
    // right hand side of constraints
    // zsense[ii]='E';//not all the candidate elements have to receive a
    // connection. At the most each object (except garbage potential) receives 1
    // element
    // zrhs[ii]=0.0;
    if (Ae[ii] != 0.0) return numeric_limits<double>::max();  // not feasible
  }

  // zsense[ii-1] = 'L';//garbage potential
  // zrhs[ii-1] = NUMCOLS;//garbage potential
  ii++;

  // third constraint: avoid multiple cell divisions in the same path
  for (; ii < getNumRows(); ii++) {
    // zsense[ii]='L';//look
    // http://www.rpi.edu/dept/math/math-programming/cplex66/sun4x_58/doc/refman/html/copylp.html#1001201
    // for the different possible values
    // zrhs[ii]=1.0;
    if (Ae[ii] > 1.0) return numeric_limits<double>::max();  // not feasible
  }

  return c;
}

//===========================================================
void cplexWorkspaceSpatioTemporalGraph::debugWriteLPSparseConstraintMatrix(
    string filename) {
  cout << "DEBUGGING: "
          "cplexWorkspaceMatchSeedsToHS::debugWriteSparseConstraintMatrix: "
          "output file "
       << filename << endl;
  ofstream fout(filename.c_str());

  if (!fout.is_open()) {
    cout << "ERROR: "
            "cplexWorkspaceSpatioTemporalGraph::"
            "debugWriteSparseConstraintMatrix: file "
         << filename << " could not be opened" << endl;
  }

  // write out constraint matrix
  size_t pos = 0;
  for (size_t ii = 0; ii < getNumColumns(); ii++) {
    for (int jj = 0; jj < zmatcnt[ii]; jj++) {
      fout << zmatind[pos] << "," << ii << "," << zmatval[pos] << ";" << endl;
      pos++;
    }
  }

  fout.close();
}

//======================================================================================================================================
void cplexWorkspaceSpatioTemporalGraph::debugWriteLPCostVector(
    string filename) {
  cout << "DEBUGGING: cplexWorkspaceMatchSeedsToHS::debugWriteLPCostVector: "
          "output file "
       << filename << endl;
  ofstream fout(filename.c_str());

  if (!fout.is_open()) {
    cout << "ERROR: cplexWorkspaceSpatioTemporalGraph::debugWriteLPCostVector: "
            "file "
         << filename << " could not be opened" << endl;
  }

  // write out constraint matrix
  int parIdx, ch1Idx, ch2Idx;
  for (size_t ii = 0; ii < zobj.size(); ii++) {
    getEdgeIndex(ii, parIdx, ch1Idx, ch2Idx);
    fout << parIdx << "," << ch1Idx << "," << ch2Idx << "," << zobj[ii] << ";"
         << endl;
  }

  fout.close();
}

//======================================================================================================================================
void cplexWorkspaceSpatioTemporalGraph::debugWriteLPsolution(
    string filename, vector<bool>& edgeAssignmentId) {
  cout << "DEBUGGING: cplexWorkspaceMatchSeedsToHS::debugWriteLPsolution: "
          "output file "
       << filename << endl;
  ofstream fout(filename.c_str());

  if (!fout.is_open()) {
    cout << "ERROR: cplexWorkspaceSpatioTemporalGraph::debugWriteLPsolution: "
            "file "
         << filename << " could not be opened" << endl;
  }

  // write out constraint matrix
  int parIdx, ch1Idx, ch2Idx;
  for (size_t ii = 0; ii < zobj.size(); ii++) {
    getEdgeIndex(ii, parIdx, ch1Idx, ch2Idx);
    fout << edgeAssignmentId[ii] << "," << parIdx << "," << ch1Idx << ","
         << ch2Idx << "," << zobj[ii] << ";" << endl;
  }

  fout.close();
}

//======================================================================================================================================
void cplexWorkspaceSpatioTemporalGraph::debugPrintProblemSize(ostream& out) {
  out << "DEBUGGING: cplexWorkspaceSpatioTemporalGraph::debugPrintProblemSize"
      << endl;
  out << "NumSV = " << numSupervoxels << ";numRows = " << numRows
      << ";numCellDivisionEdges = " << numCellDivisionEdges
      << "; saveTemporalCostFeatures = " << saveTemporalCostFeatures << endl;
  out << "Size of \t zobj = " << zobj.size() << "; zmatbeg = " << zmatbeg.size()
      << "; zmatcnt = " << zmatcnt.size() << "; zmatind = " << zmatind.size()
      << "; zmatval = " << zmatval.size() << endl;
  out << "Capacity of \t zobj = " << zobj.capacity()
      << "; zmatbeg = " << zmatbeg.capacity()
      << "; zmatcnt = " << zmatcnt.capacity()
      << "; zmatind = " << zmatind.capacity()
      << "; zmatval = " << zmatval.capacity() << endl;
  out << "Size of \t parIdxPtr = " << parIdxPtr.size()
      << "; iniTMedgesIdx = " << iniTMedgesIdx.size()
      << "; garbageIdx = " << garbageIdx.size()
      << "; temporalCostFeatures = " << temporalCostFeatures.size() << endl;
  out << "Capacity of \t parIdxPtr = " << parIdxPtr.capacity()
      << "; iniTMedgesIdx = " << iniTMedgesIdx.capacity()
      << "; garbageIdx = " << garbageIdx.capacity()
      << "; temporalCostFeatures = " << temporalCostFeatures.capacity() << endl;
  size_t n = 0, nU = 0, m = 0;
  for (size_t ii = 0; ii < cellDivisionEdgePathId.size(); ii++) {
    n += cellDivisionEdgePathId[ii].size();
    nU += cellDivisionEdgePathId[ii].size();
    m += sizeof(cellDivisionEdgePathId[ii]);
    // cout<<cellDivisionEdgePathId[ii].size()<<" , ";
  }
  out << endl
      << "Num paths = " << numPaths
      << "; cellDivisionEdgePathId.size() = " << cellDivisionEdgePathId.size()
      << ". Total int elements = " << n
      << ". Total insertions trials = " << numInsertDebug << endl;
  cout << "WARNING: std::unordered_set cellDivisionEdgePathId takes much more "
          "space than a list of ints. Assymptotically, in VS2010 ratio is 1:6"
       << endl;
  size_t memRatioHash = 6;
  n *= memRatioHash;
  nU *= memRatioHash;

  size_t mem =
      sizeof(double) * (zmatval.capacity() + zobj.capacity()) +
      sizeof(size_t) * (parIdxPtr.capacity() + iniTMedgesIdx.capacity() +
                        garbageIdx.capacity()) +
      sizeof(int) *
          (nU + zmatbeg.capacity() + zmatcnt.capacity() + zmatind.capacity()) +
      sizeof(float) * temporalCostFeatures.capacity();
  mem += sizeof(zmatval) + sizeof(zobj) + sizeof(parIdxPtr) +
         sizeof(iniTMedgesIdx) + sizeof(garbageIdx) +
         sizeof(cellDivisionEdgePathId) + sizeof(zmatbeg) + sizeof(zmatcnt) +
         sizeof(zmatind) + sizeof(temporalCostFeatures) + m;
  size_t memU =
      sizeof(double) * (zmatval.size() + zobj.size()) +
      sizeof(size_t) *
          (parIdxPtr.size() + iniTMedgesIdx.size() + garbageIdx.size()) +
      sizeof(int) * (n + zmatbeg.size() + zmatcnt.size() + zmatind.size()) +
      sizeof(float) * temporalCostFeatures.size();
  memU += sizeof(zmatval) + sizeof(zobj) + sizeof(parIdxPtr) +
          sizeof(iniTMedgesIdx) + sizeof(garbageIdx) +
          sizeof(cellDivisionEdgePathId) + sizeof(zmatbeg) + sizeof(zmatcnt) +
          sizeof(zmatind) + sizeof(temporalCostFeatures) + m;
  out << "Total (estimate) reserved memory = " << ((double)mem) / pow(2.0, 20)
      << "MB"
      << ". Used = " << ((double)memU) / pow(2.0, 20) << "MB" << endl;
}

//======================================================================
void cplexWorkspaceSpatioTemporalGraph::temporalCostFeaturesGivenASolution(
    const vector<bool>& e, vector<float>& fVec) const {
  fVec.resize(temporalCost::getNumFeaturesTotal());
  memset(&(fVec[0]), 0, sizeof(float) * fVec.size());

  long long int offset = temporalCost::getNumFeaturesTotal();
  long long int pos = 0;
  for (vector<bool>::const_iterator iter = e.begin(); iter != e.end();
       ++iter, pos += offset) {
    if (*iter == true) {
      for (size_t ii = 0; ii < fVec.size(); ii++) {
        fVec[ii] += temporalCostFeatures[pos + ii];
      }
    }
  }
}

//===============================================================================================
void cplexWorkspaceSpatioTemporalGraph::
    generateRandomFeasibleSolutionWithHighProbability(vector<bool>& eRand,
                                                      mylib::CDF* masterCDF) {
  eRand.resize(getNumEdges());
  for (size_t ii = 0; ii < eRand.size(); ii++)  // reset eRandom (not using
                                                // memset because of
                                                // vector<bool> compact
                                                // representation)
    eRand[ii] = false;

  // reset parIdxPtr if needed
  if (parIdxPtr.size() != numSupervoxels + 1) setParIdxPtr();

  // calculate garbageIdx if needed
  if (garbageIdx.size() != numSupervoxels) setGarbageIdx();

  size_t nnzOld = 10;  // to check if we have added one more solution
  size_t nnz = 0;

  // find possible candidates among the first time point
  vector<int> nodeIdxVec(iniTMedgesIdx.size());
  int parIdx, ch1Idx, ch2Idx;

  for (int ii = 0; ii < iniTMedgesIdx.size(); ii++) {
    getEdgeIndex(iniTMedgesIdx[ii], nodeIdxVec[ii], ch1Idx, ch2Idx);
  }
  // remove repeated elements
  sort(nodeIdxVec.begin(), nodeIdxVec.end());
  vector<int>::iterator it = unique(nodeIdxVec.begin(), nodeIdxVec.end());
  nodeIdxVec.resize(std::distance(nodeIdxVec.begin(), it));

  // try to generate multiple paths
  while (nnz != nnzOld) {
    // recursive function to build path at random
    extendRandomFeasibleSolutionWithHighProbability(nodeIdxVec, eRand,
                                                    masterCDF);

    // count number of noon-zero in the solution
    nnzOld = nnz;
    nnz = 0;
    for (vector<bool>::const_iterator iter = eRand.begin(); iter != eRand.end();
         ++iter) {
      if ((*iter) == true) nnz++;
    }
  }
}

//================================================================
void cplexWorkspaceSpatioTemporalGraph::
    extendRandomFeasibleSolutionWithHighProbability(vector<int>& nodeIdxVec,
                                                    vector<bool>& eRand,
                                                    mylib::CDF* masterCDF) {
  size_t garbageEdgeId = getNumSupervoxels() - 1;

  vector<int> possibleEdgesIdx;
  vector<double> possibleEdgesWeights;  // weights (normalized cost) to
                                        // calculate probabilities of being
                                        // selected
  possibleEdgesIdx.reserve(getNumEdges() / 10);
  possibleEdgesWeights.reserve(getNumEdges() / 10);
  vector<double> Ae;  // to preallocate memory

  size_t auxIdx;
  int nodeIdx;
  for (size_t kk = 0; kk < nodeIdxVec.size(); kk++) {
    nodeIdx = nodeIdxVec[kk];
    if (nodeIdx < 0 ||
        nodeIdx >= garbageEdgeId)  // it means we have reached an end
      continue;

    for (size_t ii = parIdxPtr[nodeIdx]; ii < parIdxPtr[nodeIdx + 1]; ii++) {
      auxIdx = ii;
      if (eRand[auxIdx] == true)  // already in the solution
        continue;

      // test if this solution would be feasible
      eRand[auxIdx] = true;
      // I need to add the edge to the garbage potential, or the conservation of
      // flow constraint would not be satisfied
      int parIdx, ch1Idx, ch2Idx;
      getEdgeIndex(auxIdx, parIdx, ch1Idx, ch2Idx);
      bool reverseCh1 = true, reverseCh2 = true;  // flags to mark if the edge
                                                  // to the garbage collector
                                                  // was already marked
      if (ch1Idx < garbageEdgeId) {
        if (eRand[garbageIdx[ch1Idx]] == true) reverseCh1 = false;
        eRand[garbageIdx[ch1Idx]] = true;
      }
      if (ch2Idx >= 0) {
        if (eRand[garbageIdx[ch2Idx]] == true) reverseCh2 = false;
        eRand[garbageIdx[ch2Idx]] = true;
      }

      if (solutionSatisfyConstraints(eRand, Ae) ==
          numeric_limits<double>::max())  // does not satisfy constraints
      {
        eRand[auxIdx] = false;
        if (ch1Idx < garbageEdgeId && reverseCh1 == true) {
          eRand[garbageIdx[ch1Idx]] = false;
        }
        if (ch2Idx >= 0 && reverseCh2 == true) {
          eRand[garbageIdx[ch2Idx]] = false;
        }
        continue;
      }

      // undo and add edges to posible solution
      eRand[auxIdx] = false;
      if (ch1Idx < garbageEdgeId && reverseCh1 == true) {
        eRand[garbageIdx[ch1Idx]] = false;
      }
      if (ch2Idx >= 0 && reverseCh2 == true) {
        eRand[garbageIdx[ch2Idx]] = false;
      }
      possibleEdgesIdx.push_back(auxIdx);
      possibleEdgesWeights.push_back(
          -zobj[auxIdx]);  // negative because in structure learning QP code we
                           // try to get (f^{star} - f^{hat})*w >= 1 - epsilon,
                           // while in LP C++ code we want to minimize
    }
  }

  if (possibleEdgesIdx.empty() == true)  // no way to continue
  {
    return;
  }

  // due to struture learning weights, possibleEdgesWeights might contain
  // negative values -> offset to be able to transform them into probabilities
  double minElem =
      *min_element(possibleEdgesWeights.begin(), possibleEdgesWeights.end());
  for (vector<double>::iterator iterE = possibleEdgesWeights.begin();
       iterE != possibleEdgesWeights.end(); ++iterE)
    (*iterE) -=
        (minElem +
         0.1);  // small offset so smaller weight still has positive probability

  // set them at uniform if required by user (usually teh first pass of
  // training)
  if (useUniformSampling == true) {
    for (vector<double>::iterator iterE = possibleEdgesWeights.begin();
         iterE != possibleEdgesWeights.end(); ++iterE)
      (*iterE) = 1.0;
  }

  // select one edge at random according to weighted probability
  mylib::CDF* rnd = mylib::Bernouilli_CDF(
      possibleEdgesWeights.size(),
      &(possibleEdgesWeights[0]));  // it normalizes the weights
  mylib::Link_CDF(masterCDF, rnd);  // so they share state

  int selectedIdx = possibleEdgesIdx[((size_t)(mylib::Sample_CDF(rnd))) -
                                     1];  // bernoulli returns between [1,n]
  eRand[selectedIdx] = true;
  // release memory
  mylib::Free_CDF(rnd);

  // call recursively to continue generating solution
  // continue path
  int parIdx, ch1Idx, ch2Idx;
  getEdgeIndex(selectedIdx, parIdx, ch1Idx, ch2Idx);

  vector<int> ch1NodeIdxVec(1, ch1Idx);
  if (ch2Idx < 0)  // no cell dvision
  {
    extendRandomFeasibleSolutionWithHighProbability(ch1NodeIdxVec, eRand,
                                                    masterCDF);
  } else {  // cell division proposed
    bool reverseCh2 = true;
    if (eRand[garbageIdx[ch2Idx]] == true) reverseCh2 = false;
    eRand[garbageIdx[ch2Idx]] =
        true;  // so solution is still feasible while extending ch1

    extendRandomFeasibleSolutionWithHighProbability(ch1NodeIdxVec, eRand,
                                                    masterCDF);

    if (reverseCh2 == true)  // undo edge to garbage potential for ch2
      eRand[garbageIdx[ch2Idx]] = false;
    vector<int> ch2NodeIdxVec(1, ch2Idx);
    extendRandomFeasibleSolutionWithHighProbability(ch2NodeIdxVec, eRand,
                                                    masterCDF);
  }
}

//=================================================================
void cplexWorkspaceSpatioTemporalGraph::setParIdxPtr() {
  parIdxPtr.resize(numSupervoxels + 1);
  parIdxPtr[0] = 0;

  int parIdx, ch1Idx, ch2Idx;

  int parIdxOld = 0;

  for (size_t ii = 0; ii < getNumEdges(); ii++) {
    getEdgeIndex(ii, parIdx, ch1Idx, ch2Idx);

    if (parIdx < parIdxOld ||
        parIdx > parIdxOld + 1)  // they should be consecutive
    {
      // Note: we are assuming every supervoxel has at least one edge. BUT
      // "GARBAGE" SUPERVOXEL DOES NOT HAVE ANY
      cout << "ERROR: cplexWorkspaceSpatioTemporalGraph::setParIdxPtr(): "
              "parIdx = "
           << parIdx << " and parIdxOld=" << parIdxOld
           << " should be consecutive" << endl;
      exit(3);
    }

    if (parIdx != parIdxOld) {
      parIdxOld = parIdx;
      parIdxPtr[parIdxOld] = ii;
    }
  }

  parIdxPtr[parIdxOld + 1] = getNumEdges();
}

//=================================================================
void cplexWorkspaceSpatioTemporalGraph::setGarbageIdx() {
  garbageIdx =
      vector<size_t>(numSupervoxels, -1);  // reinitialize everything to -1
  size_t garbageEdgeId = getNumSupervoxels() - 1;
  int parIdx, ch1Idx, ch2Idx;
  int nn = 0;
  for (size_t ii = 0; ii < getNumEdges(); ii++) {
    getEdgeIndex(ii, parIdx, ch1Idx, ch2Idx);

    if (ch1Idx == garbageEdgeId)  // garbage potential
    {
      garbageIdx[parIdx] = ii;
      nn++;
    }
  }

  assert((nn + 1) == numSupervoxels);
}

void cplexWorkspaceSpatioTemporalGraph::
    debugPrintLPSolutionForVisualizationMatlab(
        const vector<vector<bool> >& edgeAssignmentIdVec,
        vector<supervoxel>& mapNodeId2Sv, string pathBasename,
        string imgFilenamePattern, vector<float>& costVec) const {
  int maxKnn = 1;

  // WE ASSUME THIS HAS BEEN DONE BEFORE HAND SOMEWHERE ELSE
  // calculate all stats for mapNodeId2Sv
  // for(size_t ii =0 ;ii < mapNodeId2Sv.size(); ii++)
  //	mapNodeId2Sv[ii].weightedGaussianStatistics<unsigned short int>(true);

  int iniTM = mapNodeId2Sv.front().TM;
  int endTM = mapNodeId2Sv.back().TM;
  int numTM = endTM - iniTM + 1;

  cout << "=========WARNING:cplexWorkspaceSpatioTemporalGraph::"
          "debugPrintLPSolutionForVisualizationMatlab TODO: GENERATE FOLDER TO "
          "WRITE OUT FILE FROM pathBasename!!!!======"
       << endl;
  ;

  // count number of supervoxels per time point
  unordered_map<int, int> numSvPerTM;
  numSvPerTM.rehash(ceil(numTM / numSvPerTM.max_load_factor()));
  for (vector<supervoxel>::iterator iterS = mapNodeId2Sv.begin();
       iterS != mapNodeId2Sv.end(); ++iterS) {
    if (numSvPerTM.find(iterS->TM) == numSvPerTM.end())  // not present
      numSvPerTM[iterS->TM] = 1;
    else
      numSvPerTM[iterS->TM]++;
  }

  // create folder
  char extra[256];
#if defined(_WIN32) || defined(_WIN64)
  SYSTEMTIME str_t;
  GetSystemTime(&str_t);
  sprintf(extra, "%s\\Sample_%d_%d_%d_%d_%d_%d", pathBasename.c_str(),
          str_t.wYear, str_t.wMonth, str_t.wDay, str_t.wHour, str_t.wMinute,
          str_t.wSecond);
#else
  sprintf(extra, "/Users/amatf/TrackingNuclei/tmp/GMEMtracking3D_%ld/",
          time(NULL));
#endif
  string filenameOutBasename(extra);

  string cmd("mkdir " + filenameOutBasename);
  int error = system(cmd.c_str());
  if (error > 0) {
    cout << "ERROR (" << error << "): generating path " << filenameOutBasename
         << endl;
    cout << "Wtih command " << cmd << endl;
    exit(error);
  }

  // write supervoxel binary file
  filenameOutBasename += "\\GMEMstructLearningTrainingSamples_frame";
  char buffer[256];
  int TMold = mapNodeId2Sv.begin()->TM;
  sprintf(buffer, "%.4d", TMold);
  string itoa(buffer);

  ofstream fout((filenameOutBasename + itoa + ".svb").c_str(),
                ios::binary | ios::out);
  if (fout.is_open() == false) {
    cout << "ERROR: opening file " << filenameOutBasename + itoa + ".svb"
         << endl;
    exit(0);
  }
  int aux = numSvPerTM[TMold];
  fout.write((char*)(&aux),
             sizeof(int));  // number of supervoxels in a given time point
  int count = 0;
  for (vector<supervoxel>::iterator iterS = mapNodeId2Sv.begin();
       iterS != mapNodeId2Sv.end(); ++iterS) {
    if (TMold != iterS->TM) {
      count = 0;  // to set Id with respect to time appropiately
      fout.close();
      TMold = iterS->TM;
      sprintf(buffer, "%.4d", TMold);
      itoa = string(buffer);
      fout.open(filenameOutBasename + itoa + ".svb", ios::binary | ios::out);
      aux = numSvPerTM[TMold];
      fout.write((char*)(&aux),
                 sizeof(int));  // number of supervoxels in a given time point
    }

    iterS->writeToBinary(fout);
    iterS->tempWildcard =
        (float)count;  // save svId per time point in tempWildcard
    count++;
  }
  fout.close();

  //-----------------------------------------------------------------
  // map parent and children id for the TGMM output
  vector<int> countTM(
      numTM, 0);  // to keep count of how many elements are for each time point
  vector<unordered_map<int, int> > mapNodeId(
      edgeAssignmentIdVec.size());  // from node index in mapNodeSv to node
                                    // index in time point for TGMM
  vector<unordered_map<int, int> > mapNodePar(edgeAssignmentIdVec.size());
  vector<unordered_map<int, int> > mapNodeCh1(edgeAssignmentIdVec.size());
  vector<unordered_map<int, int> > mapNodeCh2(edgeAssignmentIdVec.size());

  // find the mapping of all nodes
  size_t edgePos = 0;
  int parIdx, ch1Idx, ch2Idx, currentTM, offsetTM;
  vector<vector<supervoxel*> > svListTM(numTM);
  vector<vector<int> > svListNodeId(numTM);
  vector<vector<int> > svListNodeCost(numTM);
  vector<vector<int> > svListNodeSolN(numTM);  // lineage id
  int solN = 0;
  for (vector<vector<bool> >::const_iterator
           edgeAssignmentId = edgeAssignmentIdVec.begin();
       edgeAssignmentId != edgeAssignmentIdVec.end();
       ++edgeAssignmentId, solN++) {
    edgePos = 0;
    for (vector<bool>::const_iterator iterE = edgeAssignmentId->begin();
         iterE != edgeAssignmentId->end(); ++iterE, edgePos++) {
      if ((*iterE) == true)  // part of the solution
      {
        getEdgeIndex(edgePos, parIdx, ch1Idx, ch2Idx);

        if (ch1Idx >= 0 && ch1Idx < mapNodeId2Sv.size()) {
          currentTM = mapNodeId2Sv[parIdx].TM;
          offsetTM = currentTM - iniTM;
          if (mapNodeId[solN].find(parIdx) == mapNodeId[solN].end()) {
            mapNodeId[solN][parIdx] = countTM[offsetTM];
            countTM[offsetTM]++;
            svListTM[offsetTM].push_back(&(mapNodeId2Sv[parIdx]));
            svListNodeId[offsetTM].push_back(parIdx);
            svListNodeCost[offsetTM].push_back(costVec[solN]);
            svListNodeSolN[offsetTM].push_back(solN);
          }

          currentTM = mapNodeId2Sv[ch1Idx].TM;
          offsetTM = currentTM - iniTM;
          if (mapNodeId[solN].find(ch1Idx) == mapNodeId[solN].end()) {
            mapNodeId[solN][ch1Idx] = countTM[offsetTM];
            countTM[offsetTM]++;
            svListTM[offsetTM].push_back(&(mapNodeId2Sv[ch1Idx]));
            svListNodeId[offsetTM].push_back(ch1Idx);
            svListNodeCost[offsetTM].push_back(costVec[solN]);
            svListNodeSolN[offsetTM].push_back(solN);
          }
        }

        if (ch2Idx >= 0 && ch2Idx < mapNodeId2Sv.size()) {
          currentTM = mapNodeId2Sv[ch2Idx].TM;
          offsetTM = currentTM - iniTM;
          if (mapNodeId[solN].find(ch2Idx) == mapNodeId[solN].end()) {
            mapNodeId[solN][ch2Idx] = countTM[offsetTM];
            countTM[offsetTM]++;
            svListTM[offsetTM].push_back(&(mapNodeId2Sv[ch2Idx]));
            svListNodeId[offsetTM].push_back(ch2Idx);
            svListNodeCost[offsetTM].push_back(costVec[solN]);
            svListNodeSolN[offsetTM].push_back(solN);
          }
        }
      }
    }
  }
  // find out mapping of parent and children for each node
  solN = 0;
  for (vector<vector<bool> >::const_iterator
           edgeAssignmentId = edgeAssignmentIdVec.begin();
       edgeAssignmentId != edgeAssignmentIdVec.end();
       ++edgeAssignmentId, solN++) {
    edgePos = 0;
    for (vector<bool>::const_iterator iterE = edgeAssignmentId->begin();
         iterE != edgeAssignmentId->end(); ++iterE, edgePos++) {
      if ((*iterE) == true)  // part of the solution
      {
        getEdgeIndex(edgePos, parIdx, ch1Idx, ch2Idx);

        if (ch1Idx >= 0 && ch1Idx < mapNodeId2Sv.size()) {
          currentTM = mapNodeId2Sv[parIdx].TM;
          offsetTM = currentTM - iniTM;
          if (offsetTM == 0) mapNodePar[solN][parIdx] = -1;
          mapNodeCh1[solN][parIdx] = mapNodeId[solN][ch1Idx];

          mapNodePar[solN][ch1Idx] = mapNodeId[solN][parIdx];
        }

        if (ch2Idx >= 0 && ch2Idx < mapNodeId2Sv.size()) {
          mapNodeCh2[solN][parIdx] = mapNodeId[solN][ch2Idx];

          mapNodePar[solN][ch2Idx] = mapNodeId[solN][parIdx];
        }
      }
    }
  }
  //---------------------------------------------------------------------

  // save lineage
  int auxInt;
  float auxF;
  unsigned int auxUInt2[2];
  fout.open((filenameOutBasename + "_stackMCMC.bin").c_str(),
            ios::binary | ios::out);
  string auxText("blobstruct");
  fout.write(auxText.c_str(), auxText.length());
  fout.write((char*)(&numTM), sizeof(int));
  fout.write((char*)(&maxKnn), sizeof(int));
  fout.write((char*)(&dimsImage), sizeof(int));
  fout.write((char*)(supervoxel::getScale()), sizeof(float) * dimsImage);

  for (size_t ii = 0; ii < svListTM.size(); ii++) {
    string imgFilename = parseImagePath(
        imgFilenamePattern, ii + iniTM);  // TODO: get this figure out
    sprintf(buffer, "%.4d", ii + iniTM);
    itoa = string(buffer);
    string svFilename(filenameOutBasename + itoa + ".svb");
    fout.write(imgFilename.c_str(), imgFilename.length());
    auxText = string("*");
    fout.write(auxText.c_str(), auxText.length());

    fout.write(svFilename.c_str(), svFilename.length());
    fout.write(auxText.c_str(), auxText.length());

    auxInt = svListTM[ii].size();  // number of objects
    fout.write((char*)(&auxInt), sizeof(int));
    auxInt = ii;
    fout.write((char*)(&auxInt), sizeof(int));

    for (size_t aa = 0; aa < svListTM[ii].size(); aa++) {
      solN = svListNodeSolN[ii][aa];

      auxInt = aa;
      fout.write((char*)(&(auxInt)), sizeof(int));
      fout.write((char*)(svListTM[ii][aa]->centroid),
                 dimsImage * sizeof(float));
      auxF = 10.0f;
      fout.write((char*)(&(auxF)), sizeof(float));  // alpha

      // TODO: calculate KNN
      auxInt = ii;
      fout.write((char*)(&ii), sizeof(int));
      fout.write((char*)(&(auxInt)), sizeof(int));

      auxInt = 1;  // one supervoxel per blob in this case
      fout.write((char*)(&(auxInt)), sizeof(int));
      auxInt = (int)(svListTM[ii][aa]->tempWildcard);
      fout.write((char*)(&(auxInt)), sizeof(int));

      auxInt = dimsImage + (dimsImage * (1 + dimsImage) / 2);
      fout.write((char*)(&(auxInt)), sizeof(int));

      fout.write((char*)(svListTM[ii][aa]->precisionW),
                 (dimsImage * (1 + dimsImage) / 2) * sizeof(float));
      fout.write((char*)(svListTM[ii][aa]->centroid),
                 dimsImage * sizeof(float));

      int ch1 = -1, ch2 = -1;
      int numCh = 0;
      if (mapNodeCh1[solN].find(svListNodeId[ii][aa]) !=
          mapNodeCh1[solN].end()) {
        ch1 = mapNodeCh1[solN][svListNodeId[ii][aa]];
        numCh++;
      }
      if (mapNodeCh2[solN].find(svListNodeId[ii][aa]) !=
          mapNodeCh2[solN].end()) {
        ch2 = mapNodeCh2[solN][svListNodeId[ii][aa]];
        numCh++;
      }

      float cost = svListNodeCost[ii][aa];
      if (numCh == 0 || ii == svListTM.size() - 1)  // no children
      {
        if (ii == 0 ||
            mapNodePar[solN].find(svListNodeId[ii][aa]) ==
                mapNodePar[solN].end())  // no parent
        {
          fout.write((char*)(&cost), sizeof(float));
          auxInt = aa;
          fout.write((char*)(&auxInt), sizeof(int));
          auxUInt2[0] = 4294967295;
          auxUInt2[1] = 0;
          fout.write((char*)(&auxUInt2), sizeof(unsigned int) * 2);
          auxInt = 0;
          fout.write((char*)(&auxInt), sizeof(int));
        } else {
          fout.write((char*)(&cost), sizeof(float));
          auxInt = aa;
          fout.write((char*)(&auxInt), sizeof(int));
          auxUInt2[0] = ii - 1;
          auxUInt2[1] = mapNodePar[solN][svListNodeId[ii][aa]];
          fout.write((char*)(&auxUInt2), sizeof(unsigned int) * 2);
          auxInt = 0;
          fout.write((char*)(&auxInt), sizeof(int));
        }
      } else {  // chidlren found
        if (ii == 0 ||
            mapNodePar[solN].find(svListNodeId[ii][aa]) ==
                mapNodePar[solN].end())  // no parent
        {
          fout.write((char*)(&cost), sizeof(float));
          auxInt = aa;
          fout.write((char*)(&auxInt), sizeof(int));
          auxUInt2[0] = 4294967295;
          auxUInt2[1] = 0;
          fout.write((char*)(&auxUInt2), sizeof(unsigned int) * 2);
          fout.write((char*)(&numCh), sizeof(int));
          if (ch1 >= 0) {
            auxUInt2[0] = ii + 1;
            auxUInt2[1] = mapNodeCh1[solN][svListNodeId[ii][aa]];
            fout.write((char*)(&auxUInt2), sizeof(unsigned int) * 2);
          }
          if (ch2 >= 0) {
            auxUInt2[0] = ii + 1;
            auxUInt2[1] = mapNodeCh2[solN][svListNodeId[ii][aa]];
            fout.write((char*)(&auxUInt2), sizeof(unsigned int) * 2);
          }
        } else {  // parent found
          fout.write((char*)(&cost), sizeof(float));
          auxInt = aa;
          fout.write((char*)(&auxInt), sizeof(int));
          auxUInt2[0] = ii - 1;
          auxUInt2[1] = mapNodePar[solN][svListNodeId[ii][aa]];
          fout.write((char*)(&auxUInt2), sizeof(unsigned int) * 2);
          fout.write((char*)(&numCh), sizeof(int));
          if (ch1 >= 0) {
            auxUInt2[0] = ii + 1;
            auxUInt2[1] = mapNodeCh1[solN][svListNodeId[ii][aa]];
            fout.write((char*)(&auxUInt2), sizeof(unsigned int) * 2);
          }
          if (ch2 >= 0) {
            auxUInt2[0] = ii + 1;
            auxUInt2[1] = mapNodeCh2[solN][svListNodeId[ii][aa]];
            fout.write((char*)(&auxUInt2), sizeof(unsigned int) * 2);
          }
        }
      }
    }
  }

  fout.close();
}

//=====================================================================
//=======================================================================

string cplexWorkspaceSpatioTemporalGraph::parseImagePath(
    const string& imgRawPath_, int frame) {
  string imgRawPath(imgRawPath_);
  size_t found = imgRawPath.find_first_of("?");
  while (found != string::npos) {
    int intPrecision = 0;
    while ((imgRawPath[found] == '?') && found != string::npos) {
      intPrecision++;
      found++;
      if (found >= imgRawPath.size()) break;
    }

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

    found = imgRawPath.find_first_of("?");
    imgRawPath.replace(found, intPrecision, itoaTM);

    // find next ???
    found = imgRawPath.find_first_of("?");
  }

  return imgRawPath;
}
