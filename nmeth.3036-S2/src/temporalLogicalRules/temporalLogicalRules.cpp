/*
 * See license.txt for full license and copyright notice.
 *
 * \brief Implements methods to inforce temporal logical rules in a cell lineage
 *
 */

#if defined(_WIN32) || defined(_WIN64)
#define NOMINMAX
#include <Psapi.h>
#include <Windows.h>
#endif

#include <stdio.h>
#include <time.h>
#include <fstream>
#include <limits>
#include <list>
#include <string>
#include "binaryTree.h"
#include "lineageWindowFeatures.h"
#include "nuclei.h"  //contains the class to use for each node. You can change thsi depending on the application and teh attributes you want to keep for each in the tree
#include "temporalLogicalRules.h"
#include "trackletCalculation.h"

//==========================================================
int mainTestTemporalLogicalRules(int argc, const char** argv) {
  // raw data (to generate annotation data)
  string imgRawPrefix("G:/12-07-17/TimeFused/Dme_E1_SpiderGFP-His2ARFP.TM");
  string imgRawSuffix("_timeFused_blending/SPC0_TM");
  string imgRawSuffix2("_CM0_CM1_CHN00_CHN01.fusedStack");

  // local subtracted data
  string imgPrefix(
      "G:/12-07-17/TimeFused_BackgrSubtraction_thrPctile40_maxSize3000_otzu/"
      "TM");
  string imgSuffix("/CM0_CM1_CHN00_CHN01.fusedStack_bckgSub_");
  string pathTGMMresult(
      "G:/TGMMrunsArchive/"
      "GMEMtracking3D_2012_11_14_16_27_42_dataset_12_07_17_drosophila_TM0_300_"
      "TPini_gastrulation_localBckgSubtOtzu/XML_finalResult");
  int iniFrame = 20;  // 20;
  int endFrame = 32;  // 32;
  int tau = 14;
  unsigned int KmaxNumNN = 10;
  float KmaxDistKNN = 50.0f;
  int devCUDA = 0;
  lineageHyperTree lht(endFrame + 1);

  int err = parseGMMtrackingFilesToHyperTree(
      imgPrefix, imgSuffix, pathTGMMresult, iniFrame, endFrame, tau, lht);
  if (err > 0) return err;

  //-----------------------------------------
  /*
  PROCESS_MEMORY_COUNTERS pmc;
  Sleep(1000);
GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
  cout<<"Memory used before "<<pmc.WorkingSetSize<<endl;
  list< lineage >::iterator iter = lht.lineagesList.begin();
  iter++;
  for(; iter != lht.lineagesList.end();)
  {
          lht.deleteLineage(iter);
  }
  lht.debugPrintLineageForLocalLineageDisplayinMatlab("_debugLeakage");
  Sleep(1000);
  GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
  cout<<"Memory used after  "<<pmc.WorkingSetSize<<endl;
  return 0;
  */
  //----------------------------------------------------------

  cout << "Calculating nearest neighbors" << endl;
  time_t start, end;
  time(&start);
  for (int ii = iniFrame; ii <= endFrame; ii++) {
    err = lht.supervoxelNearestNeighborsInTimeForward(ii, KmaxNumNN,
                                                      KmaxDistKNN, devCUDA);
    if (err > 0) return err;
    err = lht.supervoxelNearestNeighborsInTimeBackward(ii, KmaxNumNN,
                                                       KmaxDistKNN, devCUDA);
    if (err > 0) return err;
    int err = lht.supervoxelNearestNeighborsInSpace(ii, KmaxNumNN, KmaxDistKNN,
                                                    devCUDA);
    if (err > 0) return err;
  }

  //-------------------------------testing lineage
  // features------------------------------
  {
    cout << "Testing lineage features" << endl;
    time_t start, t_end;
    int numTest = min(1000, (int)(lht.lineagesList.size()));
    int winLength = 10;

    list<lineage>::iterator iterL = lht.lineagesList.begin();
    float fVec[lineageWindowNumFeatures];
    time(&start);
    for (int ii = 0; ii < numTest; ii++, ++iterL) {
      getLineageWindowFeatures<unsigned short int>(iterL->bt.pointer_mainRoot(),
                                                   winLength, fVec);
    }
    time(&end);

    cout << "It took " << difftime(end, start) << " secs for " << numTest
         << " lineages" << endl;
    return 0;
  }
  //------------------------------------------------------------------------------------

  //------------------------------------------------------------------------
  /*
  list<supervoxel>::iterator iterF= lht.supervoxelsList[25].begin();
  for(int ii =0;ii<4526;ii++) iterF++;
  cout<<"Neighbors of nuclei : "<<*iterF<<endl;;
  for(vector< SibilingTypeSupervoxel >::iterator iter =
  iterF->nearestNeighborsInTimeForward.begin(); iter !=
  iterF->nearestNeighborsInTimeForward.end(); ++iter)
  {
          cout<<"supervoxel:"<<
  (*(*iter))<<";distance="<<iterF->Euclidean2Distance(*(*iter))<<endl;
  }
  list<supervoxel>::iterator iterF2= lht.supervoxelsList[26].begin();
  for(int ii =0;ii<4546;ii++) iterF2++;
  cout<<"======================================="<<endl;
  cout<<"supervoxel:"<<
  ((*iterF2))<<";distance="<<iterF->Euclidean2Distance(*(iterF2))<<endl;

  return 0;
  */
  //-----------------------------------------------------------------------

  //-------------------------------------------------------------------------
  /*
  lineageHyperTree lhtSub(endFrame-iniFrame+1);//copy of sublineage for all time
  points left
  vector<rootSublineage> vecRoot;
  vecRoot.push_back(lht.lineagesList.front().bt.pointer_mainRoot());
  vecRoot.push_back(lht.lineagesList.back().bt.pointer_mainRoot()->left);
  lht.cutSublineage(vecRoot , lhtSub);
  err = lhtSub.debugCheckHierachicalTreeConsistency();
  if(err > 0)
          return err;
  lhtSub.debugPrintLineage(0);
  lhtSub.debugPrintLineage(1);

  //modified something in lineage1 and check that it is independent from
  original lineage
  lhtSub.nucleiList[0].front().centroid[0] = 300.001;
  cout<< *(lhtSub.lineagesList.front().bt.pointer_mainRoot()->data) <<endl;
  cout<< *(lht.lineagesList.front().bt.pointer_mainRoot()->data) <<endl;

  //paste sublineage
  cout<<"====================pasting
  sublineage============================="<<endl;
  lhtSub.pasteOpenEndSublineage(lht);
  lht.debugPrintLineage(0);
  err = lht.debugCheckHierachicalTreeConsistency();
  if(err >0) return err;
  cout<< *(lht.lineagesList.front().bt.pointer_mainRoot()->data) <<endl;
  */
  //-------------------------------------------------------------------------

  //-------------------------------------------------------------------------
  /*
  cout<<"Extracting cell division / death sublineages ..."<<endl;
  unsigned int winRadiusTime = 5;
  vector< lineageHyperTree > lhtVec;
  lht.cutSublineageCellDeathDivisionEvents(winRadiusTime,endFrame-winRadiusTime,
  lhtVec);

  cout<<"Extracted "<<lhtVec.size()<<" critical sublineages"<<endl;
  //for(size_t ii = 0;ii < lhtVec[5].lineagesList.size();ii++)
  //	lhtVec[5].debugPrintLineage(ii);

  cout<<"Printing all the sublineages to visualize with Matlab"<<endl;
  for(size_t ii =0 ; ii<lhtVec.size(); ii++)
          lhtVec[ii].debugPrintLineageForLocalLineageDisplayinMatlab();

  //release images (stored in supervoxel)
  lht.releaseData();
  */
  //--------------------------------------------------------------------------

  //-------------------------------------------------------------------------
  //--------------------statistics------------------------------------------
  /*
  //length before death after split
  string
  foutFilename("E:/temp/temporalLogicalRulesStats/daughterLengthToNearestNeighborDivision.txt");
  ofstream fout(foutFilename.c_str());
  if(fout.is_open() == false)
  {
          cout<<"ERROR: file "<<foutFilename<<" could not be openend"<<endl;
          return 3;
  }
  vector<int> ll;
  for(list<lineage>::const_iterator iter = lht.lineagesList.begin(); iter !=
  lht.lineagesList.end(); ++iter)
  {
          iter->daughterLengthToDivisionAll(ll);
          //lht.daughterLengthToNearestNeighborDivisionAll(ll,iter->bt.pointer_mainRoot());
          for(vector<int>::const_iterator iterV = ll.begin(); iterV != ll.end();
  ++iterV)
                  fout<<(*iterV)<<",";
  }
  fout.close();
  return 0;
  */
  //-------------------------------------------------------------------------

  //========================================================================
  //-----------------------test corrections---------------------------------

  //-----------------short lived daughter events--------------------
  /*
          int winRadiusTime = 5;
          unsigned int TM = endFrame - winRadiusTime;
          vector< lineageHyperTree > lhtVec;
          cout<<"Extracting cell division / death sublineages to perform
     corrections at time point " << TM << endl;
          lht.cutSublineageCellDeathDivisionEvents(winRadiusTime,TM, lhtVec);

          cout<<"Merging shorted live daughter"<<endl;
          uint64 ll;
          int countL = 0;
          int countFixed = 0;
          int bogus;
          for(vector< lineageHyperTree >::iterator iter = lhtVec.begin(); iter
     != lhtVec.end(); ++iter)
          {
                  char buffer[16];
                  sprintf(buffer,"%.5d",countL);
                  string itoa(buffer);
                  ll = iter->debugNumberOfTotalNuclei();//to know if there has
     been any changes
                  //printout lineage before changes to check in Matlab
                  iter->debugPrintLineageForLocalLineageDisplayinMatlab("_lineage"
     + itoa);
                  //check for merges across all the sublineages
                  iter->mergeShortLivedDaughtersAll(winRadiusTime - 1,
     2147483647, bogus, bogus);
                  if( ll != iter->debugNumberOfTotalNuclei())//print again if it
     has changed
                  {
                          //check that lineage is still coherent
                          if(iter->debugCheckHierachicalTreeConsistency() > 0)
     return 3;
                          iter->debugPrintLineageForLocalLineageDisplayinMatlab("_lineage"
     + itoa);
                          countFixed++;
                  }

                  countL++;
          }
          cout<<countFixed<<" out of "<<lhtVec.size()<<" ("<<100.0f * ( (float)
     countFixed/((float)lhtVec.size()) )<<"%) cell division deaths/splits we
     fixed"<<endl;
          */
  //-----------------dead cells-------------------------
  int dummy;
  lht.mergeShortLivedDaughtersAll(3, 2147483647, dummy,
                                  dummy);  // first clean easy ones
  int winRadiusTime = 5;

  int KNN = 4;  // number of nearest neighbors to check for options
  const int numChildren =
      2;  // set to 0 to look for cell deaths and set it to 2 for cell divisions

  float KNNdistRatio = 1.5 * 1.5;  // maximum distance ratio between nearest
                                   // neighbors in order to consider it as an
                                   // option (power of 2 because we avoid sqrt)
  // float dist;
  // list<nucleus>::iterator iterN_NN;
  vector<ChildrenTypeLineage> iterN_NNvec(KNN);
  vector<float> distVec(KNN);
  dummy = 0;

  string prefixAction;
  if (numChildren == 0)
    prefixAction = string("deadCells");
  else if (numChildren == 2)
    prefixAction = string("dividingCells");

  // set the imagePath and the imageLPath
  string imgPath, imgLpath;
  getImgPath(imgPrefix, imgSuffix, tau, imgPath, imgLpath, 5);

  string imgRawPath, imgRawLpath;
  getImgPath2(imgRawPrefix, imgRawSuffix, imgRawSuffix2, tau, imgRawPath,
              imgRawLpath, 4);

  for (int frame = iniFrame + winRadiusTime; frame < lht.getMaxTM() - 1;
       frame++)  // we do not want to check all the deths in the last frame
                 // (basically all cells)
  {
    for (list<nucleus>::iterator iterN = lht.nucleiList[frame].begin();
         iterN != lht.nucleiList[frame].end(); ++iterN) {
      if (iterN->treeNodePtr->getNumChildren() ==
          numChildren)  // nucleus dies or divides
      {
        cout << "Calculating sublineage for  " << prefixAction << " " << dummy
             << endl;

        int errKNN =
            lht.findKNearestNucleiNeighborInTimeForwardSupervoxelEuclideanL2(
                iterN, iterN_NNvec, distVec);
        if (errKNN > 0) return errKNN;

        lineageHyperTree lhtSub(2 * winRadiusTime +
                                1);  // length of the sublineage
        // find root to cut sublineage
        rootSublineage aux = iterN->treeNodePtr;
        int count = winRadiusTime;
        while (count >= 0 && aux != NULL) {
          if (aux->parent == NULL) break;
          aux = aux->parent;
          count--;
        }
        vector<rootSublineage> vecRoot;
        vecRoot.push_back(aux);
        for (unsigned int ii = 0; ii < distVec.size(); ii++) {
          if (distVec[ii] < 1e30f &&
              distVec[ii] < KNNdistRatio * distVec[0])  // it has a neighbor and
                                                        // it is close enough
                                                        // with respect to
                                                        // nearest neighbor
          {
            aux = iterN_NNvec[ii]->treeNodePtr;
            count = winRadiusTime;
            while (count >= 0 && aux != NULL) {
              if (aux->parent == NULL) break;
              aux = aux->parent;
              count--;
            }
            vecRoot.push_back(aux);
          }
        }
        lht.cutSublineage(vecRoot, lhtSub);
        char bufferD[16];
        sprintf(bufferD, "%s_TM%d_%d_N%.5d_KNN%d", prefixAction.c_str(),
                iniFrame, endFrame, dummy, KNN);
        string itoaD(bufferD);
        lhtSub.debugPrintLineageForLocalLineageDisplayinMatlab(
            imgPath, imgLpath, itoaD + "_KNNforward", imgRawPath);

        //----------------------------------------------------------------
        /*
        cout<<"DEBUGGING neighboring supervoxels: return at the end"<<endl;
        int64 boundarySize[dimsImage];
        int conn3D = 4;
        int64* neighOffset = supervoxel::buildNeighboorhoodConnectivity(conn3D,
        boundarySize);
        //lhtSub.mergeNonSeparatingDaughters(iterN->treeNodePtr,conn3D,
        neighOffset);
        lhtSub.mergeNonSeparatingDaughters(iterN->treeNodePtr,conn3D,
        neighOffset,10);
        delete[] neighOffset;
        exit(3)	;
        */
        //---------------------------------------------------------------

        // calculate tracklets
        int errT =
            calculateTrackletsWithSparseHungarianAlgorithm(lhtSub, 0, 0.9, 8);
        if (errT > 0) return errT;
        errT = lhtSub.debugCheckHierachicalTreeConsistency();
        if (errT > 0) return errT;
        lhtSub.debugPrintLineageForLocalLineageDisplayinMatlab(
            imgPath, imgLpath, itoaD + "_KNNforward" + "_tracklets",
            imgRawPath);

        dummy++;
      }
    }
  }

  //-----------------dead cells with nearest neighbor
  // division-------------------------
  /*
  int winRadiusTime = 5;
  int lengthThrSplitDeathDivisionPattern = 2;
  int dumb;
  lht.mergeShortLivedDaughtersAll(5, 2147483647,dumb,dumb);//first correct
  simple deaths (otherwise there is too much overlap)
  float dist;
  list<nucleus>::iterator iterN_NN;
  int countL = 0;
  int countP = 0;
  for(int frame = 0; frame < lht.getMaxTM()-1; frame++)//we do not want to check
  all the deths in the last frame (basically all cells)
  {
          for(list<nucleus>::iterator iterN = lht.nucleiList[frame].begin();
  iterN != lht.nucleiList[frame].end();  ++iterN)
          {
                  if( iterN->treeNodePtr->getNumChildren() == 0)//nucleus dies
                  {
                          int dd =
  lht.splitDeathDivisionPattern(iterN->treeNodePtr,
  lengthThrSplitDeathDivisionPattern);
                          if( dd > 0)//pattern is satisfied
                          {
                                  char buffer[16];
                                  sprintf(buffer,"%.5d",countP);
                                  string itoa(buffer);
                                  lineageHyperTree lhtSub(2 * winRadiusTime +
  1);//length of the sublineage
                                  //find root to cut sublineage
                                  rootSublineage aux = iterN->treeNodePtr;
                                  int count = winRadiusTime;
                                  while(count >= 0 && aux!=NULL)
                                  {
                                          if( aux->parent == NULL) break;
                                          aux = aux->parent;
                                          count--;
                                  }
                                  vector< rootSublineage > vecRoot;
                                  vecRoot.push_back(aux);
                                  dist =
  lht.findNearestNucleusNeighborInSpaceEuclideanL2(iterN, iterN_NN);
                                  aux = iterN_NN->treeNodePtr;
                                  count = winRadiusTime;
                                  while(count >= 0 && aux!=NULL)
                                  {
                                          if( aux->parent == NULL) break;
                                          aux = aux->parent;
                                          count--;
                                  }
                                  vecRoot.push_back(aux);
                                  lht.cutSublineage(vecRoot, lhtSub);
                                  //lhtSub.debugPrintLineageForLocalLineageDisplayinMatlab("_lineage"
  + itoa);
                                  countP++;
                          }else{
                                  //cout<<"Nearest neighbor distance
  "<<dist<<endl;
                          }
                          countL++;
                  }
          }
  }
  cout<<countP<<" out of "<<countL<<" ("<<100.0f * ( (float)
  countP/((float)countL) )<<"%) division after death were fixed"<<endl;
  */
  //=======================================================================

  //======================================================================
  //----------------------perform corrections----------------------------
  /*
  const int lengthThrDaughterShortLived = 5;
  int aux1, aux2;
  lht.mergeShortLivedDaughtersAll(lengthThrDaughterShortLived, 2147483647,
  aux1,aux2);
  cout<<aux1<<" out of "<<aux2<<" ("<<100.0f * ( (float) aux1/((float)aux2)
  )<<"%) cell divisions were fixed"<<endl;



  lht.debugCheckHierachicalTreeConsistency();
  */
  //======================================================================

  return 0;
}

//===========================================================
int parseGMMtrackingFilesToHyperTree(string imgPrefix, string imgSuffix,
                                     string pathTGMMresult, int iniFrame,
                                     int endFrame, int tau,
                                     lineageHyperTree& lht) {
  // clear hyper tree
  lht.clear();

  vector<GaussianMixtureModelRedux*> vecGM;
  vecGM.reserve(30000);  // avoid too much realloc
  for (int frame = iniFrame; frame <= endFrame; frame++) {
    cout << "DEBUGGING: at parseGMMtrackingFilesToHyperTree: starting frame "
         << frame << endl;
    char itoaB[16];
    sprintf(itoaB, "%.4d", frame);
    string itoa4(itoaB);
    char itoaI[16];
    sprintf(itoaI, "%.5d", frame);
    string itoa5(itoaI);
    char itoaTauB[16];
    sprintf(itoaTauB, "%d", tau);
    string itoaTau(itoaTauB);

    // read supervoxels
    string filenameImg(imgPrefix + itoa5 + imgSuffix + itoa5 + ".tif");
    string filenameSupervoxels(imgPrefix + itoa5 + imgSuffix +
                               "PersistanceSeg_tau" + itoaTau + "_" + itoa5 +
                               ".tif");

    int err = lht.readListSupervoxelsFromTifWithWeightedCentroid(
        filenameImg, filenameSupervoxels, frame);
    if (err > 0) return err;

    // read which nuclei belongs to each supervoxel (rnk file)
    err = lht.readBinaryRnkTGMMfile(
        pathTGMMresult + "/rnk_frame" + itoa4 + ".bin", frame);
    if (err > 0) return err;

    // read new frame
    err = readGaussianMixtureModelXMLfile(
        (pathTGMMresult + "/GMEMfinalResult_frame" + itoa4 + ".xml").c_str(),
        vecGM);
    if (err > 0) return err;
    err = lht.parseTGMMframeResult(vecGM, frame);
    if (err > 0) return err;

    // delete any possible content
    for (unsigned int ii = 0; ii < vecGM.size(); ii++) delete (vecGM[ii]);
    vecGM.clear();
  }

  return 0;
}

void testListIteratorProperties() {
  cout << "DEBUGGING: at testListIteratorProperties()" << endl;

  int N = 10;

  list<int> ll, ll2;
  for (int ii = 0; ii < N; ii++) {
    ll.push_back(ii);
    ll2.push_back(ii + N);
  }
  vector<list<int>::iterator> vecIter1, vecIter10, vecIter2;

  for (list<int>::iterator iter = ll.begin(); iter != ll.end(); ++iter) {
    vecIter1.push_back(iter);
    vecIter10.push_back(iter);
  }
  for (list<int>::iterator iter = ll2.begin(); iter != ll2.end(); ++iter) {
    vecIter2.push_back(iter);
  }

  for (int ii = 0; ii < N; ii++) {
    // cout<<ii<<" "<< (vecIter1[ii] == vecIter10[ii]) <<" "<< (vecIter1[ii] ==
    // vecIter2[ii]) <<" " << (*(vecIter1[ii]))<<endl;//THIS GIVES A RUN-TIME
    // ERROR WHEN TRYING TO COMPARE ITERATORS FROM DIFFERENT LISTS
    cout << ii << " " << (vecIter1[ii] == vecIter10[ii]) << " "
         << (*(vecIter1[ii])) << endl;
  }

  ll.erase(vecIter1[3]);
  vecIter1.erase(vecIter1.begin() + 3);
  int count = 0;
  for (list<int>::iterator iter = ll.begin(); iter != ll.end(); ++iter) {
    cout << count << " " << (vecIter1[count] == iter) << " " << (*iter) << " "
         << (*(vecIter1[count])) << endl;
    count++;
  }

  cout << "erase one element inside the for loop" << endl;
  count = 0;
  for (list<int>::iterator iter = ll2.begin(); iter != ll2.end(); ++iter) {
    if (count == 7) ll2.erase(iter--);
    cout << count << " " << (*iter) << endl;
    count++;
  }
}

void getImgPath(string imgPrefix, string imgSuffix, int tau, string& imgPath,
                string& imgLpath, int intPrecision) {
  // char itoaI[16];
  // sprintf(itoaI,"%.5d",frame);
  // string itoa5(itoaI);
  string itoa5("?????");
  switch (intPrecision) {
    case 1:
      itoa5 = string("?");
      break;
    case 2:
      itoa5 = string("??");
      break;
    case 3:
      itoa5 = string("???");
      break;
    case 4:
      itoa5 = string("????");
      break;
    case 5:
      itoa5 = string("?????");
      break;
    case 6:
      itoa5 = string("??????");
      break;
  }

  char itoaTauB[16];
  sprintf(itoaTauB, "%d", tau);
  string itoaTau(itoaTauB);

  // read supervoxels
  imgPath = string(imgPrefix + itoa5 + imgSuffix + itoa5 + ".tif");
  imgLpath = string(imgPrefix + itoa5 + imgSuffix + "PersistanceSeg_tau" +
                    itoaTau + "_" + itoa5 + ".tif");
}

void getImgPath2(string imgPrefix, string imgSuffix, string imgSuffix2, int tau,
                 string& imgPath, string& imgLpath, int intPrecision) {
  // char itoaI[16];
  // sprintf(itoaI,"%.5d",frame);
  // string itoa5(itoaI);
  string itoa5("?????");
  switch (intPrecision) {
    case 1:
      itoa5 = string("?");
      break;
    case 2:
      itoa5 = string("??");
      break;
    case 3:
      itoa5 = string("???");
      break;
    case 4:
      itoa5 = string("????");
      break;
    case 5:
      itoa5 = string("?????");
      break;
    case 6:
      itoa5 = string("??????");
      break;
  }

  char itoaTauB[16];
  sprintf(itoaTauB, "%d", tau);
  string itoaTau(itoaTauB);

  // read supervoxels
  imgPath = string(imgPrefix + itoa5 + imgSuffix + itoa5 + imgSuffix2 + ".tif");
  imgLpath = string(imgPrefix + itoa5 + imgSuffix + itoa5 + imgSuffix2 +
                    "_PersistanceSeg_tau" + itoaTau + ".tif");
}
