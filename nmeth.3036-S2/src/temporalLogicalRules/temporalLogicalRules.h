/*
 * See license.txt for full license and copyright notice.
 *
 * \brief Implements methods to inforce temporal logical rules in a cell lineage
 *
 */


#ifndef __TEMPORAL_LOGICAL_RULES_H__
#define __TEMPORAL_LOGICAL_RULES_H__

#include <iostream>
#include "GaussianMixtureModel_Redux.h"
#include "lineageHyperTree.h"

using namespace std;


int mainTestTemporalLogicalRules( int argc, const char** argv );

//functions to parse data from GMM pipeline
int parseGMMtrackingFilesToHyperTree(string imgPrefix, string imgSuffix, string basenameXML,int iniFrame,int endFrame,int tau, lineageHyperTree &lht);
void getImgPath(string imgPrefix, string imgSuffix, int tau, string& imgPath, string& imgLpath, int intPrecision);
void getImgPath2(string imgPrefix, string imgSuffix, string imgSuffix2, int tau, string& imgPath, string& imgLpath, int intPrecision);

//debugging functions
void testListIteratorProperties();

#endif
