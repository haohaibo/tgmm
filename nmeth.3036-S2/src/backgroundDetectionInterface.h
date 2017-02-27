/*
 * Copyright (C) 2011-2013 by  Fernando Amat
 * See license.txt for full license and copyright notice.
 *
 * Authors: Fernando Amat 
 *  supportFunctionsMain.h
 *
 *  Created on: January 21st, 2012
 *      Author: Fernando Amat
 *
 * \brief Contains different routines that are called form the main.cpp part of theprogram to not make the code so cluttr
 *        
 *
 */
#ifndef __BACKGROUND_DETECTION_INTERFACE_H__
#define __BACKGROUND_DETECTION_INTERFACE_H__

#include <string>
#include "lineageHyperTree.h"


//---------------------------------------------------------------------------------
//-----------------------------implementing background detection------------------
/*
\brief uses a boosting classifier to determine incoherent tracks that might belong to background. It stores the results at lht.nucleiList[TM][i].probBackground
*/
int setProbBackgroundTracksAtTM(lineageHyperTree &lht, int TM, int temporalWindowSizeForBackground, string &classifierFilename, int devCUDA);

//--------------------------------------------------------------------------------


/*
\brief uses background detector score to remove background elements. If a single branch has an element higher than thrBackground -> the whole branch is remove->we need a complete forward-backward pass
*/

int applyProbBackgroundMinMaxRulePerBranch(const string &basenameTGMMresult, int iniFrame, int endFrame, const string& outputFolder, float thrBackground);


int applyProbBackgroundAvgRulePerBranch(const string &basenameTGMMresult, int iniFrame, int endFrame, const string& outputFolder, float thrBackground);

int applyProbBackgroundHysteresisRulePerBranch(const string &basenameTGMMresult, int iniFrame, int endFrame, const string& outputFolder, float thrLow, float thrHigh);//two thresholds for hysteresis

//=============================================================================
//--------------------------debugging------------------------------


#endif //__BACKGROUND_DETECTION_INTERFACE_H__