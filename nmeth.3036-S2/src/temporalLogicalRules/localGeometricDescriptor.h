/*
 * Copyright (C) 2011-2013 by  Fernando Amat
 * See license.txt for full license and copyright notice.
 *
 * Authors: Fernando Amat
 *  localGeometricDescriptor.h
 *
 *  Created on: July 21st, 2013
 *      Author: Fernando Amat
 *
 * \brief Local geometric descriptors for point clouds
 *
 *
 */
#ifndef __LOCAL_GEOMETRIC_DESCRIPTOR_H__
#define __LOCAL_GEOMETRIC_DESCRIPTOR_H__

#include <iostream>
#include <vector>

using namespace std;

static const int maxTM_LGD =
    3000;  // maximum number of time points to store ref points

// D: dimensionality of the local descriptor
template <int D>
class localGeometricDescriptor {
 public:
  // constructor / desctructor
  localGeometricDescriptor();

  // set / get functions
  bool empty() { return neighPts[0].empty(); };
  size_t size() { return neighPts[0].size(); };
  static void setNeighRadius(float p) { neighRadius = p; };
  static float getNeighRadius() { return neighRadius; };
  void reserveNeighPts(int n) {
    for (int ii = 0; ii < D; ii++) neighPts[ii].reserve(n);
  };
  void setNeighPts(const float p0[D], const std::vector<float> neighPts_[D]);
  void addNeighPts(const float p0[D], const float p[D]) {
    for (int ii = 0; ii < D; ii++) neighPts[ii].push_back((p[ii] - p0[ii]));
  }
  static void addRefPt(int TM, const float p[D]) {
    if (TM >= maxTM_LGD || TM < 0) {
      cout << "ERROR: localGeometricDescriptor::addRefPt: TM = " << TM
           << "exceeded maximum number of time points " << maxTM_LGD
           << ". Change parameters and recompile code" << endl;
    }
    for (int ii = 0; ii < D; ii++) refPts[TM][ii].push_back(p[ii]);
  };
  static std::vector<float>* getRefPts(int TM) {
    return refPts[TM];
  };  // returns a vector<float>[D] element (the same as as neighPts)

  /*
  \brief main function. Calculates the distance between two templates

  \param in k:	number of elements to compare (to make it robust)
  */
  float distance(const localGeometricDescriptor& p, size_t k,
                 float scale[D]) const;

 protected:
 private:
  // variables
  std::vector<float> neighPts[D];  // stores neighboring points in local
                                   // coordinates. If D = 3 -> X = neighPts[0],
                                   // Y = neighPts[1], Z = neighPts[2]
  static float neighRadius;  // radius of the ball to look around the point in
                             // order to find neighbors
  static std::vector<float> refPts[maxTM_LGD][D];  // for each time point we
                                                   // have a set of reference
                                                   // points (or landmarks) that
                                                   // we will use to construct
                                                   // local descriptors
};

#endif  //__LOCAL_GEOMETRIC_DESCRIPTOR_H__
