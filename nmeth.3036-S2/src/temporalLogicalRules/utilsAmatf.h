/*
* See license.txt for full license and copyright notice.
*
*  utils.h
*
* \brief Miscellanea of simple functions to perform different tasks
*
*/

#ifndef __UTILS_AMATF_H__
#define __UTILS_AMATF_H__

#include <math.h>
#include <limits>  // std::numeric_limits
#include <vector>
#include "../constants.h"

using namespace std;

static const double utilsAmatf_PI_ = 3.14159265358979311600;
static const double utilsAmatf_SQRT3_ = 1.73205080756887719318;

//=======================================================================================================
//===========================================================================
// eigen value functions: I need to have everythign in one file

//=============================================================================================
//=========================================================================
// determinant for 3x3 symmetric matrix
template <class T>
inline T utilsAmatf_determinantSymmetricW_3D(const T *W_k) {
  return W_k[0] * (W_k[3] * W_k[5] - W_k[4] * W_k[4]) -
         W_k[1] * (W_k[1] * W_k[5] - W_k[2] * W_k[4]) +
         W_k[2] * (W_k[1] * W_k[4] - W_k[2] * W_k[3]);
}
//=========================================================================
// inverse for a 3x3 symmetric matrix
template <class T>
inline void utilsAmatf_inverseSymmetricW_3D(T *W, T *W_inverse) {
  T detW = utilsAmatf_determinantSymmetricW_3D(W);
  if (fabs(detW) < std::numeric_limits<T>::epsilon())  // matrix is singular
  {
    W_inverse[0] = std::numeric_limits<T>::max();
    W_inverse[1] = std::numeric_limits<T>::max();
    W_inverse[2] = std::numeric_limits<T>::max();
    W_inverse[3] = std::numeric_limits<T>::max();
    W_inverse[4] = std::numeric_limits<T>::max();
    W_inverse[5] = std::numeric_limits<T>::max();
    return;
  }
  W_inverse[0] = (W[3] * W[5] - W[4] * W[4]) / detW;
  W_inverse[1] = (W[4] * W[2] - W[1] * W[5]) / detW;
  W_inverse[2] = (W[1] * W[4] - W[3] * W[2]) / detW;

  W_inverse[3] = (W[0] * W[5] - W[2] * W[2]) / detW;
  W_inverse[4] = (W[1] * W[2] - W[0] * W[4]) / detW;

  W_inverse[5] = (W[0] * W[3] - W[1] * W[1]) / detW;

  return;
}
// analytical solution for eigenvalues 3x3 real symmetric matrices: IT DOES NOT
// HAVE TO BE POSITIVE-DEFINITE (ONLY SYMMETRIC)
// formula for eigenvalues from
// http://en.wikipedia.org/wiki/Eigenvalue_algorithm#Eigenvalues_of_3.C3.973_matrices
template <class T>
inline void utilsAmatf_eig3(const T *w, T *d, T *v) {
  double m, p, q;  // to keep precision
  int vIsZero = 0;
  double phi, aux1, aux2, aux3;

  // calculate determinant to check if matrix is singular
  q = utilsAmatf_determinantSymmetricW_3D(w);

  if (fabs(q) < 1e-24)  // we consider matrix is singular
  {
    d[0] = 0.0;
    // solve a quadratic equation
    m = -w[0] - w[3] - w[5];
    q = -w[1] * w[1] - w[2] * w[2] - w[4] * w[4] + w[0] * w[3] + w[0] * w[5] +
        w[3] * w[5];
    p = m * m - 4.0 * q;
    if (p < 0)
      p = 0.0;  // to avoid numerical errors (symmetric matrix should have real
                // eigenvalues)
    else
      p = sqrt(p);

    d[1] = 0.5 * (-m + p);
    d[2] = 0.5 * (-m - p);

  } else {                           // matrix not singular
    m = (w[0] + w[3] + w[5]) / 3.0;  // trace of w /3
    q = 0.5 *
        ((w[0] - m) * ((w[3] - m) * (w[5] - m) - w[4] * w[4]) -
         w[1] * (w[1] * (w[5] - m) - w[2] * w[4]) +
         w[2] * (w[1] * w[4] - w[2] * (w[3] - m)));  // determinant(a-mI)/2
    p = (2.0 * (w[1] * w[1] + w[2] * w[2] + w[4] * w[4]) +
         (w[0] - m) * (w[0] - m) + (w[3] - m) * (w[3] - m) +
         (w[5] - m) * (w[5] - m)) /
        6.0;

    // NOTE: the follow formula assume accurate computation and therefor
    // q/p^(3/2) should be in range of [1,-1],
    // but in real code, because of numerical errors, it must be checked. Thus,
    // in case abs(q) >= abs(p^(3/2)), set phi = 0;
    phi = q / pow(p, 1.5);
    if (phi <= -1)
      phi = utilsAmatf_PI_ / 3.0;
    else if (phi >= 1)
      phi = 0;
    else
      phi = acos(phi) / 3.0;

    aux1 = cos(phi);
    aux2 = sin(phi);
    aux3 = sqrt(p);

    // eigenvalues
    d[0] = m + 2.0 * aux3 * aux1;
    d[1] = m - aux3 * (aux1 + utilsAmatf_SQRT3_ * aux2);
    d[2] = m - aux3 * (aux1 - utilsAmatf_SQRT3_ * aux2);
  }

  if (v == NULL)  // we are only interested in eigenvalues
    return;

  // eigenvectors
  v[0] = w[1] * w[4] - w[2] * (w[3] - d[0]);
  v[1] = w[2] * w[1] - w[4] * (w[0] - d[0]);
  v[2] = (w[0] - d[0]) * (w[3] - d[0]) - w[1] * w[1];
  v[3] = w[1] * w[4] - w[2] * (w[3] - d[1]);
  v[4] = w[2] * w[1] - w[4] * (w[0] - d[1]);
  v[5] = (w[0] - d[1]) * (w[3] - d[1]) - w[1] * w[1];
  v[6] = w[1] * w[4] - w[2] * (w[3] - d[2]);
  v[7] = w[2] * w[1] - w[4] * (w[0] - d[2]);
  v[8] = (w[0] - d[2]) * (w[3] - d[2]) - w[1] * w[1];

  // normalize eigenvectors
  phi = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (phi > 1e-12) {
    v[0] /= phi;
    v[1] /= phi;
    v[2] /= phi;
  } else {  // numerically seems zero: we need to try the other pair of vectors
            // to form the null space (it could be that v1 and v2 were parallel)
    v[0] = w[1] * (w[5] - d[0]) - w[2] * w[4];
    v[1] = w[2] * w[2] - (w[5] - d[0]) * (w[0] - d[0]);
    v[2] = (w[0] - d[0]) * w[4] - w[1] * w[2];
    phi = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (phi > 1e-12) {
      v[0] /= phi;
      v[1] /= phi;
      v[2] /= phi;
    } else
      vIsZero += 1;
  }

  phi = sqrt(v[3] * v[3] + v[4] * v[4] + v[5] * v[5]);
  if (phi > 1e-12) {
    v[3] /= phi;
    v[4] /= phi;
    v[5] /= phi;
  } else {  // numerically seems zero: we need to try the
    v[3] = w[1] * (w[5] - d[1]) - w[2] * w[4];
    v[4] = w[2] * w[2] - (w[5] - d[1]) * (w[0] - d[1]);
    v[5] = (w[0] - d[1]) * w[4] - w[1] * w[2];
    phi = sqrt(v[3] * v[3] + v[4] * v[4] + v[5] * v[5]);
    if (phi > 1e-12) {
      v[3] /= phi;
      v[4] /= phi;
      v[5] /= phi;
    } else
      vIsZero += 2;
  }

  phi = sqrt(v[6] * v[6] + v[7] * v[7] + v[8] * v[8]);
  if (phi > 1e-12) {
    v[6] /= phi;
    v[7] /= phi;
    v[8] /= phi;
  } else {  // numerically seems zero: we need to try the
    v[6] = w[1] * (w[5] - d[2]) - w[2] * w[4];
    v[7] = w[2] * w[2] - (w[5] - d[2]) * (w[0] - d[2]);
    v[8] = (w[0] - d[2]) * w[4] - w[1] * w[2];
    phi = sqrt(v[6] * v[6] + v[7] * v[7] + v[8] * v[8]);
    if (phi > 1e-12) {
      v[6] /= phi;
      v[7] /= phi;
      v[8] /= phi;
    } else
      vIsZero += 4;
  }

  // adjust v in case zome eigenvalues are zeros
  switch (vIsZero) {
    case 1:
      v[0] = v[4] * v[8] - v[5] * v[7];
      v[1] = v[5] * v[6] - v[3] * v[8];
      v[2] = v[4] * v[6] - v[3] * v[7];
      break;

    case 2:
      v[3] = v[1] * v[8] - v[2] * v[7];
      v[4] = v[2] * v[6] - v[0] * v[8];
      v[5] = v[1] * v[6] - v[0] * v[7];
      break;

    case 4:
      v[6] = v[4] * v[2] - v[5] * v[1];
      v[7] = v[5] * v[0] - v[3] * v[2];
      v[8] = v[4] * v[0] - v[3] * v[1];
      break;
    case 3:
      phi = sqrt(v[7] * v[7] + v[6] * v[6]);
      if (phi < 1e-12)  // it means first eigenvector is [0 0 1]
      {
        v[3] = 1.0;
        v[4] = 0.0;
        v[5] = 0.0;
      } else {
        v[3] = -v[7] / phi;
        v[4] = v[6] / phi;
        v[5] = 0.0;
      }
      v[0] = v[4] * v[8] - v[5] * v[7];
      v[1] = v[5] * v[6] - v[3] * v[8];
      v[2] = v[3] * v[7] - v[4] * v[6];
      break;

    case 6:
      phi = sqrt(v[1] * v[1] + v[0] * v[0]);
      if (phi < 1e-12)  // it means first eigenvector is [0 0 1]
      {
        v[6] = 1.0;
        v[7] = 0.0;
        v[8] = 0.0;
      } else {
        v[6] = -v[1] / phi;
        v[7] = v[0] / phi;
        v[8] = 0.0;
      }
      v[3] = v[1] * v[8] - v[2] * v[7];
      v[4] = v[2] * v[6] - v[0] * v[8];
      v[5] = v[0] * v[7] - v[1] * v[6];
      break;

    case 5:
      phi = sqrt(v[4] * v[4] + v[5] * v[5]);
      if (phi < 1e-12)  // it means first eigenvector is [0 0 1]
      {
        v[0] = 1.0;
        v[1] = 0.0;
        v[2] = 0.0;
      } else {
        v[0] = -v[4] / phi;
        v[1] = v[5] / phi;
        v[2] = 0.0;
      }
      v[6] = v[4] * v[2] - v[5] * v[1];
      v[7] = v[5] * v[0] - v[3] * v[2];
      v[8] = v[1] * v[3] - v[4] * v[0];
      break;

    case 7:  // matrix is basically zero: so we set eigenvectors to identity
             // matrix
      v[1] = v[2] = v[3] = v[5] = v[6] = v[7] = 0.0;
      v[0] = v[4] = v[8] = 1.0;
      break;
  }

  // make sure determinant is +1 for teh rotation matrix
  phi = v[0] * (v[4] * v[8] - v[5] * v[7]) -
        v[1] * (v[3] * v[8] - v[5] * v[6]) + v[2] * (v[3] * v[7] - v[4] * v[6]);
  if (phi < 0) {
    v[0] = -v[0];
    v[1] = -v[1];
    v[2] = -v[2];
  }
}
//----------------------------------------------------------------
//-------------------------------------------------------------------
// analytical solution for eigenvalues 2x2 real symmetric matrices
template <class T>
inline void utilsAmatf_eig2(const T *w, T *d, T *v) {
  double aux1, phi;
  int vIsZero = 0;

  aux1 = (w[0] + w[2]) / 2.0;
  phi = sqrt(4.0 * w[1] * w[1] + (w[0] - w[2]) * (w[0] - w[2])) / 2.0;

  d[0] = aux1 + phi;
  d[1] = aux1 - phi;

  // calculate eigenvectors
  // eigenvectors
  v[0] = -w[1];
  v[1] = w[0] - d[0];
  v[2] = -w[1];
  v[3] = w[0] - d[1];

  // normalize eigenvectors
  phi = sqrt(v[0] * v[0] + v[1] * v[1]);
  if (phi > 0) {
    v[0] /= phi;
    v[1] /= phi;
  } else
    vIsZero += 1;

  phi = sqrt(v[2] * v[2] + v[3] * v[3]);
  if (phi > 0) {
    v[2] /= phi;
    v[3] /= phi;
  } else
    vIsZero += 2;

  switch (vIsZero) {
    case 1:
      v[0] = -v[3];
      v[1] = v[2];
      break;
    case 2:
      v[2] = -v[1];
      v[3] = v[0];
      break;
    case 3:  // matrix is basically zero: so we set eigenvectors to identity
             // matrix
      v[1] = v[2] = 0.0;
      v[0] = v[3] = 1.0;
      break;
  }
  // make sure determinant is +1 for teh rotation matrix
  phi = v[0] * v[3] - v[1] * v[2];
  if (phi < 0) {
    v[0] = -v[0];
    v[1] = -v[1];
  }
}

//===============================end of eigenvalus
// functionality============================================================

// VIP: THIS FUNCTION HAS TO MATCH
// GaussianMixtureModel::regularizePrecisionMatrix(void)
// needed for W regularization
// static const double lambdaMin2=0.02;//aux=scaleSigma/(maxRadius*maxRadius)
// with scaleSigma=2.0 and maxRadius=10 (adjust with scale)
// static const double lambdaMax2=0.2222;//aux=scaleSigma/(maxRadius*maxRadius)
// with scaleSigma=2.0 and minRadius=3.0 (adjust with scale)  (when nuclei
// divide they can be very narrow)
// static const double maxExcentricity2=3.0*3.0;//maximum excentricity allowed:
// sigma[i]=1/sqrt(d[i]). Therefore maxExcentricity needs to be squared to used
// in terms of radius.

template <class T>
inline void utilsAmatf_regularizePrecisionMatrix(T *W_k, float *scale,
                                                 bool W_4DOF_) {
  if (regularizePrecisionMatrixConstants::lambdaMax < 0.0) {
    cout << "ERROR: utilsAmatf_regularizePrecisionMatrix: "
            "regularizePrecisionMatrixConstants have not been initialize"
         << endl;
    exit(3);
  }

  double auxMax = regularizePrecisionMatrixConstants::
      lambdaMax;  // aux=scaleSigma/(minRadius*minRadius) with scaleSigma=2.0
  double auxMin = regularizePrecisionMatrixConstants::
      lambdaMin;  // aux=scaleSigma/(maxRadius*maxRadius) with scaleSigma=2.0

  // to adjust for scale: this values is empirical
  // If I don;t do this rescaling, I would have to find which eigenvector
  // corresponds to Z dirction to check for min/max Radius
  // basically, maxRadius_z=scaleGMEMCUDA[0]*maxRadius_x/scaleGMEMCUDA[2]
  int count = 0;
  for (int ii = 0; ii < dimsImage; ii++) {
    if (scale[ii] < 1e-3) {
      cout << "ERROR: utilsAmatf_regularizePrecisionMatrix: scale is below "
              "zero. it shoudl be set properly before calling this function"
           << endl;
      exit(3);
    }
    W_k[count++] /= scale[ii] * scale[ii];
    for (int jj = ii + 1; jj < dimsImage; jj++)
      W_k[count++] /= scale[ii] * scale[jj];
  }

  if (W_4DOF_ == true) {
    W_k[2] = 0.0f;
    W_k[4] = 0.0f;
  }
  T d[dimsImage], v[dimsImage * dimsImage];
  // calculate eigenvalues and eigenvectors
  utilsAmatf_eig3(W_k, d, v);  // NOTE: if dimsImage!=3 it won't work

  // avoid minimum size
  if (d[0] > auxMax) d[0] = auxMax;
  if (d[1] > auxMax) d[1] = auxMax;
  if (d[2] > auxMax) d[2] = auxMax;

  // avoid maximum size
  if (d[0] < auxMin) d[0] = auxMin;
  if (d[1] < auxMin) d[1] = auxMin;
  if (d[2] < auxMin) d[2] = auxMin;

  // avoid too much excentricity
  double auxNu = d[0] / d[1];
  if (auxNu > regularizePrecisionMatrixConstants::maxExcentricity)
    d[0] = regularizePrecisionMatrixConstants::maxExcentricity * d[1];
  else if (1. / auxNu > regularizePrecisionMatrixConstants::maxExcentricity)
    d[1] = regularizePrecisionMatrixConstants::maxExcentricity * d[0];

  auxNu = d[0] / d[2];
  if (auxNu > regularizePrecisionMatrixConstants::maxExcentricity)
    d[0] = regularizePrecisionMatrixConstants::maxExcentricity * d[2];
  else if (1. / auxNu > regularizePrecisionMatrixConstants::maxExcentricity)
    d[2] = regularizePrecisionMatrixConstants::maxExcentricity * d[0];

  auxNu = d[1] / d[2];
  if (auxNu > regularizePrecisionMatrixConstants::maxExcentricity)
    d[1] = regularizePrecisionMatrixConstants::maxExcentricity * d[2];
  else if (1. / auxNu > regularizePrecisionMatrixConstants::maxExcentricity)
    d[2] = regularizePrecisionMatrixConstants::maxExcentricity * d[1];

  // reconstruct W_k=V*D*V'
  W_k[0] = d[0] * v[0] * v[0] + d[1] * v[3] * v[3] + d[2] * v[6] * v[6];
  W_k[3] = d[0] * v[1] * v[1] + d[1] * v[4] * v[4] + d[2] * v[7] * v[7];
  W_k[5] = d[0] * v[2] * v[2] + d[1] * v[5] * v[5] + d[2] * v[8] * v[8];

  W_k[1] = d[0] * v[0] * v[1] + d[1] * v[3] * v[4] + d[2] * v[6] * v[7];

  if (W_4DOF_ == false) {
    W_k[2] = d[0] * v[0] * v[2] + d[1] * v[3] * v[5] + d[2] * v[6] * v[8];
    W_k[4] = d[0] * v[1] * v[2] + d[1] * v[4] * v[5] + d[2] * v[7] * v[8];
  }

  // undo adjust for scale:
  // to adjust for scale: this values is empirical
  // If I don't do this rescaling, I would have to find which eigenvector
  // corresponds to Z dirction to check for min/max Radius
  count = 0;
  for (int ii = 0; ii < dimsImage; ii++) {
    W_k[count++] *= scale[ii] * scale[ii];
    for (int jj = ii + 1; jj < dimsImage; jj++)
      W_k[count++] *= scale[ii] * scale[jj];
  }
}

//----------------------------------------------------------------
//-------------------------------------------------------------------
// calculates p' * W_k * p  with W_k[6] a symmetric positive definite matrix
template <class T>
inline T utilsAmatf_MahalanobisDistance_3D(const T *p, const T *W_k) {
  return (p[0] * (p[0] * W_k[0] + p[1] * W_k[1] + p[2] * W_k[2]) +
          p[1] * (p[0] * W_k[1] + p[1] * W_k[3] + p[2] * W_k[4]) +
          p[2] * (p[0] * W_k[2] + p[1] * W_k[4] + p[2] * W_k[5]));
}

//======================================================================
// first and second moments of a vector calculate in a single pass
template <class T>
inline void utilsAmatf_meanAndStd(const vector<T> &vec, double *mean,
                                  double *std) {
  *mean = 0.0;
  *std = 0.0;

  for (typename vector<T>::const_iterator iter = vec.begin(); iter != vec.end();
       ++iter) {
    (*mean) += (*iter);
    (*std) += (*iter) * (*iter);
  }

  (*mean) /= vec.size();

  (*std) -= (*mean) * (*mean) * vec.size();
  (*std) /= (vec.size() - 1);
  (*std) = sqrt(*std);

  return;
}

template <class T>
inline void utilsAmatf_mean(const vector<T> &vec, double *mean) {
  *mean = 0.0;

  for (typename vector<T>::const_iterator iter = vec.begin(); iter != vec.end();
       ++iter) {
    (*mean) += (*iter);
  }

  (*mean) /= vec.size();

  return;
}

//=============================================================================
// power method (iterative) to find first principal component. YOU NEED TO
// SUBTRACT MEAN YOURSELF FROM A
// A = matrix A[0].size x 3 (skinny it has more than 3 points)
// v = first principal component (unit vector)
// lambda = scalar
// U = array of length A[0].size with projection along v
template <class T>
inline void utilsAmatf_firstPC(const vector<T> A[dimsImage],
                               double v[dimsImage], double *lambda,
                               double *U = NULL) {
  const int maxIter = 100;

  for (int ii = 0; ii < dimsImage; ii++)
    v[ii] = 1.0 / ((double)dimsImage);  // initialization

  int numIter = 0;
  size_t N = A[0].size();  // number of data points
  double vOld[dimsImage];

  double normVV = 100;

  vector<T> auxA(N);

  while (numIter < maxIter && normVV > 1e-3) {
    memcpy(vOld, v, sizeof(double) * dimsImage);

    // v = A' * (Av);
    memset(&(auxA[0]), 0, sizeof(T) * N);
    for (int ii = 0; ii < dimsImage; ii++) {
      for (size_t jj = 0; jj < A[ii].size(); jj++) {
        auxA[jj] += A[ii][jj] * vOld[ii];
      }
    }
    memset(v, 0, sizeof(double) * dimsImage);
    for (int ii = 0; ii < dimsImage; ii++) {
      for (size_t jj = 0; jj < A[ii].size(); jj++) {
        v[ii] += A[ii][jj] * auxA[jj];
      }
    }

    // normalize v
    *lambda = 0.0f;
    for (int ii = 0; ii < dimsImage; ii++) (*lambda) += (v[ii] * v[ii]);

    (*lambda) = sqrt(*lambda);

    // check if we have converged
    normVV = 0;
    for (int ii = 0; ii < dimsImage; ii++) {
      v[ii] /= (*lambda);
      normVV += (v[ii] - vOld[ii]) * (v[ii] - vOld[ii]);
    }
    numIter++;
  }

  // principal component lambda = sqrt( eig )
  (*lambda) = sqrt(*lambda);

  if (U != NULL)  // calculate projections
  {
    memset(U, 0, sizeof(double) * N);
    for (int ii = 0; ii < dimsImage; ii++) {
      for (size_t jj = 0; jj < A[ii].size(); jj++) {
        U[jj] += v[ii] * A[ii][jj];  // dot product
      }
    }

    // normalize U
    for (size_t jj = 0; jj < A[0].size(); jj++) U[jj] /= (*lambda);
  }
}

//=========================================================================================================

// PCA for matrices with 3 features taking advanatge that we have eig3
// A = matrix A[0].size x 3 (skinny it has more than 3 points)
// v = principal components (unit vector)
// lambda = array of length 3
// U = array of with the same number of elements as A with projection along v.
// U[0],U[1],U[2],...U[N-1] contains teh porjections for all the points along
// v_0
template <class T>
inline void utilsAmatf_PCA(const vector<T> A[dimsImage],
                           double v[dimsImage * dimsImage], double *lambda,
                           double *U = NULL) {
  size_t N = A[0].size();  // number of data points

  double w[dimsImage * (1 + dimsImage) /
           2];  // to store A'*A (3x3 symmetric matrix)
  utilsAmatf_eig3<double>(w, lambda, v);

  if (U != NULL)  // calculate projections
  {
    int offsetV, offsetU;
    memset(U, 0, sizeof(double) * dimsImage * N);
    for (int aa = 0; aa < dimsImage; aa++) {
      for (int ii = 0; ii < dimsImage; ii++) {
        offsetV = aa * dimsImage + ii;
        offsetU = ii * N;  // offsetU = jj + ii * N;
        for (size_t jj = 0; jj < A[ii].size(); jj++) {
          U[offsetU] += v[offsetV] * A[ii][jj];  // dot product
          offsetU++;
        }
      }
    }
  }
}

//====================================================================================================
// retursn the roots of a cubic polynomial of the form AX3 + BX2 + CX + D = 0
// code adapted from http://www.1728.org/cubic.htm

template <class T>
inline void utilsAmatf_rootsCubibcPolynomial(T a, T b, T c, T d,
                                             T solutionReal[3],
                                             T solutionImag[3]) {
  cout
      << "ERROR: utilsAmatf_rootsCubibcPolynomial: not tested yet!!!!!!!!!"
      << endl;  // TODO: test teh function since it was adapted from java script
  exit(3);
  /*
  //--EVALUATING THE 'f'TERM
  T f = ((3*c)/a) - (((b*b)/(a*a)))/3;

  //--EVALUATING THE 'g'TERM -->
  T g = (2*((b*b*b)/(a*a*a))-(9*b*c/(a*a)) + ((27*(d/a))))/27;

  //--EVALUATING THE 'h'TERM
  T h = ((g*g)/4) + ((f*f*f)/27);

  if (h > 0)
  {
          T m = -(g/2)+ (sqrt(h));

          //-- K is used because math.pow cannot compute negative cube roots -->
          T k=1;
          if (m < 0)
                  k=-1;
          else
                  k=1;

          T m2 = (pow((m*k),(1.0/3)));
          m2 = m2*k;
          k=1;
          n = (-(g/2)- (Math.sqrt(h)));
          if (n < 0) k=-1; else k=1;
          n2 = (pow((n*k),(1/3)));
          n2 = n2*k;
          k=1;
          x1= eval ((m2 + n2) - (b/(3*a)));

          //--                      ((S+U)     - (b/(3*a)))-->
          x2=(-1*(m2 + n2)/2 - (b/(3*a)) + " + i* " + ((m2 - n2)/2)*pow(3,.5));
          //                      -(S + U)/2  - (b/3a) + i*(S-U)*(3)^.5-->
          x3=(-1*(m2 + n2)/2 - (b/(3*a)) + " - i* " + ((m2 - n2)/2)*pow(3,.5));
  }

  //                      -(S + U)/2  - (b/3a) - i*(S-U)*(3)^.5-->

  if (h<=0)
  {
          r = ((Math.sqrt((g*g/4)-h)));
          k=1;
          if (r<0) k=-1;
          // rc is the cube root of 'r' -->
          rc = pow((r*k),(1/3))*k;
          k=1;
          theta =Math.acos((-g/(2*r)));
          x1=eval (2*(rc*Math.cos(theta/3))-(b/(3*a)));
          x2a=rc*-1;
          x2b= Math.cos(theta/3);
          x2c= Math.sqrt(3)*(Math.sin(theta/3));
          x2d= (b/3*a)*-1;
          x2=(x2a*(x2b + x2c))-(b/(3*a));
          x3=(x2a*(x2b - x2c))-(b/(3*a));

          x1=x1*1E+14;
          x1=Math.round(x1);
          x1=(x1/1E+14);
          x2=x2*1E+14;
          x2=Math.round(x2);
          x2=(x2/1E+14);
          x3=x3*1E+14;
          x3=Math.round(x3);
          x3=(x3/1E+14);

  }



  if ((f+g+h)==0)
  {
          if (d<0) {sign=-1};if (d>=0) {sign=1};
          if (sign>0){dans=pow((d/a),(1/3));dans=dans*-1};
          if (sign<0){d=d*-1;dans=pow((d/a),(1/3))};
          x1=dans; x2=dans;x3=dans;
  }
  */
}

#endif
