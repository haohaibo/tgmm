/*
 * GMEMupdateCUDA.h
 *
 */

#ifndef GMEMUPDATECUDA_H_
#define GMEMUPDATECUDA_H_

#include <fstream>
#include <vector>
#include "../Utils/CSparse.h"
#include "../constants.h"

using namespace std;
// reduced version of the GaussianMixtureModel class to operate in the GPU
// VIP: if you update this structure you need to update void
// copy2GMEM_CUDA(GaussianMixtureModel *GM,GaussianMixtureModelCUDA *GMCUDAtemp)
// in variationalInference.cpp

static const int MAX_SUPERVOXELS_PER_GAUSSIAN =
    16;  // since I do it in the GPU I can not use vector for dynamic allocation
struct GaussianMixtureModelCUDA {
  double beta_k, nu_k, alpha_k;
  double m_k[dimsImage];
  double W_k[dimsImage * (dimsImage + 1) / 2];  // symmetric matrix

  double expectedLogDetCovarianceCUDA;
  double expectedLogResponsivityCUDA;
  double splitScore;

  // priors
  double beta_o, nu_o, alpha_o;
  double m_o[dimsImage];
  double W_o[dimsImage * (dimsImage + 1) / 2];  // symmetric matrix

  // indicators
  bool fixed;

  // to store indexes of the supervoxel that the Gaussian belongs to
  int supervoxelIdx[MAX_SUPERVOXELS_PER_GAUSSIAN];
  int supervoxelNum;

  ostream &writeXML(ostream &os, int id);
  static ostream &writeXMLheader(ostream &os) {
    os << "<?xml version=\"1.0\" encoding=\"utf-8\"?>" << endl
       << "<document>" << endl;
    return os;
  };
  static ostream &writeXMLfooter(ostream &os) {
    os << "</document>" << endl;
    return os;
  };

  // assignment operator
  GaussianMixtureModelCUDA &operator=(const GaussianMixtureModelCUDA &p);

  // constructor
  // GaussianMixtureModelCUDA();
};

void allocateMemoryForGaussianMixtureModelCUDA(
    GaussianMixtureModelCUDA **vecGMCUDA, pxi **rnkCUDA,
    double **totalAlphaTempCUDA, long long int query_nb, int ref_nb,
    int numGridsCheck);
void updateMemoryForGaussianMixtureModelCUDA(
    GaussianMixtureModelCUDA *vecGMCUDAtemp,
    GaussianMixtureModelCUDA *vecGMCUDA, int ref_nb);
void GMEMcomputeRnkCUDA(pxi *rnk, pxi *rnkCUDA, int *indCUDA, float *queryCUDA,
                        long long int query_nb, float *imgDataCUDA,
                        GaussianMixtureModelCUDA *vecGMCUDA);
void GMEMcomputeRnkCUDAInplace(pxi *rnkCUDA, int *indCUDA, float *queryCUDA,
                               long long int query_nb, float *imgDataCUDA,
                               GaussianMixtureModelCUDA *vecGMCUDA);
void deallocateMemoryForGaussianMixtureModelCUDA(
    GaussianMixtureModelCUDA **vecGMCUDA, pxi **rnkCUDA,
    double **totalAlphaTempCUDA);
void GMEMupdatePriorConstantsCUDA(GaussianMixtureModelCUDA *vecGMCUDA,
                                  int ref_nb, double totalAlphaDiGamma);
double calculateTotalAlpha(GaussianMixtureModelCUDA *vecGMCUDA,
                           double *totalAlphaTempCUDA, double *totalAlphaTemp,
                           int ref_nb, int numGridsCheck);
void copyScaleToDvice(float *scale);

// void GMEMvariationalInferenceCUDA(float *queryCUDA,float
// *imgDataCUDA,GaussianMixtureModelCUDA *vecGMtemp,long long int query_nb,int
// ref_nb,int maxIterEM,double tolLikelihood,int devCUDA,int frame);
void GMEMvariationalInferenceCUDA(
    float *queryCUDA, float *imgDataCUDA, pxi *rnkCUDA, pxi *rnkCUDAtr,
    int *indCUDA, int *indCUDAtr, GaussianMixtureModelCUDA *vecGMtemp,
    long long int query_nb, int ref_nb, int maxIterEM, double tolLikelihood,
    int devCUDA, int frame, bool W4DOF, string debugPath = "");
void GMEMvariationalInferenceCUDAWithSupervoxels(
    float *queryCUDA, float *imgDataCUDA, pxi *rnkCUDA, pxi *rnkCUDAtr,
    int *indCUDA, int *indCUDAtr, float *centroidLabelPositionCUDA,
    long long int *labelListPtrCUDA, GaussianMixtureModelCUDA *vecGMtemp,
    long long int query_nb, int ref_nb, unsigned short int maxLabels,
    int maxIterEM, double tolLikelihood, int devCUDA, int frame, bool W4DOF,
    string debugPath = "");
double addTotalLikelihood(float *imgDataCUDA, float *likelihoodVecCUDA,
                          float *finalSumVectorInHostF,
                          GaussianMixtureModelCUDA *vecGMCUDA, int *indCUDA,
                          float *queryCUDA, int query_nb, int ref_nb,
                          double totalAlpha);
double addTotalLikelihoodWithSupervoxels(
    float *imgDataCUDA, float *likelihoodVecCUDA, float *finalSumVectorInHostF,
    GaussianMixtureModelCUDA *vecGMCUDA, int *indCUDA, float *queryCUDA,
    long long int *labelListPtrCUDA, int query_nb, unsigned short int numLabels,
    double totalAlpha);
double addLocalLikelihoodWithSupervoxels(
    float *imgDataCUDA, float *likelihoodVecCUDA, float *finalSumVectorInHostF,
    GaussianMixtureModelCUDA *vecGMCUDA, int *indCUDA, float *queryCUDA,
    long long int *labelListPtrCUDA, long long int query_nb, int ref_nb,
    unsigned short int numLabels, double totalAlpha,
    int *listSupervoxelsIdxCUDA, int listSupervoxelsIdxLength);

/*
\brief Calculates local likelihood. Each region is defined by a list of nearby
supervoxels. Useful to perform likelihood ratio tests for cell division
*/
void calculateLocalLikelihood(vector<double> &localLikelihood,
                              const vector<vector<int> > &listSupervoxels,
                              float *queryCUDA, float *imgDataCUDA,
                              pxi *rnkCUDA, int *indCUDA,
                              GaussianMixtureModelCUDA *vecGMtemp,
                              long long int *labelListPtrCUDA,
                              long long int query_nb, int ref_nb,
                              int numLabels);

void GMEMinitializeMemory(float **queryCUDA, float *query, float **imgDataCUDA,
                          float *imgData, float *scale, long long int query_nb,
                          float **rnkCUDA, int **indCUDA, int **indCUDAtr);
void GMEMinitializeMemoryWithSupervoxels(
    float **queryCUDA, float *query, float **imgDataCUDA, float *imgData,
    float *scale, long long int query_nb, float **rnkCUDA, int **indCUDA,
    int **indCUDAtr, unsigned short int numLabels,
    float **centroidLabelPositionCUDA, float *centroidLabelPositionlong,
    long long int **labelListPtrCUDA, long long int *labelListPtrHOST);
void GMEMreleaseMemory(float **queryCUDA, float **imgDataCUDA, float **rnkCUDA,
                       int **indCUDA, int **indCUDAtr);
void GMEMreleaseMemoryWithSupervoxels(float **queryCUDA, float **imgDataCUDA,
                                      float **rnkCUDA, int **indCUDA,
                                      int **indCUDAtr,
                                      float **centroidLabelPositionCUDA,
                                      long long int **labelListPtrCUDA);

void GMEMsetcsrRowPtrA(int *csrRowPtrAHost, int *csrRowPtrACUDA, int *indCUDAtr,
                       long long int query_nb, int ref_nb);
void findVoxelAssignmentBasedOnRnk(int *kAssignment, long long int query_nb,
                                   float *rnkCUDA, int *indCUDA);
// debugging
void mainTestGMEMcomputeRnkCUDA(long long int query_nb);
void mainTestGMEMcomputeVariationalInference(long long int query_nb);
void printGaussianMixtureModelCUDA(GaussianMixtureModelCUDA *vecGMCUDA,
                                   float *refTempCUDA, int idx);
void writeXMLdebugCUDA(GaussianMixtureModelCUDA *vecGMtemp,
                       unsigned int numElem);

#endif /* GMEMUPDATECUDA_H_ */
