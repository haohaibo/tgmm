/*
 * testKnnResults.h
 */

#ifndef TESTKNNRESULTS_H_
#define TESTKNNRESULTS_H_

bool testKnnResults(float *ref, float *query, int *ind, int ref_nb,
                    int query_nb, int dimsImage, int maxGaussiansPerVoxel);
int mainTestKnnCudaCppEmule(void);

#endif /* TESTKNNRESULTS_H_ */
