
/* Bring in the declarations for the string functions */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <limits>
#include "sparseHungarianAlgorithm.h"

#include "munkres-cpp-master/src/munkres.h"

void workspaceSpraseHungarian_init(workspaceSparseHungarian* W, int N_, int M_, int E_, double thrCost_)
{
	W->N = N_;
	W->M = M_;
	W->E = E_;
	W->edgesId = (int*) malloc( E_ * sizeof(int));
	W->edgesPtr = (int*) malloc( (N_+1) * sizeof(int));
	W->edgesW = (double*) malloc( E_ * sizeof(double) );
	W->thrCost = thrCost_;
}
void workspaceSpraseHungarian_destroy(workspaceSparseHungarian* W)
{
	if( W->edgesId !=NULL)
	{
		free(W->edgesId);
		W->edgesId = NULL;
	}
	if( W->edgesPtr !=NULL)
	{
		free(W->edgesPtr);
		W->edgesPtr = NULL;
	}
	if( W->edgesW !=NULL)
	{
		free(W->edgesW);
		W->edgesW = NULL;
	}
}


int solveSparseHungarianAlgorithm(const workspaceSparseHungarian* W, int* assignmentId)
{
	//nrows=number of to be matched
	//ncols=number of candidate points (including garbage potential)
	//nrows<=ncols or the code crashes (but this is guaranteed due to the garbage potential)

	int nrows = W->N;
	int ncols = W->M + W->N;//one garbage possibility per element
	Matrix<double> matrix(nrows, ncols);


	// Initialize matrix with infinite value (to block assignment)
	for ( int row = 0 ; row < nrows ; row++ ) {
		for ( int col = 0 ; col < ncols ; col++ ) {
			matrix(row,col) = std::numeric_limits<double>::max();
		}
	}

	//set weights for defined edges
	for( int row =0 ; row < nrows; row++)
	{
		for(int ii = W->edgesPtr[row]; ii < W->edgesPtr[row+1]; ii++)
		{
			matrix(row,W->edgesId[ii]) = W->edgesW[ii];
		}
		//set weight for the garbage potential
		matrix(row, W->M+row) = W->thrCost;
	}


	//solve assignment problem
	Munkres m;
	m.solve(matrix);

	//parse assignemnt
	for( int row =0 ; row < nrows; row++)
	{
		assignmentId[row] = -1;//by default it is garbage potential
		for ( int col = 0 ; col < W->M ; col++ )
		{
			if( matrix(row,col) == 0 )
			{
				assignmentId[row] = col;
				break;
			}
		}
	}

	return 0;
}

//=========================================================
int mainTestSparseHungarianAlgorithm (int argc, const char **argv)
{
	int N = 3;//number of points to be matched
	int M = 4;//number of candidate points
	int E= 5; //number of edges
	double thrCost = 0.85; //edge cost for the garbagae potential

	int edgesId[5] = {0, 1, 1, 2, 3};
	int edgesPtr[4] = {0, 2, 3, 5};
	double edgesW[5] = {0.7, 0.5, 0.8, 0.9, 0.8};


	int *assignmentId = NULL;
	int ii;

	//create workspace
	workspaceSparseHungarian *W = (workspaceSparseHungarian*) malloc(sizeof(workspaceSparseHungarian));
	workspaceSpraseHungarian_init(W,N,M,E,thrCost);

	assignmentId = (int*) malloc(sizeof(int) * N);

	//copy values
	for(ii = 0;ii<W->E; ii++)
	{
		W->edgesId[ii] = edgesId[ii];
		W->edgesW[ii] = edgesW[ii];
	}
	for(ii = 0;ii<W->N+1; ii++)
	{
		W->edgesPtr[ii] = edgesPtr[ii];
	}

	
	//call CPP code to solve Hungarian algorithm
	ii = solveSparseHungarianAlgorithm(W, assignmentId);
	if(ii > 0)
		return ii;

	for(ii = 0; ii<W->N; ii++)
		printf("%d->%d\n",ii,assignmentId[ii]);


	//release memory
	workspaceSpraseHungarian_destroy(W);
	free(assignmentId);
	free(W);


	return 0;
}
