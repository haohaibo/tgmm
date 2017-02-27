/*
 * CSparse.h
 *
 *  Created on: Jul 21, 2011
 *      Author: amatf
 *
 *      Very condensed version from CSparse library to use data structures and basic functionality
 */

#ifdef __cplusplus
 extern "C" {
#endif


#ifndef CSPARSE_H_
#define CSPARSE_H_

#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stddef.h>

#define CS_VER 3                    /* CSparse Version */
#define CS_SUBVER 0
#define CS_SUBSUB 1
#define CS_DATE "Jan 19, 2010"     /* CSparse release date */
#define CS_COPYRIGHT "Copyright (c) Timothy A. Davis, 2006-2010"

/* -------------------------------------------------------------------------- */
/* In version 3.0.0 of CSparse, "int" is no longer used.  32-bit MATLAB is
   becoming more rare, as are 32-bit computers.  CSparse now uses "csi" as its
   basic integer, which is ptrdiff_t by default (the same as mwSignedIndex in a
   MATLAB mexFunction).  That makes the basic integer 32-bit on 32-bit
   computers and 64-bit on 64-bit computers.  It is #define'd below, in case
   you wish to change it to something else (back to "int" for example).  You
   can also compile with -Dcsi=int (or whatever) to change csi without editting
   this file. */


/* --- primary CSparse routines and data structures ------------------------- */


#ifndef csi
	#define csi long long int
	#define pxi float
#endif

typedef struct cs_sparse    /* matrix in compressed-column or triplet form */
{
    csi nzmax ;     /* maximum number of entries */
    csi m ;         /* number of rows */
    csi n ;         /* number of columns */
    int *p ;        /* col indices (size nzmax) */
    csi *i ;        /* row indices, size nzmax */
    pxi *x ;     /* numerical values, size nzmax */
    csi nz ;        /* # of entries in triplet matrix, -1 for compressed-col */
} cs ;

/* allocate a sparse matrix (triplet form or compressed-column form) */
cs *cs_spalloc (csi m, csi n, csi nzmax);
/* free a sparse matrix */
cs *cs_spfree (cs *A);
/* add an entry to a triplet matrix; return 1 if ok, 0 otherwise */
csi cs_entry (cs *T, csi i, int j, pxi x);
/* wrapper for malloc */
void *cs_malloc (csi n, size_t size);
/* wrapper for calloc */
void *cs_calloc (csi n, size_t size);
/* wrapper for free */
void *cs_free (void *p);
/* wrapper for realloc */
void *cs_realloc (void *p, csi n, size_t size, csi *ok);


#define CS_MAX(a,b) (((a) > (b)) ? (a) : (b))
#define CS_MIN(a,b) (((a) < (b)) ? (a) : (b))
#define CS_FLIP(i) (-(i)-2)
#define CS_UNFLIP(i) (((i) < 0) ? CS_FLIP(i) : (i))
#define CS_MARKED(w,j) (w [j] < 0)
#define CS_MARK(w,j) { w [j] = CS_FLIP (w [j]) ; }
#define CS_CSC(A) (A && (A->nz == -1))
#define CS_TRIPLET(A) (A && (A->nz >= 0))

#endif /* CSPARSE_H_ */


#ifdef __cplusplus
}
#endif
