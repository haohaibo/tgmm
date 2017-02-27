/*
 * CSparse.c
 *
 *  Created on: Jul 21, 2011
 *      Author: amatf
 */

#include "CSparse.h"
/* allocate a sparse matrix (triplet form or compressed-column form) */
cs *cs_spalloc (csi m, csi n, csi nzmax)
{
    cs *A = cs_calloc (1, sizeof (cs)) ;    /* allocate the cs struct */
    if (!A) return (NULL) ;                 /* out of memory */
    A->m = m ;                              /* define dimensions and nzmax */
    A->n = n ;
    A->nzmax = nzmax = CS_MAX (nzmax, 1) ;
    A->nz = 0;
    A->p = cs_malloc (nzmax , sizeof (int)) ;
    A->i = cs_malloc (nzmax, sizeof (csi)) ;
    A->x = cs_malloc (nzmax, sizeof (pxi));
    return ((!A->p || !A->i || (!A->x)) ? cs_spfree (A) : A) ;
}

/* free a sparse matrix */
cs *cs_spfree (cs *A)
{
    if (!A) return (NULL) ;     /* do nothing if A already NULL */
    cs_free (A->p) ;
    cs_free (A->i) ;
    cs_free (A->x) ;
    return (cs_free (A)) ;      /* free the cs struct and return NULL */
}

/* add an entry to a triplet matrix; return 1 if ok, 0 otherwise */
csi cs_entry (cs *T, csi i, int j, pxi x)
{
    if ( i < 0 || j < 0) return (0) ;     /* check inputs */
    if (T->nz >= T->nzmax) return (0) ;
    if (T->x) T->x [T->nz] = x ;
    T->i [T->nz] = i ;
    T->p [T->nz++] = j ;
    T->m = CS_MAX (T->m, i+1) ;
    T->n = CS_MAX (T->n, j+1) ;
    return (1) ;
}

/* wrapper for malloc */
void *cs_malloc (csi n, size_t size)
{
    return (malloc (CS_MAX (n,1) * size)) ;
}

/* wrapper for calloc */
void *cs_calloc (csi n, size_t size)
{
    return (calloc (CS_MAX (n,1), size)) ;
}

/* wrapper for free */
void *cs_free (void *p)
{
    if (p) free (p) ;       /* free p if it is not already NULL */
    return (NULL) ;         /* return NULL to simplify the use of cs_free */
}

/* wrapper for realloc */
void *cs_realloc (void *p, csi n, size_t size, csi *ok)
{
    void *pnew ;
    pnew = realloc (p, CS_MAX (n,1) * size) ; /* realloc the block */
    *ok = (pnew != NULL) ;                  /* realloc fails if pnew is NULL */
    return ((*ok) ? pnew : p) ;             /* return original p if failure */
}
