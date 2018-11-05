/*	set_unionC.h

        Header file for union-find data structure implementation

        by: Steven Skiena
*/

/*
Copyright 2003 by Steven S. Skiena; all rights reserved.

Permission is granted for use in non-commerical applications
provided this copyright notice remains intact and unchanged.

This program appears in my book:

"Programming Challenges: The Programming Contest Training Manual"
by Steven Skiena and Miguel Revilla, Springer-Verlag, New York 2003.

See our website www.programming-challenges.com for additional information.

This book can be ordered from Amazon.com at

http://www.amazon.com/exec/obidos/ASIN/0387001638/thealgorithmrepo/

*/

#ifndef SET_UNION_H
#define SET_UNION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include "bool.h"
#include "constants.h"

#if defined _WIN32
#define fmax max
#define fmin min
#endif

typedef int64 labelType;  // change this to int if  images are smaller than 2GB

typedef struct {
  /* parent element */
  labelType *p;

  /* number of elements in subtree i.
     Only the root has the right information */
  labelType *size;

  /* maximum value of the function in subtree i.
   Only the root has the right information */
  imgVoxelType *fMax;

  /* number of elements in set */
  labelType n;
} set_unionC;

void set_union_init(set_unionC *s, labelType n);
void set_union_destroy(set_unionC *s);
labelType find(const set_unionC *s, labelType x);
void union_sets(set_unionC *s, labelType s1, labelType s2);
boolC same_component(set_unionC *s, labelType s1, labelType s2);
void print_set_union(set_unionC *s);
void add_new_component(set_unionC *s, labelType x, imgVoxelType fVal);
void add_element_to_set(set_unionC *s, labelType xParent, labelType x,
                        imgVoxelType fVal);
imgVoxelType get_fMax(set_unionC *s, labelType x);
void find_and_get_fMax(set_unionC *s, labelType x, labelType *root,
                       imgVoxelType *fMax_);

#ifdef __cplusplus
}
#endif

#endif  // SET_UNION_H
