/*	set_unionC.c
 *
 * Implementation of a heap / priority queue abstract data type.
 *
 * by: Steven Skiena
 * begun: March 27, 2002
 */


/*
 * Copyright 2003 by Steven S. Skiena; all rights reserved.
 *
 * Permission is granted for use in non-commerical applications
 * provided this copyright notice remains intact and unchanged.
 *
 * This program appears in my book:
 *
 * "Programming Challenges: The Programming Contest Training Manual"
 * by Steven Skiena and Miguel Revilla, Springer-Verlag, New York 2003.
 *
 * See our website www.programming-challenges.com for additional information.
 *
 * This book can be ordered from Amazon.com at
 *
 * http://www.amazon.com/exec/obidos/ASIN/0387001638/thealgorithmrepo/
 *
 */


#include "set_union.h"
#include "bool.h"
#include <stdio.h>
#include <stdlib.h>



/*
 //initialize each element to 1 object
void set_union_init(set_unionC *s, labelType n)
{
    labelType i;				// counter 
    
    for (i=0; i<n; i++) {
        s->p[i] = i;
        s->size[i] = 1;
    }
    
    s->n = n;
}
*/

labelType *temp_q;

//initialize all elements to null assignment
void set_union_init(set_unionC *s, labelType n)
{
    labelType i;				// counter 
    
    //allocate memory
    s->p=(labelType*)malloc(n*sizeof(labelType));
    s->size=(labelType*)malloc(n*sizeof(labelType));
    s->fMax=(imgVoxelType*)malloc(n*sizeof(imgVoxelType));
 
    temp_q = (labelType*)malloc(n*sizeof(labelType));
    
    for (i=0; i<n; i++) {
        s->p[i] = -1;
        s->size[i] = 0;
        //s->fMax[i]=-1e300;//we do not need to initialize it
    }
    
    s->n = n;
}

void set_union_destroy(set_unionC *s)
{
    free(s->p);
    free(s->size);
    free(s->fMax);
}
        
        
//labelType find(const set_unionC *s, labelType x)
//{
	/*
    if(s->p[x]<0)
	{
		//printf("I am here!\n");
        return -1;//elements might not be assigned
	}
	*/
//    if (s->p[x] == x)
//        return(x);
//    else
//        return( find(s,s->p[x]) );
//}


//labelType find(const set_unionC *s, labelType x)
//{
	
   /* if(s->p[x]<0)
	{
		//printf("I am here!\n");
        return -1;//elements might not be assigned
	}
    */
	
//    if(s->p[x] == x)
//        return x;
 
//    while(s->p[x] != x)
//    {
//	x = s->p[x];
//	if(s->p[x] == x)
//	   return(x);
//    }
//   // printf("go into find x=%d, s->p[x]=%d\n",x,s->p[x]);
//}
//labelType find(const set_unionC *s, labelType x)
//{
//  /*
//  if(s->p[x]<0)
//  {
//  	//printf("I am here!\n");
//      return -1;//elements might not be assigned
//  }
//  */
//
//   if(s->p[x] == x)
//       return x;
//   return s->p[x] = find(s, s->p[x]); 
//  
// // printf("go into find x=%d, s->p[x]=%d\n",x,s->p[x]);
//}

labelType find(const set_unionC *s, labelType x)
{
  /*
  if(s->p[x]<0)
  {
  	//printf("I am here!\n");
      return -1;//elements might not be assigned
  }
  */

   if(s->p[x] == x)
       return x;
  //return s->p[x] = find(s, s->p[x]); 

   int count = 0;
  
   while(s->p[x] != x)
   {
       temp_q[count++] = x;
       //printf("count = %d, s = %d\n", count, temp_q[count - 1]);
       x = s->p[x];   
       //printf("go in while\n");
       if(s->p[x] == x)
       {
           int root = x;
           if(count > 10)
             printf("current root is %d, tree depth is %d\n",x,count);
           while(count > 0)
           {
               //printf("count = %d, s = %d\n", count, temp_q[count - 1]);
               s->p[temp_q[--count]] = root;
           }
           //puts("233");
           return x;
       }
   }
  
  
 // printf("go into find x=%d, s->p[x]=%d\n",x,s->p[x]);
}

imgVoxelType get_fMax(set_unionC *s, labelType x)
{
	/*
    if(s->p[x]<0)
	{
		//printf("I am here 2\n");
        return imgVoxelMinValue;
	}
	*/
    return s->fMax[find(s,x)];
}

void find_and_get_fMax(set_unionC *s, labelType x, labelType* root, imgVoxelType* fMax_)//combines both into one to avoid double callin
{
	*root = find(s,x);

	/*
	if( *root == -1)
	{
		*fMax_ = imgVoxelMinValue;
		printf("I am here 3\n");
	}
	else
	*/
		*fMax_ = s->fMax[ *root ];
}


void union_sets(set_unionC *s, labelType s1, labelType s2)
{
    labelType r1, r2;			/* roots of sets */
    
    r1 = find(s,s1);
    r2 = find(s,s2);
    
   // printf("s1=%d r1=%d s2=%d r2=%d\n",s1,r1,s2,r2);
    
    if (r1 == r2 ) return;		/* already in same set */

	//if( r1 < 0 || r2 <0 )
	//	printf("I am here 4\n");
    
    if (s->size[r1] >= s->size[r2]) {
        s->size[r1] = s->size[r1] + s->size[r2];
        s->p[ r2 ] = r1;
        s->fMax[r1]=fmax(s->fMax[r1],s->fMax[r2]);
    }
    else {
        s->size[r2] = s->size[r1] + s->size[r2];
        s->p[ r1 ] = r2;
        s->fMax[r2]=fmax(s->fMax[r1],s->fMax[r2]);
    }
}

//xParent should have a label and s->p[x] should be -1
void add_element_to_set(set_unionC *s, labelType xParent, labelType x,imgVoxelType fVal)
{
    labelType r;			/* roots of sets */
    
    r = find(s,xParent);
    
    if (s->p[x]>=0) 
    {
        printf("ERROR:You cannot add an elements that has already been assigned\n");		/* already in same set */
        exit(2);
    }
    
    s->size[r]++;
    s->p[x]=r;
    s->fMax[r]=fmax(s->fMax[r],fVal);
}

//you should make sure s->p[x]<-1 before calling this function
void add_new_component(set_unionC *s,labelType x,imgVoxelType fVal)
{
    s->p[x]=x;
    s->size[x]=1;
    s->fMax[x]=fVal;
}

boolC same_component(set_unionC *s, labelType s1, labelType s2)
{
    labelType r1, r2;			/* roots of sets */
    
    r1 = find(s,s1);
    r2 = find(s,s2);
    //if(r1<0 || r2<0)
     //   return FALSE;
    //else
        return ( r1==r2 );
}



void print_set_union(set_unionC *s)
{
    labelType i;                          /* counter */
    
    for (i=1; i<=s->n; i++)
        printf("%i  set=%lu size=%lu \n",i,s->p[i],s->size[i]);
    
    printf("\n");
}


