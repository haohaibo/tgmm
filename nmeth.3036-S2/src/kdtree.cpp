/*
 * kdtree.cpp
 *
 *  Created on: Oct 5, 2010
 *      Author: amatf
 *
 *
 */
#include "kdtree.h"

template <typename T>
KDTree<T>::KDTree()
{
	dims=0;
	needRebalance=true;
};

template <typename T>
KDTree<T>::KDTree(int dims_)
{
	dims=dims_;
	needRebalance=true;
};
template <typename T>
KDTree<T>::KDTree(const KDTree& p)
{
	dims=p.dims;
	radius=p.radius;
	needRebalance=p.needRebalance;
	clear_nearestQueue();//VIP: PRIORITY QUEUE JUST GETS EMPTIED!!!!NOT COPIED11 IT SHOULD BE A TEMPORARY STRUCTURE ANYWAYS
}

template <typename T>
KDTree<T>::~KDTree()
{
	clear_nearestQueue();
};

template <typename T>
KDTree<T>& KDTree<T>::operator = (const KDTree<T> &p)
{
	if (this != &p) // protect against invalid self-assignment
	{
		dims=p.dims;
		radius=p.radius;
		needRebalance=p.needRebalance;
		clear_nearestQueue();//VIP: PRIORITY QUEUE JUST GETS EMPTIED!!!!NOT COPIED11 IT SHOULD BE A TEMPORARY STRUCTURE ANYWAYS
	}
	// by convention, always return *this
	return *this;
}



//recursive routine to build balanced tree
//the most expensive operation is sort to compute median

//According to C++ STL
//Approximately N*logN comparisons on average (where N is last-first).
//In the worst case, up to N2, depending on specific sorting algorithm used by library implementation.
//So probably this part could be sped up
template <typename T>
void KDTree<T>::buildSubKDTree(vector<T*> &blobs,int start, int end, int axis)
{
	sort(blobs.begin()+start, blobs.begin()+end+1, sort_func(axis));

	int midpoint = (end-start)/2 + start;

	if (start<midpoint-1) // Build left tree
		buildSubKDTree(blobs,start,midpoint-1,(axis+1)%dims);
	if (end>midpoint+1) // Build right tree
		buildSubKDTree(blobs,midpoint+1, end,(axis+1)%dims);
}

template <typename T>
void KDTree<T>::balanceTree(vector<T*> &blobs)
{
	if(!needRebalance) return;
	buildSubKDTree(blobs,0,blobs.size()-1,0);
	needRebalance=false;
}

template <typename T>
void KDTree<T>::findSubNearest(vector<T*> &blobs,const float *position, const unsigned int count, const int start, const int end, const int axis,const float max_radius)
{

	//Calculate current point: go down the tree from root
	int midpoint = (end-start)/2 + start;

	//if radius is small, we stop looking down the tree very soon
	if (blobs[midpoint]->center[axis]-radius<=position[axis]) // Look in right subtree
		if (end>=midpoint+1)
			findSubNearest(blobs,position, count, midpoint+1, end, (axis+1)%dims,max_radius);

	if (blobs[midpoint]->center[axis]+radius>=position[axis]) // Look in left subtree
		if (start<=midpoint-1)
			findSubNearest(blobs,position, count, start, midpoint-1, (axis+1)%dims,max_radius);

	//Add this blob to the priority queue if close enough
	float dist = blobs[midpoint]->distBlobs(position);//we do not account for different time frames

	T *qBlob;
	if (nearest_queue.size()>=count)
	{
		qBlob = nearest_queue.top();
		radius=qBlob->dist;
		if (dist<qBlob->dist) { // Replace top blob: dist<max_radius is guaranteed since nearest_queue.top()->dist<max_radius
			nearest_queue.pop();
			radius = dist;
			qBlob = blobs[midpoint];
			qBlob->dist = dist;
			nearest_queue.push(qBlob);
		}
	} else {
		if(dist<max_radius)
		{
			qBlob = blobs[midpoint];
			qBlob->dist = dist;
			nearest_queue.push(qBlob);
		}
	}
}

template <typename T>
void KDTree<T>::findNearest(vector<T*> &blobs,const float *position, const unsigned int count, const float max_radius, float *_radius)
{
	if(needRebalance) balanceTree(blobs);
	radius=max_radius;
	clear_nearestQueue();
	findSubNearest(blobs,position, count, 0, blobs.size()-1,0,max_radius);
	*_radius=radius;
	return;
}


