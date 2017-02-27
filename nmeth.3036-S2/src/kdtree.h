/*
 * kdtree.h
 *
 *  Created on: Oct 5, 2010
 *      Author: amatf
 *
 *      Based on the code from http://www2.imm.dtu.dk/~bdl/02561/photonmap/
 *
 *	    @warning You have to use #include "kdtree.cpp" to avoid undefined symbol error in C++ linker due to template usage
 *
 *      @warning Template class T needs to have the following elements
 *
 *      @warning float center[dims]
 *      @warning float dist
 *      @warning float T.distBlobs(float position[dims]);
 */

#ifndef KDTREE_H_
#define KDTREE_H_


#include <vector>
#include <algorithm>
#include <queue>
#include <iostream>

using namespace std;

template <typename T>
class KDTree {
private:
	float radius;

	void buildSubKDTree(vector<T*> &blobs,int start, int end, int axis);
	void findSubNearest(vector<T*> &blobs,const float *position, const unsigned int count, const int start, const int end, const int axis,const float max_radius);


	struct sort_func : public binary_function<T, T, bool> {
		int sort_axis;
		sort_func(int axis) : sort_axis(axis) {}
		bool operator()(T* x, T* y) { return  x->center[sort_axis]<y->center[sort_axis];  }
	};

	struct priority_func : public binary_function<T, T, bool> {
		bool operator()(T* x, T* y) {
			return  x->dist<y->dist;  } //top of the queue are the elements with largest distance (descending order)
	};

	int dims;//number of dimensions: needs to be specified at the beginning
public:
	priority_queue<T*, vector<T*>, priority_func> nearest_queue;//contains nearest neighbors
	bool needRebalance;//true->indicates we have added or removed components since the last treerebalancing

	//constructor
		KDTree();
		KDTree(int dims_);
		~KDTree();
		KDTree(const KDTree& p);//VIP: PRIORITY QUEUE JUST GETS EMPTIED!!!!NOT COPIED11 IT SHOULD BE A TEMPORARY STRUCTURE ANYWAYS

	//operators
	KDTree<T> & operator = (const KDTree<T> & p);	//VIP: PRIORITY QUEUE JUST GETS EMPTIED!!!!NOT COPIED11 IT SHOULD BE A TEMPORARY STRUCTURE ANYWAYS

	//basic functions to create a kd-tree
	void balanceTree(vector<T*> &blobs);//balance tree to make search more efficient
	/*
	 * Find nearest neighbors. Nearest neighbors are stored in the priority_queue nearest_queue
	 @param blobs is the vector containing all the points
	@param position is the point where the nearest neighbors should be found. It should have as many dimensions as dim
	@param count is the maximum number of neighbors that should be found.
	@param max_radius is the maximum distance to the count neighbors (Only used to optimize. Set this value to a large number if no optimization is wanted).
	@param radius is a value that this function returns and contains the maximum distance to the count neighbors.
	@out nearest_queue stores the results

	@warning count and max_radius affect the performance a lot. For count>20 and an unconstrained max_radius, it might be better to use brute force because of the slowness of priority_queue.pop() operation
	*/
	void findNearest(vector<T*> &blobs,const float *position, const unsigned int count, const float max_radius, float *radius);//find nearest neighbors
	void clear_nearestQueue();

};

//----------------------------------INLINE FUNCTIONS CALLED FROM OUTSIDE---------------------------------
template <typename T>
inline void KDTree<T>::clear_nearestQueue()
{

	while(!nearest_queue.empty())
	{
		//delete nearest_queue.top();
		nearest_queue.pop();
	}

}

#endif /* KDTREE_H_ */

