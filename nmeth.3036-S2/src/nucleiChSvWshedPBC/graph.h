/*
 * Copyright (C) 2011-2012 by  Fernando Amat
 * See license.txt for full license and copyright notice.
 *
 * Authors: Fernando Amat 
 *  graph.h
 *
 *  Created on: January 17th, 2013
 *      Author: Fernando Amat
 *
 * \brief Graph data structure based on "The algorithm design book" by Steven S. Skiena
 *
 */


#ifndef __GRAPH_FAMAT_H__
#define __GRAPH_FAMAT_H__


#include <iostream>

using namespace std;

#define NUM_THREADS 8

struct GraphEdge
{
   unsigned int e1, e2;//edge goes from nodes e1 to e2
   GraphEdge *next;
   double weight; //edge weight

   GraphEdge()
   {
	   next = NULL;
   }
   GraphEdge(unsigned int e1_, unsigned int e2_, double weight_)
   {
	   next = NULL;
	   e1 = e1_;
	   e2 = e2_;
	   weight = weight_;
   }
};

template<class ItemType>
class graphFA
{
public:

	//main variables
	vector< ItemType > nodes;//in case you want to hold some information in the node, we made it a template class

    //keeps track of which nodes are active and which ones have been
    //"deleted" (we need to do this because we use indexes to nodes
    //instead of pointers to objects floating in different parts of memory)
	vector< bool > activeNodes;
	vector< GraphEdge* > edges; //adjecincy info for each node (it is a single list node, so for each element in the graph it just has teh first edge). It should be teh same size as nodes.
	//vector<int> degrees;//degrees (number of edges) for each node. It should have the same size as nodes.
	bool isDirected;

	//constructor/destructor
	graphFA();
	graphFA(bool isDirected_);
	~graphFA();

	//basic set/get functions
	size_t getNumNodes();//because deleteNodes just "isolate" nodes (it does not physically delete them from memory because we would have to change all the indexes

	//added by Haibo Hao
	//Openmp version of getNumNodes()
	size_t getNumNodes_Omp();
	//added by Haibo Hao
	
	void reserveNodeSpace(size_t numExpectedNodes)
	{
		nodes.reserve(numExpectedNodes);
		edges.reserve(numExpectedNodes);
		activeNodes.reserve(numExpectedNodes);
	}

	GraphEdge* findEdge(unsigned int e1, unsigned int e2);//finds if edge e1->e2 exists (linear search). Returns NULL if not found
	size_t getNumEdges(unsigned int e1);//returns number of edges for node e1

	//main functions for the graph
	GraphEdge* insert_edge(unsigned int e1, unsigned int e2, double weight,  bool checkDirection = true);//returns NULL if there was an error
	void insert_node(const ItemType& p);//TODO: use activeNodes to avoid creating new spaces if there are inactive nodes


	//other functions
	void mergeNodes(unsigned int e1, unsigned int e2, ItemType& newNode, double (*weightOp) (double, double) );
	void deleteNode(unsigned int e1);
	void deleteEdge(unsigned int e1, unsigned int e2, bool checkDirection = true);



	//debugging functions. Return 0 if no problem
	int debugCheckUnidrectionalConditions();//if it is undirectional graph, all edges should be duplicated
	int debugCheckIfThereAreEdgesToItself();

protected:

private:
};

//==========================================================
template<class ItemType>
graphFA<ItemType>::graphFA()
{
	isDirected = false;
	nodes.reserve(1000);
	edges.reserve(1000);
	activeNodes.reserve(1000);
}

template<class ItemType>
graphFA<ItemType>::graphFA(bool isDirected_)
{
	isDirected = isDirected_;	
	nodes.reserve(1000);
	edges.reserve(1000);
	activeNodes.reserve(1000);
}

template<class ItemType>
graphFA<ItemType>::~graphFA()
{
	GraphEdge *nextAux, *nextAux2;
	for(vector< GraphEdge* >::iterator iter = edges.begin(); iter != edges.end(); ++iter)
	{
		nextAux = (*iter);
		while( nextAux != NULL )
		{
			nextAux2 = nextAux;
			nextAux = nextAux->next;
			delete nextAux2;
		}
	}
}

//=======================================================
template<class ItemType>
inline size_t graphFA<ItemType>::getNumNodes()
{
	size_t count = 0;
	for(size_t ii = 0; ii < activeNodes.size(); ii++)
	{
		if( activeNodes[ii] == true )
			count++;
	}

	return count;
}

template<class ItemType>
inline size_t graphFA<ItemType>::getNumNodes_Omp()
{
	size_t count = 0;

	size_t sizeOfActiveNodes = activeNodes.size();

	//printf("sizeOfActiveNodes =  %d\n",sizeOfActiveNodes);
#pragma omp parallel for reduction(+:count) num_threads(NUM_THREADS)	
	for(size_t ii = 0; ii < sizeOfActiveNodes; ii++)
	{
		if( activeNodes[ii] == true )
			count++;
	}

	return count;
}
//=================================================================

template<class ItemType>
GraphEdge* graphFA<ItemType>::insert_edge(unsigned int e1, unsigned int e2, double weight,  bool checkDirection)
{
	if( e1 >= nodes.size() || e2 >= nodes.size() )
	{
		cout<<"ERROR: at graphFA<ItemType>::insert_edge:"
            <<" requested id for edges is larger than the"
            <<" current number of nodes"
            <<endl;
		return NULL;
	}

	GraphEdge* p = new GraphEdge(e1, e2, weight);
	p->next = edges[e1];

	edges[e1] = p; //insert at head of list

	if( isDirected == false  && checkDirection == true)
		insert_edge(e2,e1,weight, false);

	return p;
}


//=================================================================

template<class ItemType>
void graphFA<ItemType>::insert_node(const ItemType& p)
{
	nodes.push_back(p);
	edges.push_back(NULL);
	activeNodes.push_back(true);
}


//================================================================
template<class ItemType>
inline GraphEdge* graphFA<ItemType>::findEdge(unsigned int e1, unsigned int e2)
{
	if( e1 >= nodes.size() || e2 >= nodes.size() )
		return NULL;

	GraphEdge* p = edges[e1];

	while( p != NULL)
	{
		if( p->e2 == e2 )
			break;
		p = p->next;
	}

	return p;
}

//================================================================
template<class ItemType>
inline size_t graphFA<ItemType>::getNumEdges(unsigned int e1)
{
	if( e1 >= nodes.size() )
		return 0;

	GraphEdge* p = edges[e1];

	size_t numEdges = 0;
	while( p != NULL)
	{
		numEdges++;
		p = p->next;
	}

	return numEdges;
}


//==================================================================
template<class ItemType>
void graphFA<ItemType>::mergeNodes(unsigned int e1, unsigned int e2, ItemType& newNode, double (*weightOp)(double, double))
{

	if( e1 == e2 )
		return;
	//make a copy of node e1 so we can put the new merged node in there
	ItemType e1NodeOld = nodes[e1];
	nodes[e1] = newNode;

	//the edges of e1 are going to be there for sure (they might be modified by e2 but thats all.
	//construct all possible new edges

	GraphEdge* auxEdge = edges[e2], *auxEdge2;
	double auxW;
	while(auxEdge != NULL)
	{
		auxEdge2 = findEdge(e1,auxEdge->e2);
		if( auxEdge2 == NULL )//new edge to be added
		{
			if( auxEdge->e2 != e1 )//to make sure we do not introduce an edge to itself
			{
				auxEdge2 = insert_edge(e1,auxEdge->e2, auxEdge->weight);
				if( isDirected == false )//update the other direction
				{
					auxEdge2 = findEdge(auxEdge2->e2, e1);
					/*
					if( auxEdge2 == NULL )
					{
						cout<<"ERROR: at mergeNodes: graph is undirected but we could not find counterpart (new edge)"<<endl;
						exit(3);
					}else
					*/
					{
						auxEdge2->e2 = e1;
						auxEdge2->weight = auxEdge->weight;
					}
				}
			}

		}else{//edge already exists. We just need to update weights
			auxW = weightOp(auxEdge2->weight, auxEdge->weight);//merge weights
			auxEdge2->weight = auxW;
			if( isDirected == false )//update the other direction
			{
				auxEdge2 = findEdge(auxEdge2->e2,e1);//this edge already existed pointing at e1
				/*
				if( auxEdge2 == NULL )
				{
					cout<<"ERROR: at mergeNodes: graph is undirected but we could not find counterpart (edge already exists)"<<endl;
					exit(3);
				}else
				*/
				{
					auxEdge2->weight = auxW;
				}
			}
		}

		auxEdge = auxEdge->next;
	}
	
	//delete elements
	if( isDirected == true )
	{
		//we need to find all the edges that were pointing towards e2 and make them point towards e1
		for(size_t ii = 0 ; ii < nodes.size(); ii++)
		{
			auxEdge2 = findEdge(ii, e2);
			if( auxEdge2 != NULL )//edge found
			{
				auxEdge2->e2 = e1;
				auxEdge = findEdge(ii, e1);//check if we also have one pointing to e1
				if( auxEdge != NULL)//merge edges
				{
					auxEdge2->weight = weightOp( auxEdge->weight, auxEdge2->weight);
					deleteEdge(ii, e1);
				}
			}
		}
	}
	deleteEdge(e1,e2);
	deleteNode(e2); //this routine deletes all the edges that point toward e2
}


//==================================================================
template<class ItemType>
void graphFA<ItemType>::deleteNode(unsigned int e1)
{
	if( e1 >= nodes.size() )
		return;

	GraphEdge* auxEdge = edges[e1];
	GraphEdge* auxEdge2;
	if( isDirected == false) //easier since we don't have to look all over the graph to find all the edges
	{
		while(auxEdge != NULL )
		{
			deleteEdge(auxEdge->e2, e1, false);

			//delete edge
			auxEdge2 = auxEdge->next;
			delete auxEdge;
			auxEdge = auxEdge2;
		}
	}else{
		while(auxEdge != NULL )
		{
			//delete edge
			auxEdge2 = auxEdge->next;
			delete auxEdge;
			auxEdge = auxEdge2;
		}
		//we have to find all the edges pointing to this node
		for(unsigned int ii = 0 ; ii < nodes.size(); ii++)
		{
			deleteEdge(ii, e1);//if such an edge exists it will delete it
		}
	}

	edges[e1] = NULL;
	activeNodes[e1] = false;//"deactivate" from list
}


//=================================================================
template<class ItemType>
void graphFA<ItemType>::deleteEdge(unsigned int e1, unsigned int e2, bool checkDirection)
{
	if( e1 >= nodes.size() || e2 >= nodes.size() )
		return;

	GraphEdge* auxEdge = edges[e1];
	GraphEdge* auxEdgePrev = NULL;
	while( auxEdge != NULL )
	{

		if( auxEdge->e2 == e2 )//we found the edge
		{
			if( auxEdgePrev != NULL )
				auxEdgePrev->next = auxEdge->next;//skip (e1, e2) edge
			else//we are in the very first item of the list of edges
				edges[e1] = auxEdge->next;

			delete auxEdge;
			break;
		}
		auxEdgePrev = auxEdge;
		auxEdge = auxEdge->next;
	}

	if( isDirected == false && checkDirection == true )//delete the otehr direction
		deleteEdge(e2,e1,false);
}

//======================================================================
template<class ItemType>
int graphFA<ItemType>::debugCheckUnidrectionalConditions()
{
	cout<<"DEBUGGING: debugCheckUnidrectionalConditions: MAKE SURE ALL EDGES ARE DUPLICATED!!!!"<<endl;
	if ( isDirected == true )
		return 0;

	GraphEdge* findEdge;
	for(size_t ii = 0; ii < edges.size(); ii ++)
	{
		GraphEdge* auxEdge = edges[ii];

		while( auxEdge != NULL )
		{
			findEdge = this->findEdge(auxEdge->e2,ii);
			if(  findEdge == NULL )
			{
				cout<<"ERROR: edge "<<ii<<"->"<<auxEdge->e2<<" is not duplicated at the other end"<<endl;
				return 1;
			}else{
				if(findEdge->weight != auxEdge->weight )
				{
					cout<<"ERROR: edge "<<ii<<"->"<<auxEdge->e2<<" have different weights "<<findEdge->weight<<"; "<<auxEdge->weight<<endl;;
					return 2;
				}
			}
			if( auxEdge->e1 != ii )
			{
				cout<<"ERROR: node "<<ii<<" has edge e1 equal to"<<auxEdge->e1<<" (it should be the same!!!!)"<<endl;
				return 3;
			}

			auxEdge= auxEdge->next;
		}
	}
	return 0;
}


//===========================================================
template<class ItemType>
int graphFA<ItemType>::debugCheckIfThereAreEdgesToItself()
{
	cout<<"DEBUGGING: debugCheckIfThereAreEdgesToItself: MAKE SURE THERE AREA NO EDGES TO ITSELF!!!!"<<endl;
	if ( isDirected == true )
		return 0;

	for(size_t ii = 0; ii < edges.size(); ii ++)
	{
		GraphEdge* auxEdge = edges[ii];

		while( auxEdge != NULL )
		{
			if(ii == auxEdge->e2 )
			{
				cout<<"ERROR: edge "<<ii<<"->"<<auxEdge->e2<<" are the same!!"<<endl;
				return 2;
			}
			auxEdge= auxEdge->next;
		}
	}
	return 0;
}













#endif
