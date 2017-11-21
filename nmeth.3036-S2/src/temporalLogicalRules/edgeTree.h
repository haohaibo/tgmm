/*
 *
 * edgeTree.h
 *
 * \brief template to add tree like (parent and children) relationship to a given class
 *
 */


#ifndef __EDGE_TREE_H__
#define __EDGE_TREE_H__

#include <iostream>
#include <vector>
#include "../constants.h"

using namespace std;


template <class ParentType, class ChildrenType>
class edgeTree
{
public:

	edgeTree()
	{
		//parent = ParentType();//default initialization. Not clear what this value is, so I should look for a better strategy to set this
		hasParent_ = false;
	};
	edgeTree(int expectedNumChildren)
	{
		//parent = ParentType();
		hasParent_ = false;
		children.reserve(expectedNumChildren);
	};
	edgeTree(const edgeTree< ParentType,  ChildrenType>& p);
	void reset();//removes all edges from edgeTree
	size_t getNumChildren(){return children.size();};
	void addChild(const ChildrenType &ch){children.push_back(ch);};
	int  deleteChild(const ChildrenType &ch);
	int deleteChild(unsigned int posCh);
	void deleteChildrenAll(){ children.clear(); };
	void setParent(const ParentType &p){parent = p; hasParent_ = true;};
	void deleteParent(){hasParent_ = false;};
	bool getParent(ParentType& p){p = parent;return hasParent_;};
	ParentType& getParent(){return parent;};
	bool hasParent(){return hasParent_;};
	vector< ChildrenType >& getChildren(){return children;};

	//-------------operators-----------------------
	edgeTree<ParentType, ChildrenType>& operator=(const edgeTree<ParentType, ChildrenType>& p);
protected:

private:
	//since we need to coordinate parent with has parent
	ParentType parent;//if you want to use pointers, just declare ParentType as a pointer type
	bool hasParent_;//needed to keep track if parent is valid. YOU SHOULD ALWAYS CHECK IF HASPRENT == TRUE BEFORE USING PARENT
	vector< ChildrenType > children;
};

template <class ParentType, class ChildrenType>
inline int edgeTree<ParentType, ChildrenType>::deleteChild(unsigned int pos)
{
	if(children.size() <= pos)
	{
		cout<<"ERROR: at edgeTree<class ParentType, class ChildrenType>::deleteChild children position is larger than number of choildren "<<endl;
		return 1;
	}

	children.erase(children.begin() + pos);
};
//assignment operator
template <class ParentType, class ChildrenType>
edgeTree<ParentType, ChildrenType>& edgeTree<ParentType,ChildrenType>::operator=(const edgeTree<ParentType, ChildrenType>& p)
{
	if (this != &p)
	{
		parent = p.parent;
		hasParent_ = p.hasParent_;
		children = p.children;
	}
	return *this;
};
//copy constructor
template <class ParentType, class ChildrenType>
edgeTree< ParentType,  ChildrenType>::edgeTree(const edgeTree< ParentType,  ChildrenType>& p) 
{
    parent = p.parent;
	hasParent_ = p.hasParent_;
	children = p.children;
};

template <class ParentType, class ChildrenType>
inline int edgeTree< ParentType,  ChildrenType>::deleteChild(const ChildrenType &ch)
{
	for(typename vector<ChildrenType>::iterator iter = children.begin(); iter != children.end(); ++iter)
	{
		if((*iter) == ch)
		{
			children.erase( iter );
			return 0;
		}
	}

	cout<<"ERROR: at edgeTree<class ParentType, class ChildrenType>::deleteChild children pointer not found "<<endl;
	return 1;
};

template <class ParentType, class ChildrenType>
void edgeTree< ParentType,  ChildrenType>::reset()
{
	hasParent_ = false;
	children.clear();
}

#endif
