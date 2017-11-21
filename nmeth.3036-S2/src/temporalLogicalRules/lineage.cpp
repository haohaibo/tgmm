/*
 *
 * \brief 
 *
 */

#include <stdlib.h> 
#include "lineage.h"

lineage::lineage()
{
	//bt will be initialized with default constructor
}
lineage::~lineage()
{
	bt.clear();
}
lineage::lineage(const lineage& p)
{
	bt = p.bt;
}
lineage& lineage::operator=(const lineage & p)
{
	if(&p != this)
	{
		bt = p.bt;
	}
	return *this;
}

//==================================================================
//--------------------statistics-----------------------------
int lineage::daughterLengthToDeath(TreeNode<ChildrenTypeLineage>* rootSplit) const
{
	if(rootSplit == NULL)
	{
		cout<<"ERROR: at lineage::daughterLengthToDeath: rootSplit node is null"<<endl;
		exit(3);
	}
	if(rootSplit->left == NULL || rootSplit->right == NULL)
	{
		cout<<"ERROR: at lineage::daughterLengthToDeath: rootSplit is not a split"<<endl;
		exit(2);
	}
	
	//calculate length for left branch
	int llAux1 = 1;
	TreeNode<ChildrenTypeLineage>* aux = rootSplit->left;
	while( aux->getNumChildren() == 1)
	{
		if(aux->left != NULL)
			aux = aux->left;
		else
			aux = aux->right;
		llAux1++;
	}
	if( aux->getNumChildren() == 2)//daughter has divided
		llAux1 = 2147483647;//equivalent to infinite for int32

	//calculate length for right branch
	int llAux2 = 1;
	aux = rootSplit->right;
	while( aux->getNumChildren() == 1 && llAux2 < llAux1) //if we overtake llAux1 it does not make sense
	{
		if(aux->left != NULL)
			aux = aux->left;
		else
			aux = aux->right;
		llAux2++;
	}
	
	if(aux ->getNumChildren() == 2)
	{
		if( llAux1 == 2147483647)//both daughters havr divided
			return -1;//both daughters divided
		else return llAux1;
	}else{//right daughter did not reach division
		if( llAux1 == 2147483647)
			return llAux2;
		else return min(llAux1, llAux2);//none of the daughters divided
	}

}
//--------------------------------------------------------------------
void lineage::daughterLengthToDeathAll(vector<int>& ll) const
{
	ll.clear();

	//traverse the tree and for each splitting event we find calculate distance
	queue< TreeNode<ChildrenTypeLineage>* > q;
	q.push( bt.pointer_mainRoot() );
	TreeNode<ChildrenTypeLineage>* aux;

	while( q.empty() == false )
	{
		aux = q.front();
		q.pop();

		if(aux != NULL)
		{
			if(aux->left != NULL) q.push(aux->left);
			if(aux->right != NULL) q.push(aux->right);

			if(aux->getNumChildren() == 2) //split event
				ll.push_back( daughterLengthToDeath(aux) );
		}
	}
}
//--------------------statistics-----------------------------
int lineage::daughterLengthToDivision(TreeNode<ChildrenTypeLineage>* rootSplit, bool Left) const
{
	if(rootSplit == NULL)
	{
		cout<<"ERROR: at lineage::daughterLengthToDivision: rootSplit node is null"<<endl;
		exit(3);
	}
	if(rootSplit->left == NULL || rootSplit->right == NULL)
	{
		cout<<"ERROR: at lineage::daughterLengthToDivision: rootSplit is not a split"<<endl;
		exit(2);
	}
	
	//calculate length
	int llAux1 = 1;
	TreeNode<ChildrenTypeLineage>* aux;
	if( Left == true) aux = rootSplit->left;
	else aux = rootSplit->right;
	while( aux->getNumChildren() == 1)
	{
		if(aux->left != NULL)
			aux = aux->left;
		else
			aux = aux->right;
		llAux1++;
	}
	if( aux->getNumChildren() == 0)//daughter has died
		llAux1 = -llAux1;

	return llAux1;
}

//--------------------------------------------------------------------
void lineage::daughterLengthToDivisionAll(vector<int>& ll) const
{
	ll.clear();

	//traverse the tree and for each splitting event we find calculate distance
	queue< TreeNode<ChildrenTypeLineage>* > q;
	q.push(bt.pointer_mainRoot());
	TreeNode<ChildrenTypeLineage>* aux;

	while( q.empty() == false )
	{
		aux = q.front();
		q.pop();

		if(aux != NULL)
		{
			if(aux->left != NULL) q.push(aux->left);
			if(aux->right != NULL) q.push(aux->right);

			if(aux->getNumChildren() == 2) //split event
			{
				ll.push_back( daughterLengthToDivision(aux, true) );
				ll.push_back( daughterLengthToDivision(aux, false) );
			}
		}
	}
}
