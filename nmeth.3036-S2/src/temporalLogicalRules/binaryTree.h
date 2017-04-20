/*
 * Copyright (C) 2011-2012 by  Fernando Amat
 * See license.txt for full license and copyright notice.
 *
 * Authors: Fernando Amat based on description at 
 * http://library.thinkquest.org/C005618/text/binarytrees.htm
 *
 * bynaryTree.h
 *
 *  Created on: August 17th, 2012
 *      Author: Fernando Amat
 *
 * \brief Template class to implement binary trees. This implementation contains 
 * specific functions for cell lineaging
 *
 *  Modified by: Haibo Hao
 *         Date: 2017.4.12
 *        Email: haohaibo@ncic.ac.cn 
 */

#ifndef __BINARY_TREE_LINEAGE_H__
#define __BINARY_TREE_LINEAGE_H__

#include <iostream>
#include <queue>


using namespace std;

//NOTE: Many books make a class for a single node, and use it to implement the tree. 
//However, we will separate the structure for each node and the entire tree to conserve
//overhead processing time. Each time a node is created, much less time and memory is 
//used than when a whole tree structure is made. Each node will store a value, and pointers 
//to its children and parent. These will be used and modified by the general tree class.
template <class ItemType>
struct TreeNode
{
   //If you want to use a pointer
   //(so data is not copied all the time, define ItemType as a pointer type)
   ItemType data;
   TreeNode<ItemType> *left;
   TreeNode<ItemType> *right;
   TreeNode<ItemType> *parent;
   //some times it is necessary to know location
   int nodeId;


   	TreeNode()
	{
		left = NULL;
		right = NULL;
		parent = NULL;
	}
   //simple functions
	unsigned int getNumChildren() const
	{
		unsigned int count = 0;
		if(left != NULL) count++;
		if(right != NULL) count++;
		return count;
	}
};


template <class ItemType>
class BinaryTree
{
public:
   //create empty tree with default root node which has no value. 
   //set current to main root node.
   BinaryTree(); 
   BinaryTree(TreeNode<ItemType>*,int); //create new tree with passed node as the new main root. set current to main root. if the second parameter is 0, the new object simply points to the node of the original tree. If the second parameter is 1, a new copy of the subtree is created, which the object points to.
   BinaryTree(const BinaryTree &other);
   ~BinaryTree();
   void insert(const ItemType&, int); //insert new node as child of current. 0=left 1=right
   TreeNode<ItemType>* insert(const ItemType&); //insert new node as child of current. It tries left first and right second. Returns NULL pointer if it was not possible to insert item
   void remove(TreeNode<ItemType>*); //delete node and its subtree

   ItemType& value() const; //return a pointer to teh current item

   //navigate the tree
   void left();
   void right();
   void parent();
   void reset(); //go to main_root
   void SetCurrent(TreeNode<ItemType>*);
   void SetMainRoot(TreeNode<ItemType>*);
   void SetMainRootToNULL();

   //return subtree (node) pointers
   TreeNode<ItemType>* pointer_left() const;
   TreeNode<ItemType>* pointer_right() const;
   TreeNode<ItemType>* pointer_parent() const;
   TreeNode<ItemType>* pointer_current() const;
   TreeNode<ItemType>* pointer_mainRoot() const;

   //return values of children and parent without leaving current node
   ItemType& peek_left() const;
   ItemType& peek_right() const;
   ItemType& peek_parent() const;

   //print the tree or a subtree. only works if ItemType is supported by << operator
   void DisplayInorder(TreeNode<ItemType>*) const;
   void DisplayPreorder(TreeNode<ItemType>*) const;
   void DisplayPostorder(TreeNode<ItemType>*) const;

   //find element based on data pointer. Returns false if element was not found; otherwise current points to element
   bool findTreeNodeBFS(const ItemType& d, TreeNode<ItemType>* root = NULL);//breadth-first-search. Default searches from root of the tree. ItemType needs to have an == operator
   bool findTreeNodeBFSforPointers(const ItemType& d, TreeNode<ItemType>* root = NULL);//breadth-first-search. Default searches from root of the tree. *ItemType needs to have an == operator

   void traverseBinaryTreeBFS(vector< TreeNode<ItemType>* > &vecTreeNodes, TreeNode<ItemType>* root = NULL);//returns all the nodes in the tree using breadth-first-search

   //delete all nodes in the tree
   void clear();

   //simple functions
   bool IsEmpty() const;
   bool IsFull() const;

   //operators
   //comparison operator to be able to order groups of lineages (it depends on the < from the class ItemType 
   bool operator< (BinaryTree<ItemType> const& other) const;
   BinaryTree<ItemType>& operator= (BinaryTree<ItemType> const& other);

private:
   TreeNode<ItemType>* current;
   TreeNode<ItemType>* main_root;
   //create a new copy of a subtree if passed to the constructor
   TreeNode<ItemType>* CopyTree(TreeNode<ItemType>*,TreeNode<ItemType>*) const; 
   //does it reference a part of a larger object?
   bool subtree; 
};

//=================================================================================
//The first constructor simply sets the main_root and current data members to NULL,
//since the tree has no nodes. A new tree is made, therefore it is not part of a 
//larger tree object, and the subtree value is set accordingly.
template <class ItemType>
BinaryTree<ItemType>::BinaryTree()
{
   //create a root node with no value
   main_root = NULL;
   current = NULL;
   subtree = false;
};

//================================================================================== 
//The second constructor accepts a pointer to a node, and creates a new tree object with the node that is passed acting as the new tree's main root. current is then set to the main root. The second parameter specifies whether the new subtree object points directly to the original tree's nodes (the root and its decedents), or creates a copy of the subtree and is thus a new tree. The subtree variable specifies if the subtree points directly to the original tree's nodes. As you will later find out, this is important in the class destructor.
template <class ItemType>
BinaryTree<ItemType>::BinaryTree(TreeNode<ItemType>* root, int op)
{
   if(op == 0)
   {
      main_root = root;
      current = root;
      subtree = true;
   }
   else
   {
      main_root = CopyTree(root,NULL);
      current = main_root;
      subtree = false;
   }
};

//========================================================================
template <class ItemType>
BinaryTree<ItemType>::BinaryTree (const BinaryTree<ItemType> &other)
{
	main_root = CopyTree(other.pointer_mainRoot(),NULL);
	current = main_root;
	subtree = false;
}

//========================================================================
template <class ItemType>
inline BinaryTree<ItemType>& BinaryTree<ItemType>::operator= (BinaryTree<ItemType> const& other)
{
	if(&other != this)
	{
		main_root = CopyTree(other.pointer_mainRoot(),NULL);
		current = main_root;
		subtree = false;
	}
	return *this;
}

//=================================================================================================================== 
//The CopyTree() function creates a copy of subtree root and returns a pointer to the location of the new copy's root node. The second parameter is a pointer to the parent of the subtree being passed. Since CopyTree() uses recursion to traverse the original tree, passing each node's parent as a parameter is the most efficient way of assigning each new node's parent value. Since the parent of the main root is always NULL, we pass NULL as the second parameter in the class constructor above.
template <class ItemType>
TreeNode<ItemType>* BinaryTree<ItemType>::CopyTree(TreeNode<ItemType> *root, TreeNode<ItemType> *parent) const
{
   if(root == NULL) //base case - if the node doesn't exist, return NULL.
      return NULL;
   TreeNode<ItemType>* tmp = new TreeNode<ItemType>; //make a new location in memory
   tmp->data = root->data; //make a copy of the node's data
   tmp->parent = parent; //set the new node's parent
   tmp->left = CopyTree(root->left,tmp); //copy the left subtree of the current node. pass the current node as the subtree's parent
   tmp->right = CopyTree(root->right,tmp); //do the same with the right subtree
   return tmp; //return a pointer to the newly created node.
};



//=================================================================================================================== 
//The job of the class destructor is to delete all the nodes, and free up memory as usual. The clear() function is called just as in the previous data structure implementations. However, this operation is only performed if the object is a main tree. If the object is a subtree that points to the nodes of a larger tree, it will be deleted when the main tree itself is deleted. Attempting to delete the data in the memory associated with the subtree after it has already been deleted by the main tree will have unpredictable results.
template <class ItemType>
BinaryTree<ItemType>::~BinaryTree()
{
   if(!subtree)
      clear(); //delete all nodes
};

//=================================================================================================================== 
//The insert() function creates a new node as a child of current. The first parameter is a value for the new node, and the second parameter is an integer indicating what child the new node will become. A value of 0 indicates that the new node will be a left child of current, whereas a value of 1 indicates the new node will be a right child. If a node already exists in the location that programmer wishes to insert it, that node adopts the value passed to insert(). If the tree does not have any nodes, the second parameter is disregarded, and a main root is created.
template <class ItemType>
void BinaryTree<ItemType>::insert(const ItemType &item,int pos) //insert as child of current 0=left 1=right. if item already exists, replace it
{
   //assert(!IsFull());
   //if the tree has no nodes, make a root node, disregard pos.
   if(main_root == NULL)
   {
      main_root = new TreeNode<ItemType>;
      main_root->data = item;
      main_root->left = NULL;
      main_root->right = NULL;
      main_root->parent = NULL;
      current = main_root;
      return; //node created, exit the function
   }

   if(pos == 0) //new node is a left child of current
   {
      if(current->left != NULL) //if child already exists, replace value
         (current->left)->data = item;
      else
      {
         current->left = new TreeNode<ItemType>;
         current->left->data = item;
         current->left->left = NULL;
         current->left->right = NULL;
         current->left->parent = current;
      }
   }
   else //new node is a right child of current
   {
      if(current->right != NULL) //if child already exists, replace value
         (current->right)->data = item;
      else
      {
         current->right = new TreeNode<ItemType>;
         current->right->data = item;
         current->right->left = NULL;
         current->right->right = NULL;
         current->right->parent = current;
      }
   }
};


//==============================================================================================================
template <class ItemType>
TreeNode<ItemType>* BinaryTree<ItemType>::insert(const ItemType &item)
{
   //assert(!IsFull());
   //if the tree has no nodes, make a root node, disregard pos.
   if(main_root == NULL)
   {
      main_root = new TreeNode<ItemType>;
      main_root->data = item;
      main_root->left = NULL;
      main_root->right = NULL;
      main_root->parent = NULL;
      current = main_root;
      return current ; //node created, exit the function
   }

   if(current->left == NULL) //if child does not exist, insert her
   {
      
         current->left = new TreeNode<ItemType>;
         current->left->data = item;
         current->left->left = NULL;
         current->left->right = NULL;
         current->left->parent = current;
		 return current->left;
      
   }else if(current->right == NULL) //if child does not exist, insert here in the right side
   {
      
         current->right = new TreeNode<ItemType>;
         current->right->data = item;
         current->right->left = NULL;
         current->right->right = NULL;
         current->right->parent = current;      
		 return current->right;
   }else{//node is already full. Thus return an error

	   cout<<"ERROR: at BinaryTree<ItemType>::insert(const ItemType *item): node already has 2 children"<<endl;
	   return NULL;
   }
   return 0;
};

//=================================================================================================================== 
//The remove() function removes the subtree referenced to by root, as well as the root node itself. Depending on whether it was a left or right child, the left or right pointer of the parent is set to NULL. The function uses recursion to perform the necessary operation on all nodes of the subtree. We must start with the nodes on the lowest level, and work our way up. If we were to delete the top level nodes first, we would loose the link the lower levels.
template <class ItemType>
void BinaryTree<ItemType>::remove(TreeNode<ItemType>* root)
{
   if(root == NULL) //base case - if the root doesn't exist, do nothing
      return;
   remove(root->left); //perform the remove operation on the nodes left subtree first
   remove(root->right); //perform the remove operation on the nodes right subtree first
   if(root->parent == NULL) //if the main root is being deleted, main_root must be set to NULL
      main_root = NULL;
   else
   {
      if(root->parent->left == root) //make sure the parent of the subtree's root points to NULL, since the node no longer exists
         root->parent->left = NULL;
      else
         root->parent->right = NULL;
   }
   current = root->parent; //set current to the parent of the subtree removed.
   delete root;//data is deallocated
};

//=================================================================================================================== 
template <class ItemType>
inline ItemType& BinaryTree<ItemType>::value() const
{
   return ( current->data );
};

//=================================================================================================================== 
//This is very helpful if the programmer would like to work with subtrees within the main tree object. Note, the SetCurrent() function should be used with caution. If a pointer is supplied to a node that is not within the tree, the results are unpredictable.

template <class ItemType>
inline void BinaryTree<ItemType>::left()
{
   current = current->left;
};

template <class ItemType>
inline void BinaryTree<ItemType>::right()
{
   current = current->right;
};

template <class ItemType>
inline void BinaryTree<ItemType>::parent()
{
   current = current->parent;
};

template <class ItemType>
inline void BinaryTree<ItemType>::reset()
{
   current = main_root;
};

template <class ItemType>
inline void BinaryTree<ItemType>::SetCurrent(TreeNode<ItemType>* root)
{
   current = root;
};


template <class ItemType>
inline void BinaryTree<ItemType>::SetMainRoot(TreeNode<ItemType>* root)
{
	if(main_root != NULL)
	{
		cout<<"ERROR: at BinaryTree<ItemType>::SetMainRoot(TreeNode<ItemType>* root): main_root is not NULL. You cannot use this function in this case"<<endl;
	}
	main_root = root;
};

template <class ItemType>
inline void BinaryTree<ItemType>::SetMainRootToNULL()
{
	main_root = NULL;
};

//=================================================================================================================== 
//The four functions that follow return pointers to various nodes in the tree, depending on current. This is a required parameter for a few of our other functions, such as remove() and the three display functions. It is also used by one of our class constructors, which can make a new tree object from a subtree. The only function that is required is pointer_current(), since the programmer can navigate the tree to any node. The other three functions were also included for ease of use. It is often times necessary to perform an operation on a node's children or parent without leaving the node. The functions are also useful if a programmer would like to work on a subtree. An external TreeNode* pointer can be created, set by one of the pointer returning functions, and then passed to the operation functions of the class.
template <class ItemType>
inline TreeNode<ItemType>* BinaryTree<ItemType>::pointer_left() const
{
   return current->left;
};

template <class ItemType>
inline TreeNode<ItemType>* BinaryTree<ItemType>::pointer_right() const
{
   return current->right;
};

template <class ItemType>
inline TreeNode<ItemType>* BinaryTree<ItemType>::pointer_parent() const
{
   return current->parent;
};

template <class ItemType>
inline TreeNode<ItemType>* BinaryTree<ItemType>::pointer_current() const
{
   return current;
};

template <class ItemType>
inline TreeNode<ItemType>* BinaryTree<ItemType>::pointer_mainRoot() const
{
   return main_root;
};

//=================================================================================================================== 
template <class ItemType>
inline ItemType& BinaryTree<ItemType>::peek_left() const
{
   //assert(current->left != NULL);
   return current->left->data;
};

template <class ItemType>
inline ItemType& BinaryTree<ItemType>::peek_right() const
{
   //assert(current->right != NULL);
   return current->right->data;
};

template <class ItemType>
inline ItemType& BinaryTree<ItemType>::peek_parent() const
{
   //assert(current->parent != NULL);
   return current->parent->data;
};

//=================================================================================================================== 
//The display functions as explained above are next. Note, these functions will work only if ItemType is supported by the << operator. For instance, any simple built in C/C++ type (such as int, float, char, etc.) will work without any modification. 
template <class ItemType>
void BinaryTree<ItemType>::DisplayInorder(TreeNode<ItemType>* root) const
{
   if (root == NULL)
      return;

   DisplayInorder(root->left);
   cout << (root->data) <<endl;
   DisplayInorder(root->right);
};

template <class ItemType>
void BinaryTree<ItemType>::DisplayPreorder(TreeNode<ItemType>* root) const
{
   if (root == NULL)
      return;

   cout << (root->data) <<endl;
   DisplayInorder(root->left);
   DisplayInorder(root->right);
};

template <class ItemType>
void BinaryTree<ItemType>::DisplayPostorder(TreeNode<ItemType>* root) const
{
   if (root == NULL)
      return;

   DisplayInorder(root->left);
   DisplayInorder(root->right);
   cout << (root->data) <<endl;
};

//===================================================================================================
template <class ItemType>
bool BinaryTree<ItemType>::findTreeNodeBFS(const ItemType &d, TreeNode<ItemType>* root)
{
	if(root == NULL)//default value
		root = main_root;

	queue< TreeNode<ItemType>* > q;
	q.push(root);
	TreeNode<ItemType>* aux;
	while(q.empty() == false)
	{
		aux = q.front();
		q.pop();//delete element
		if(aux != NULL)
		{
			if( (aux->data) == d )//element found. ItemType needs to have an == operator
			{
				current = aux;
				return true;
			}
			q.push(aux->left);
			q.push(aux->right);
		}
	}
	//element not found
	return false;
}

//==================================================================================================
template <class ItemType>
void BinaryTree<ItemType>::traverseBinaryTreeBFS(vector< TreeNode<ItemType>* > &vecTreeNodes, TreeNode<ItemType>* root)
{
	vecTreeNodes.clear();
	if(root == NULL)//default value
		root = main_root;

	queue< TreeNode<ItemType>* > q;
	q.push(root);
	TreeNode<ItemType>* aux;
	while(q.empty() == false)
	{
		aux = q.front();
		q.pop();//delete element
		if(aux != NULL)
		{
			vecTreeNodes.push_back(aux);		
			q.push(aux->left);
			q.push(aux->right);
		}
	}
}

//===================================================================================================
template <class ItemType>
bool BinaryTree<ItemType>::findTreeNodeBFSforPointers(const ItemType &d, TreeNode<ItemType>* root)
{
	if(root == NULL)//default value
		root = main_root;

	queue< TreeNode<ItemType>* > q;
	q.push(root);
	TreeNode<ItemType>* aux;
	while(q.empty() == false)
	{
		aux = q.front();
		q.pop();//delete element
		if(aux != NULL)
		{
			//if( (*(aux->data)).isEqual (*d) == true)//element found. (*ItemType) needs to have an isEqual function
			if( (*(aux->data)) == (*d) )//element found. (*ItemType) needs to have an == operator
			{
				current = aux;
				return true;
			}
			q.push(aux->left);
			q.push(aux->right);
		}
	}
	//element not found
	return false;
}


//=================================================================================================================== 
//The clear() function deletes all nodes in the list. This is very easy to do, since we can take advantage of the remove() function, which we has already defined. The remove() functions deletes all nodes of a subtree, as well as the root node. Therefore, we can pass the main root to remove() in order to delete all nodes in the tree.
template <class ItemType>
void BinaryTree<ItemType>::clear()
{
   remove(main_root); //use the remove function on the main root
   main_root = NULL; //since there are no more items, set main_root to NULL
   current = NULL;
};

//=================================================================================================================== 
//The IsEmpty() function works by evaluating main_root. If there aren't any nodes in the tree, main_root points to NULL.
template <class ItemType>
inline bool BinaryTree<ItemType>::IsEmpty() const
{
   return (main_root == NULL);
};

//=================================================================================================================== 
//Finally, other than the data types, the implementation of the IsFull() function does not change from previous classes.
template <class ItemType>
inline bool BinaryTree<ItemType>::IsFull() const
{
   TreeNode<ItemType> *tmp = new TreeNode<ItemType>;
   if(tmp == NULL)
      return true;
   else
   {
      delete tmp;
      return false;
   }
};
//=================================================================================================================== 
//Now let's take a look at two additional functions, which are not part of the tree class. Often times it is necessary to know how many nodes are in the list, or how many of them are leafs. One example of when a leaf count is required is in a binary expression tree. Binary expression trees store mathematical expression, for instance, 5*x+7=22. Each character of the expression is represented by one node. They are stored in such a way that the expression can then be displayed using an in-order traversal. Also, pre-order and post-order traversals will display the mathematical expression using prefix and postfix notations. This means an operator stored in a node perform an operation on its two children. In such a setup, all operators are internal nodes, whereas variables and constants are leafs. The code for the NodeCount() and LeafCount() functions is displayed below. Both are very short since recursion is used.
template <class ItemType>
inline int LeafCount(TreeNode<ItemType>* root)
{
   if(root == NULL) //base case - if the node doesn't exist, return 0 (don't count it)
      return 0;
   if((root->left == NULL) && (root->right == NULL)) //if the node has no children return 1 (it is a leaf)
      return 1;
   return LeafCount(root->left) + LeafCount(root->right); //add the leaf nodes in the left and right subtrees
};

template <class ItemType>
inline int NodeCount(TreeNode<ItemType>* root)
{
   if(root == NULL) //base case - if the return 0 if node doesn't exist (don't count it)
      return 0;
   else
      return 1 + NodeCount(root->left) + NodeCount(root->right); //return 1 for the current node, and add the amount of nodes in the left and right subtree
};


//=======================================================================
template <class ItemType>
inline bool BinaryTree<ItemType>::operator< (BinaryTree<ItemType> const& other) const 
{
	cout<<"WARNING: at BinaryTree<ItemType>::operator< : operator still needs to be implemented"<<endl;
	return this < *other;
};



#endif //__BINARY_TREE_LINEAGE_H__
