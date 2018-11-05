/*
 * See license.txt for full license and copyright notice.
 *
 * \brief Lineage as a binary tree. Each node in the tree is a pointer to a
 * nuclei (we keep as a template to be able to handle any sort of lineage
 * object)
 *
 */

#ifndef __LINEAGE_H__
#define __LINEAGE_H__

#include <iostream>
#include <list>
#include "../constants.h"
#include "binaryTree.h"

using namespace std;

// typedef declarations to build hierarchical tree
class nucleus;  // forward declaration
typedef list<nucleus>::iterator ChildrenTypeLineage;

// class to store a lineage as a binary tree
class lineage {
 public:
  BinaryTree<ChildrenTypeLineage> bt;  // main data structure to store lineage

  // constructor/destructor
  lineage();
  ~lineage();
  lineage(const lineage& p);

  // operators
  lineage& operator=(const lineage& p);

  // statistics
  int daughterLengthToDeath(TreeNode<ChildrenTypeLineage>* rootSplit)
      const;  // length in time points before ONE of the daughters from
              // rootSplit dies. If none of the daughters dies -> we assign -1
  void daughterLengthToDeathAll(
      vector<int>& ll) const;  // summarizes daughterLengthToDeath for all
                               // splits in a binary tree
  int daughterLengthToDivision(TreeNode<ChildrenTypeLineage>* rootSplit,
                               bool Left) const;  // length between two
                                                  // divisions. if Left ==
                                                  // true->check for left side.
                                                  // Returns <0 if cell dies
                                                  // before dividing again (the
                                                  // negative number is the
                                                  // length until death)
  void daughterLengthToDivisionAll(
      vector<int>& ll) const;  // summarizes daughterLengthToDivision for all
                               // splits in a binary tree

 protected:
 private:
  TreeNode<ChildrenTypeLineage>* CopyPartialLineage(
      TreeNode<ChildrenTypeLineage>* root,
      TreeNode<ChildrenTypeLineage>* parent, int boundsTM);
};

#endif
