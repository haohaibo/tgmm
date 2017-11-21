/*
 * See license.txt for full license and copyright notice.
 *
 * \brief Methods to calculate tracklets from super-voxel decomposition
 *
 */

#include <queue>

#include "trackletCalculation.h"



using namespace std;

int calculateTrackletsWithSparseHungarianAlgorithm(list< supervoxel > &svListT0, int distanceMethod, double thrCost, vector<SibilingTypeSupervoxel>& assignmentId, SibilingTypeSupervoxel* nullAssignment)
{
	//calculate maximum number of edges
	int E = 0;

	//reset all the candidates to setup indexes
	for(list< supervoxel >::iterator iter = svListT0.begin(); iter != svListT0.end(); ++iter)
	{
		for(vector<SibilingTypeSupervoxel>::iterator iterNeigh = iter->nearestNeighborsInTimeForward.begin(); iterNeigh != iter->nearestNeighborsInTimeForward.end(); ++iterNeigh)
		{
			(*iterNeigh)->tempWildcard = -1.0f; //so we know which ones have not been assigned
			E++;
		}
	}

	//setup distance method
	float (supervoxel::*dd_func_ptr)(const supervoxel&) const;//define pointer to function-member class
	
	switch(distanceMethod)
	{
	case 0://Jaccard
		dd_func_ptr = &supervoxel::JaccardDistance;
		break;
	case 1://Euclidean
		dd_func_ptr = &supervoxel::Euclidean2Distance;
		thrCost = thrCost * thrCost;//the distance is calculate without sqrt to save operations
		break;
	default:
		cout<<"ERROR: at calculateTrackletsWithSparseHungarianAlgorithm: distanceMEthod not supported by the code"<<endl;
		return 1;
	}

	//set workspace
	workspaceSparseHungarian* W = (workspaceSparseHungarian*) malloc(sizeof(workspaceSparseHungarian));
	workspaceSpraseHungarian_init(W, svListT0.size(), 0, E, thrCost);//we need to update M after going trhough all the neighbors

	//calculate distances
	int M = 0;
	E = 0;//reset
	float dd;
	W->edgesPtr[0] = 0;
	int countS = 0;
	
	for(list< supervoxel >::iterator iter = svListT0.begin(); iter != svListT0.end(); ++iter)
	{
		for(vector<SibilingTypeSupervoxel>::iterator iterNeigh = iter->nearestNeighborsInTimeForward.begin(); iterNeigh != iter->nearestNeighborsInTimeForward.end(); ++iterNeigh)
		{
			dd = ((*iter).*dd_func_ptr)(*(*iterNeigh));//call the selected distance
			if(dd < thrCost)//better than assigning to Garbage potential
			{
				//check if this supervoxel has already been assigned an id
				if((*iterNeigh)->tempWildcard < 0.0f)
				{
					(*iterNeigh)->tempWildcard = (float)M;
					M++;
				}
				W->edgesId[E] = (int)((*iterNeigh)->tempWildcard);
				W->edgesW[E] = dd;
				E++;
			}
		}
		countS++;
		W->edgesPtr[countS] = E;
	}
	//update number of edges and number of candidates
	W->E = E;
	W->M = M;

	//solve assignment problem
	int* assignmentIdAux = new int[W->N];
	if(W->E == 0)//degenerate case
	{
		for(int ii = 0;ii<W->N;ii++)
			assignmentIdAux[ii] = -1;//garbage potential (no assignment) for all of them
	}
	else{//regular case
		int err = solveSparseHungarianAlgorithm(W, assignmentIdAux);
		if(err > 0)
			return err;

	}
	//remap id to iterators to supervoxels
	countS = 0;
	assignmentId.resize(W->N);
	for(list< supervoxel >::iterator iter = svListT0.begin(); iter != svListT0.end(); ++iter, countS++)
	{
		if(assignmentIdAux[countS] < 0)//garbage potential
		{
			assignmentId[countS] = *nullAssignment;//to signal not assignment!!
			continue;
		}
		dd = (float)(assignmentIdAux[countS]);
		for(vector<SibilingTypeSupervoxel>::iterator iterNeigh = iter->nearestNeighborsInTimeForward.begin(); iterNeigh != iter->nearestNeighborsInTimeForward.end(); ++iterNeigh)
		{
			if( dd == (*iterNeigh)->tempWildcard)
			{
				assignmentId[countS] = (*iterNeigh);
				break;
			}
		}
	}


	//release memory
	delete[] assignmentIdAux;
	workspaceSpraseHungarian_destroy(W);
	free(W);
	
	return 0;
}

//=====================================================================================================
int calculateTrackletsWithSparseHungarianAlgorithm(lineageHyperTree& lht, int distanceMethod, double thrCost, unsigned int numThreads) 
{
	if(numThreads > 0)
	{
		cout<<"WARNING: at calculateTrackletsWithSparseHungarianAlgorithm: code is not multi-threaded yet"<<endl;//TODO
	}
	
	//calculate assignment for each pair of time points. TODO: parallelize this piece of code. Ishould be straight forward
	vector< vector<SibilingTypeSupervoxel> >  assignmentId;//assignmentId[ii] contains assignments from time jj->jj+1
	assignmentId.resize(lht.getMaxTM()-1);

	list< supervoxel> nullAssignmentList;//temporary supervoxel list to simulate null assignment
	nullAssignmentList.push_back(supervoxel());
	SibilingTypeSupervoxel nullAssignment = nullAssignmentList.begin();
	nullAssignment->centroid[0] = -1e32f;//characteristic to find out no assignment
	for(int ii = 0; ii< lht.getMaxTM()-1; ii++)
	{		
		//recalculate nearest neighbors
		lht.supervoxelNearestNeighborsInTimeForward(ii);
		//calculate association
		int err = calculateTrackletsWithSparseHungarianAlgorithm(lht.supervoxelsList[ii], distanceMethod, thrCost, assignmentId[ii], &nullAssignment);
		if (err > 0)
			return err;
	}

	//delete all the lineages
	for(list<lineage>::iterator iterL = lht.lineagesList.begin(); iterL != lht.lineagesList.end(); ++iterL)
	{
		iterL->bt.remove( iterL->bt.pointer_mainRoot() );//deletes all the lineage recursively
	}
	lht.lineagesList.clear();

	//delete all the nuclei
	for( int ii = 0; ii< lht.getMaxTM(); ii++)
	{
		lht.nucleiList[ii].clear();
	}
	//delete all references to nuclei from supervoxels
	for( int ii = 0; ii< lht.getMaxTM(); ii++)
	{
		int count = 0;//so I know index for each superovoxel
		for(list<supervoxel>::iterator iterS = lht.supervoxelsList[ii].begin(); iterS != lht.supervoxelsList[ii].end(); ++iterS, count++)
		{
			iterS->treeNode.deleteParent();
			iterS->tempWildcard = (float)count;			
		}
	}

	//reconstruct nuclei and lineages from assignments (in this case one nuclei correspond to one supervoxel)
	//in this case it is a little bit easier since there are no divisions
	SibilingTypeNucleus iterN;
	for( int ii = 0; ii< lht.getMaxTM()-1; ii++)
	{
		int countS = 0;
		
		for(list<supervoxel>::iterator iterS = lht.supervoxelsList[ii].begin(); iterS != lht.supervoxelsList[ii].end(); ++iterS, countS++)
		{	
			
			//if(iterS->tempWildcard < 0.0f || assignmentId[ii][countS]->centroid[0] < -1e30f) //supervoxel has already been visited or it was not assigned
			if(iterS->tempWildcard < 0.0f ) //supervoxel has already been visited. Thus, we allow tracklets of single supervoxels
				continue;

			//we start a new lineage
			SibilingTypeSupervoxel auxS = iterS;
			lht.lineagesList.push_back(lineage());
			ParentTypeNucleus iterL = (++ ( lht.lineagesList.rbegin() ) ).base();//iterator to last added lineage
			int posTM = ii, posID;
			while((auxS)->centroid[0] > 0.0f)//compare to null assignment
			{
				//create nucleus
				iterN = lht.addNucleusFromSupervoxel(posTM,auxS);
				//update lineage
				iterL->bt.SetCurrent( iterL->bt.insert(iterN) );
				
				//update hypertree
				iterN->treeNode.setParent(iterL);
				iterN->treeNodePtr = iterL->bt.pointer_current();

				//find next element
				posID = (int)((auxS)->tempWildcard);
				(auxS)->tempWildcard = -1.0f;//signal that supervoxel has been visited
				if(posTM >= lht.getMaxTM() - 1)
					break;//we reached final time point->there is no next element
				auxS = assignmentId[posTM][posID];
				posTM++;
			}
		}
	}

	//add supervoxels in the last frame that have not been assigned to anybody
	int posTM = lht.getMaxTM()-1;
	for(list<supervoxel>::iterator iterS = lht.supervoxelsList[posTM].begin(); iterS != lht.supervoxelsList[posTM].end(); ++iterS)
	{
		if(iterS->tempWildcard < 0.0f ) //supervoxel has already been visited. Thus, we allow tracklets of single supervoxels
			continue;

		//we start a new lineage
		lht.lineagesList.push_back(lineage());
		ParentTypeNucleus iterL = (++ ( lht.lineagesList.rbegin() ) ).base();//iterator to last added lineage
		//create nucleus
		iterN = lht.addNucleusFromSupervoxel(posTM,iterS);
		//update lineage
		iterL->bt.SetCurrent( iterL->bt.insert(iterN) );

		//update hypertree
		iterN->treeNode.setParent(iterL);
		iterN->treeNodePtr = iterL->bt.pointer_current();
	}
	return 0;
}


//==============================================================================================================
int extendMatchingWithClearOneToOneAssignments(list< supervoxel > &superVoxelA, list< supervoxel > &superVoxelB, vector< vector< SibilingTypeSupervoxel > > &nearestNeighborVecAtoB, vector< vector< SibilingTypeSupervoxel > > &nearestNeighborVecBtoA, assignmentOneToMany* assignmentId)
{
	int N = (int) superVoxelA.size();
	int M = (int) superVoxelB.size();

	if( superVoxelB.size() != nearestNeighborVecBtoA.size() )
	{
		std::cout<<"ERROR: calculateSupervoxelMatchingOneToManyWithSparseHungarianAlgorithm: size of nearest neighbor vector BtoA is not the same as candidates list"<<endl;
		return 2;
	}

	if( superVoxelA.size() != nearestNeighborVecAtoB.size() )
	{
		std::cout<<"ERROR: calculateSupervoxelMatchingOneToManyWithSparseHungarianAlgorithm: size of nearest neighbor vector AtoB is not the same as input list"<<endl;
		return 2;
	}


	//generate a list of candidate elements that have been matched already (so we do not duplicate)
	bool* assignedId = new bool[M];
	memset(assignedId, 0, sizeof(bool) * M );//reset
	for(int ii = 0; ii<N; ii++)
	{
		for(int jj = 0; jj<assignmentId[ii].numAssignments; jj++)
		{
			assignedId[ assignmentId[ii].assignmentId[jj] ] = true;
		}
	}

	//set indexes for the list
	int ii = 0;
	for(list<supervoxel>::iterator iterS = superVoxelA.begin(); iterS != superVoxelA.end(); ++iterS, ++ii)
		iterS->tempWildcard = ((float) ii);
	ii = 0;
	vector< list<supervoxel>::iterator > vecIterB( M );//array of iterators to be able to access list
	for(list<supervoxel>::iterator iterS = superVoxelB.begin(); iterS != superVoxelB.end(); ++iterS, ++ii)
	{
		iterS->tempWildcard = ((float) ii);
		vecIterB[ii] = iterS;
	}
	
	ii = 0;
	int numDeaths = 0, numExtended = 0;	
	for(list<supervoxel>::iterator iterS = superVoxelA.begin(); iterS != superVoxelA.end(); ++iterS, ++ii)
	{
		if( assignmentId[ii].numAssignments > 0 )
			continue;//this elements has been assigned already

		numDeaths++;

		//find the best candidate from A->B
		float costAux, costBestAB = 0.0f;
		int idBestAB = -1;
		for(vector<SibilingTypeSupervoxel>::iterator iter = nearestNeighborVecAtoB[ii].begin(); iter != nearestNeighborVecAtoB[ii].end(); ++iter )
		{
			if( assignedId[ (int) ( (*iter)->tempWildcard ) ] == true ) // already assigned
				continue;
			
			costAux = iterS->intersectionCost( *(*iter) );
			if( costAux > costBestAB )
			{
				costBestAB = costAux;
				idBestAB = (int) ( (*iter)->tempWildcard );
			}
		}

		//check if the best candidate from A->B is also from B->A
		int idBestBA = ii;		
		if( idBestAB >= 0)
		{
			list< supervoxel >::iterator iterSB = vecIterB[idBestAB];
			float costBestBA = costBestAB + 0.1f;//to avoid floating point error in comparison		

			for(vector<SibilingTypeSupervoxel>::iterator iter = nearestNeighborVecBtoA[ idBestAB ].begin(); iter != nearestNeighborVecBtoA[idBestAB].end(); ++iter )
			{				
				costAux = iterSB->intersectionCost( *(*iter) );
				if( costAux > costBestBA )
				{
					idBestBA = (int) ( (*iter)->tempWildcard );
					break;
				}
			}


			//check if there was the one-to-one correspondence to extend element
			if (idBestBA == ii)
			{
				assignmentId[ii].assignmentId[0] = idBestAB;
				assignmentId[ii].assignmentCost[0] = costBestAB;
				assignmentId[ii].numAssignments = 1;
				assignedId[ idBestAB ] = true;
				numExtended++;
			}
		}
	}



	std::cout<<"INFO: extendMatchingWithClearOneToOneAssignments: number of extended deaths "<<numExtended<<" out of "<<numDeaths<<" elements that are not assigned yet."<<endl; 

	//release memory
	delete[] assignedId;

	return 0;
}
//==============================================================================================================
//very similar to extendMatchingWithClearOneToOneAssignments but in this case is a distance (->we try to MINIMIZE)
int extendMatchingWithClearOneToOneAssignmentsEuclidean(list< supervoxel > &superVoxelA, list< supervoxel > &superVoxelB, vector< vector< SibilingTypeSupervoxel > > &nearestNeighborVecAtoB, vector< vector< SibilingTypeSupervoxel > > &nearestNeighborVecBtoA, assignmentOneToMany* assignmentId)
{
	int N = (int) superVoxelA.size();
	int M = (int) superVoxelB.size();

	if( superVoxelB.size() != nearestNeighborVecBtoA.size() )
	{
		std::cout<<"ERROR: calculateSupervoxelMatchingOneToManyWithSparseHungarianAlgorithm: size of nearest neighbor vector BtoA is not the same as candidates list"<<endl;
		return 2;
	}

	if( superVoxelA.size() != nearestNeighborVecAtoB.size() )
	{
		std::cout<<"ERROR: calculateSupervoxelMatchingOneToManyWithSparseHungarianAlgorithm: size of nearest neighbor vector AtoB is not the same as input list"<<endl;
		return 2;
	}

	//set indexes for the list
	int ii = 0;
	for(list<supervoxel>::iterator iterS = superVoxelA.begin(); iterS != superVoxelA.end(); ++iterS, ++ii)
		iterS->tempWildcard = ((float) ii);
	ii = 0;
	vector< list<supervoxel>::iterator > vecIterB( M );//array of iterators to be able to access list
	for(list<supervoxel>::iterator iterS = superVoxelB.begin(); iterS != superVoxelB.end(); ++iterS, ++ii)
	{
		iterS->tempWildcard = ((float) ii);
		vecIterB[ii] = iterS;
	}

	//generate a list of candidate elements that have been matched already (so we do not duplicate)
	bool* assignedId = new bool[M];
	memset(assignedId, 0, sizeof(bool) * M );//reset
	float thrDist2 = 0.0f;
	int thrDistN = 0;
	ii = 0;
	for(list<supervoxel>::iterator iter = superVoxelA.begin(); iter!= superVoxelA.end(); ++iter, ++ii)
	{
		for(int jj = 0; jj<assignmentId[ii].numAssignments; jj++)
		{
			assignedId[ assignmentId[ii].assignmentId[jj] ] = true;

			thrDist2 += iter->Euclidean2Distance( *(vecIterB[ assignmentId[ii].assignmentId[jj] ]) );//we need to calculate average displacement to set an adaptive threshold for matching
			thrDistN++;
		}
	}

	thrDist2 *= 5.0f / ( (float) thrDistN );//threshold is K times teh average displacement (we are looking for things that do not intersect, so we can be generous)
	
	
	ii = 0;
	int numDeaths = 0, numExtended = 0;	
	for(list<supervoxel>::iterator iterS = superVoxelA.begin(); iterS != superVoxelA.end(); ++iterS, ++ii)
	{
		if( assignmentId[ii].numAssignments > 0 )
			continue;//this elements has been assigned already

		numDeaths++;

		//find the best candidate from A->B
		float costAux, costBestAB = thrDist2;//it has to be better than the threshold
		int idBestAB = -1;
		for(vector<SibilingTypeSupervoxel>::iterator iter = nearestNeighborVecAtoB[ii].begin(); iter != nearestNeighborVecAtoB[ii].end(); ++iter )
		{
			if( assignedId[ (int) ( (*iter)->tempWildcard ) ] == true ) // already assigned
				continue;
			
			costAux = iterS->Euclidean2Distance( *(*iter) );
			if( costAux < costBestAB )
			{
				costBestAB = costAux;
				idBestAB = (int) ( (*iter)->tempWildcard );
			}
		}

		//check if the best candidate from A->B is also from B->A
		int idBestBA = ii;		
		if( idBestAB >= 0)
		{
			list< supervoxel >::iterator iterSB = vecIterB[idBestAB];
			float costBestBA = costBestAB - 0.001f;//to avoid floating point error in comparison		

			for(vector<SibilingTypeSupervoxel>::iterator iter = nearestNeighborVecBtoA[ idBestAB ].begin(); iter != nearestNeighborVecBtoA[idBestAB].end(); ++iter )			
			{
				//if( assignmentId[ (int) ( (*iter)->tempWildcard ) ].numAssignments > 0 ) continue; //uncommment this line to only check the ones that have not been assigned yet
				costAux = iterSB->Euclidean2Distance( *(*iter) );
				if( costAux < costBestBA )
				{
					idBestBA = (int) ( (*iter)->tempWildcard );
					break;
				}
			}


			//check if there was the one-to-one correspondence to extend element
			if (idBestBA == ii)
			{
				assignmentId[ii].assignmentId[0] = idBestAB;
				assignmentId[ii].assignmentCost[0] = -1.0f;//useless here
				assignmentId[ii].numAssignments = 1;
				assignedId[ idBestAB ] = true;
				numExtended++;
			}
		}
	}



	std::cout<<"INFO: extendMatchingWithClearOneToOneAssignmentsEuclidean: number of extended deaths "<<numExtended<<" out of "<<numDeaths<<" elements that are not assigned yet. ThrDist2 = "<<thrDist2<<endl; 

	//release memory
	delete[] assignedId;

	return 0;
}



//==============================================================================================
int calculateSupervoxelMatchingOneToManyWithSparseHungarianAlgorithm(list< supervoxel > &superVoxelA, list< supervoxel > &superVoxelB, vector< vector< SibilingTypeSupervoxel > > &nearestNeighborVec, int costMethod, double thrCost, assignmentOneToMany* assignmentId)
{
	
	int N = (int) superVoxelA.size();
	int M = (int) superVoxelB.size();

	if( superVoxelB.size() != nearestNeighborVec.size() )
	{
		cout<<"ERROR: calculateSupervoxelMatchingOneToManyWithSparseHungarianAlgorithm: size of nearest neighbor vector is not the same as candidates list"<<endl;
		return 2;
	}

	//setup distance method
	float (supervoxel::*cost_func_ptr)(const supervoxel&) const;//define pointer to function-member class
	
	switch(costMethod)
	{
	case 0://absolute number of intersecting points
		cost_func_ptr = &supervoxel::intersectionCost;
		break;
	case 1://relative number of intersecting points with respect to candidate size. Maximum value is 1.0
		cost_func_ptr = &supervoxel::intersectionCostRelativeToCandidate;
		break;
	default:
		cout<<"ERROR: at calculateTrackletsWithSparseHungarianAlgorithm: distanceMEthod not supported by the code"<<endl;
		return 1;
	}

	//to keep the best assignment for each input (we need it later to try to extend dead)
	int* bestAssignmentId = new int[N];
	float* bestAssignmentCost = new float[N];

	//reset solution
	for(int ii = 0;ii<N;ii++)
	{
		assignmentId[ii].numAssignments = 0;//garbage potential (no assignment) for all of them
		bestAssignmentId[ii] = -1;
		bestAssignmentCost[ii] = thrCost;
	}
	//reset wildcard in supervoxelA to identify id
	int count = 0;
	for(list< supervoxel >::iterator iter = superVoxelA.begin(); iter != superVoxelA.end(); ++iter, ++count)
	{
		iter->tempWildcard = count;
	}

	//since we allow one-to-many, we do not need Hungarian. It is just a greedy selection from the point of view of the candidates
	count = 0;
	int k, kAux;
	float costAux, costBest;
	for(list< supervoxel >::iterator iter = superVoxelB.begin(); iter != superVoxelB.end(); ++iter, ++count)
	{
		costBest = thrCost;
		k = -1;
		for(vector<SibilingTypeSupervoxel>::iterator iterNeigh = nearestNeighborVec[count].begin(); iterNeigh != nearestNeighborVec[count].end(); ++iterNeigh)
		{
			costAux = ((*iter).*cost_func_ptr)(*(*iterNeigh));//call the selected distance
			kAux = ( (int) (*iterNeigh)->tempWildcard );
			if( (costAux > costBest) )//better than assigning to garbage potential
			{
				k = kAux;
				costBest = costAux;
			}
			
			//if we were assigning it greedy (without uniqueness constraints)
			if( bestAssignmentCost[kAux] < costAux )
			{
				bestAssignmentCost[kAux] = costAux;
				bestAssignmentId[kAux] = count;
			}
		}

		if( k >= 0)
		{
			if( assignmentId[k].numAssignments < MAX_NUMBER_ASSINGMENTS_SHA )
			{
				assignmentId[k].assignmentCost[ assignmentId[k].numAssignments ] = costBest;
				assignmentId[k].assignmentId[ assignmentId[k].numAssignments++ ] = count;
			}else
			{
				cout<<"WARNING: at calculateTrackletsWithSparseHungarianAlgorithm: supervoxel "<<k<<" has exceeded the maximum number of possible assignments"<<endl;
				//return 3;
			}
		}
	}


#ifdef USE_DEATH_EXTENSION_ALGORITHM
	//we want to try to impose that every input nucleus has an assignment so we run a greedy matching to see if some of the elements that have not been assigned can "steal" an element from elements with multiple assignments
	int numExtended = 0, numNotAssigned = 0;
	vector<int> deadId;
	deadId.reserve(500);
	for(int ii = 0;ii<N;ii++)
	{
		if( assignmentId[ii].numAssignments == 0 )
		{
			numNotAssigned++;
			if(bestAssignmentId[ii] >= 0)
			{
				deadId.push_back(ii);		
			}
		}
	}
	//find who has "stolen" the best assignment
	int auxId;
	vector<int> stolenId(deadId.size(), -1);
	for(int ii = 0;ii<N;ii++)
	{
		for( int jj = 0;jj < assignmentId[ii].numAssignments; jj++)
		{			
			//linear search (we do not expect deadId to be ver large
			auxId = assignmentId[ii].assignmentId[jj];
			for(size_t kk = 0; kk < deadId.size(); kk++)
			{
				if( auxId == bestAssignmentId [ deadId[kk] ])
				{	
					stolenId[kk] = ii;
					break;
				}
			}			
		}
	}
	//check if we can extend death
	int auxId2, old;
	for(size_t ii = 0; ii < deadId.size(); ii++)
	{
		if( stolenId[ii] >= 0 && assignmentId[ stolenId[ii] ].numAssignments > 1 )
		{
			numExtended++;
			//place new assignment
			auxId = deadId[ii];
			assignmentId[ auxId ].assignmentId[0] = bestAssignmentId [ auxId ];
			assignmentId[ auxId ].assignmentCost[0] = bestAssignmentCost [ auxId ];
			assignmentId[ auxId ].numAssignments = 1;

			//remove from previous "owner"
			auxId2 = stolenId[ii];
			old = assignmentId[ auxId2 ].numAssignments;
			for( int jj = 0;jj < assignmentId[ auxId2 ].numAssignments; jj++)
			{
				if( bestAssignmentId [ auxId ] == assignmentId[ auxId2 ].assignmentId[jj] )
				{
					//delete element
					for( int kk = jj; kk < assignmentId[ auxId2 ].numAssignments -1; kk++ )
					{
						assignmentId[ auxId2 ].assignmentId[ kk ] = assignmentId[ auxId2 ].assignmentId[ kk+1 ];
						assignmentId[ auxId2 ].assignmentCost[ kk ] = assignmentId[ auxId2 ].assignmentCost[ kk+1 ];
					}
					assignmentId[ auxId2 ].numAssignments--;
					break;
				}				
			}
			if( old - assignmentId[ auxId2 ].numAssignments != 1 )
			{
				cout<<"ERROR: at calculateTrackletsWithSparseHungarianAlgorithm: I should never reach this line!!!"<<endl;
				return 4;
			}
		}
	}
	
	cout<<"INFO: calculateSupervoxelMatchingOneToManyWithSparseHungarianAlgorithm: number of extended deaths "<<numExtended<<" out of "<<deadId.size()<<" elements that are not assigned but have an edge."<<endl; 
	cout<<"INFO: calculateSupervoxelMatchingOneToManyWithSparseHungarianAlgorithm: total of not assignments is "<<numNotAssigned<<" out of "<<N<<" input elements to be matched"<<endl;
#else
	cout<<"INFO: calculateSupervoxelMatchingOneToManyWithSparseHungarianAlgorithm: greedy algorithm to try to extend deaths not in use"<<endl;
#endif

	//release memory
	delete[] bestAssignmentId;
	delete[] bestAssignmentCost;


	return 0;
}
