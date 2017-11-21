/*
 *
 * \brief Contains different routines that are called form the main.cpp part of theprogram to not make the code so cluttr
 *        
 *
 */

#include <iostream>
#include <queue>
#include "supportFunctionsMain.h"
#include "temporalLogicalRules/trackletCalculation.h"

using namespace std;



//=========================================================================
template <class imgTypeC>
int parseNucleiList2TGMM(std::vector<GaussianMixtureModel*> &vecGM,lineageHyperTree &lht,int frame, bool regularizeW, float thrDist2LargeDisplacement)
{
	if(frame >= lht.getMaxTM())
		return 0;

	list<nucleus>* listNucleiPtr = (&(lht.nucleiList[frame]));
	size_t N = listNucleiPtr->size();

	//resize vecGM
	if( N > vecGM.size())//I need to allocate
	{
		vecGM.reserve( N );
		for(size_t ii = vecGM.size(); ii<N; ii++)
			vecGM.push_back( new GaussianMixtureModel(ii) );
	}else{//I need to deallocate
		for(size_t ii = N; ii < vecGM.size(); ii++)
			delete vecGM[ii];
		vecGM.resize(N);
	}


	//reset supervoxels id
	int count = 0;
	for(list<supervoxel>::iterator iterS = lht.supervoxelsList[frame].begin(); iterS != lht.supervoxelsList[frame].end(); ++iterS, count++)
		iterS->tempWildcard = (float) count;


	//compute all the stats from centroids	
	float N_k;
	float m_k[dimsImage];
	float S_k[dimsImage * (1 + dimsImage) / 2 ];
	count =0;
	int countW;
	for(list<nucleus>::iterator iterN = listNucleiPtr->begin(); iterN != listNucleiPtr->end(); ++iterN, ++count)
	{
		lht.calculateGMMparametersFromNuclei<imgTypeC>(iterN,m_k,&N_k,S_k);

		//copy results
		vecGM[count]->N_k = N_k;

		vecGM[count]->alpha_k = N_k;
		vecGM[count]->beta_k = N_k;
		vecGM[count]->nu_k = N_k;
		vecGM[count]->alpha_o = 0;//there are no priors here
		vecGM[count]->beta_o = 0;
		vecGM[count]->nu_o = dimsImage + 1;//minimum degrees of freedom are dimsImage +1
		vecGM[count]->sigmaDist_o = 0;//to avoid complaint from Valgrind of uninitialized variable
		countW = 0;
		for(int ii =0;ii<dimsImage;ii++)
		{
			vecGM[count]->m_k(ii) = m_k[ii];
			vecGM[count]->m_o(ii) = m_k[ii];
			for(int jj = ii; jj <dimsImage; jj++)
			{
				vecGM[count]->W_k(ii,jj) = S_k[countW];
				vecGM[count]->W_k(jj,ii) = S_k[countW];
				countW++;
			}
		}

		if( vecGM[count]->W_k(dimsImage-1,dimsImage-1) < 1e-8 )//it means we have a 2D ellipsoid->just regularize W with any value (we will ignore it in the putput anyway)
		{
			vecGM[count]->W_k(dimsImage-1,dimsImage-1) = 0.5 * (vecGM[count]->W_k(0,0) + vecGM[count]->W_k(1,1));
		}
			
		vecGM[count]->W_k = ( vecGM[count]->W_k.inverse() ) / vecGM[count]->nu_k;
		

		

		if(regularizeW == true)
			vecGM[count]->regularizePrecisionMatrix(true);

		vecGM[count]->W_o = vecGM[count]->W_k * vecGM[count]->nu_k / vecGM[count]->nu_o;
		//VIP: we do not recover parent information. We just assume that parentId is the index of the nucleiList[frame]
		vecGM[count]->parentId = count;

		vecGM[count]->lineageId = count;//some pieces of the code need this info to be different than -1;in this case everything is from a different lineage

		//save supervoxel
		vecGM[count]->supervoxelIdx.resize( iterN->treeNode.getNumChildren() );
		int countS = 0;
		for( vector<ChildrenTypeNucleus>::iterator iter = iterN->treeNode.getChildren().begin(); iter != iterN->treeNode.getChildren().end(); ++iter, countS++)
		{
			vecGM[count]->supervoxelIdx[countS] = (int)((*iter)->tempWildcard);
		}

		//save background probability for this nucleus
		vecGM[count]->beta_o = iterN->probBackground;//TODO: use its own variable, although here it was set to zero anyways

		//save split score based on confidence: link to CATMAID editing
		//vecGM[count]->splitScore = iterN->confidence;		
		vecGM[count]->splitScore = lht.confidenceScoreForNucleus( iterN->treeNodePtr, thrDist2LargeDisplacement ); //TODO: calculate this before hand and save it in confidence

		//vecGM[count]->splitScore = iterN->debugVisualization;//uncomment this to be able to visualize specific cases
	}	

	//cout<<"=============DEBUGGING: confidence split score disconnected to study incoherences========================="<<endl;
	//cout<<"=============REMINDER: confidence split score NOT disconnected to study incoherences========================="<<endl;

	return 0;
}


//=============================================================================================================
int parseHierarchicalSegmentation2LineageHyperTree(hierarchicalSegmentation* hs,lineageHyperTree& lht)
{

	queue< TreeNode<nodeHierarchicalSegmentation>* > q;
	q.push( hs->dendrogram.pointer_mainRoot() );
	TreeNode<nodeHierarchicalSegmentation>* aux = NULL;
	//traverse tree top to bottom to generate lineages
	while( q.empty() == false )
	{
		aux = q.front();
		q.pop();

		if( aux->data.thrTau > hs->getMaxTau() )
		{

			//continue searching down the dendrogram for roots of lineages
			if(aux->left != NULL )
				q.push( aux->left );
			if(aux->right != NULL )
				q.push( aux->right );
			//NOTE: we do not write out objects that are clearly segmented (ie, isolated nodes that between minTau and maxTau have no connections in the dendrogram)
		}else{//generate a new lineage from this root

			lht.lineagesList.push_back( lineage() );
			list<lineage>::iterator listLineageIter = ((++ ( lht.lineagesList.rbegin() ) ).base());//iterator for the last element in the list

			queue< TreeNode<nodeHierarchicalSegmentation>* > qHS;
			queue< TreeNode<ChildrenTypeLineage>* > qLHT;//stores parent in LHT binary tree
			qHS.push( aux );
			qLHT.push( NULL );
			
			TreeNode<ChildrenTypeLineage>* auxLHT = NULL;
			int TM = 0;
			supervoxel sv;
			while( qHS.empty() == false )
			{
				//retrieve information
				aux = qHS.front();
				qHS.pop();
				auxLHT = qLHT.front();
				qLHT.pop();
				
				if( auxLHT == NULL )
					TM = 0;//root of dendrogram
				else
					TM = auxLHT->data->TM + 1;

				//generate supervoxel
				hs->supervoxelAtTreeNode(aux, sv);
				sv.TM = TM;

				//add supervoxel to lht
				lht.supervoxelsList[ TM ].push_back( sv );
				ChildrenTypeNucleus iterS = ((++ ( lht.supervoxelsList[ TM ].rbegin() ) ).base());//iterator for the last element in the list
				ParentTypeSupervoxel iterN =  lht.addNucleusFromSupervoxel(TM, iterS);//creates a nucleus from a single supervoxel and updates the hypergraph properly. Returns an iterator to the created element

				//add nucleus to lineage				
				iterN->treeNode.setParent(listLineageIter);
				if( auxLHT != NULL )
					listLineageIter->bt.SetCurrent( auxLHT );
				else //root
					listLineageIter->bt.SetMainRootToNULL();

				iterN->treeNodePtr = listLineageIter->bt.insert(iterN);
				if(iterN->treeNodePtr == NULL)
				{
					cout<<"ERROR: at parseHierarchicalSegmentation2LineageHyperTree: inerting node into binary tree"<<endl;
					return 3;
				}

				//update queues to keep descending in the lineage
				if( aux->left != NULL )
				{
					qHS.push( aux->left );
					qLHT.push( iterN->treeNodePtr );
				}
				
				if( aux->right != NULL )
				{
					qHS.push( aux->right );
					qLHT.push( iterN->treeNodePtr );
				}
			}
		}
	}

	return 0;
}

//=========================================================================================
//=============================================================================================================
int debugMergeSplitHStoLHT(hierarchicalSegmentation* hs,lineageHyperTree& lht, imgVoxelType tau)
{
	
	//generate segmentation
	hs->segmentationAtTau(tau);

	//keep a list of root nodes so I do not duplicates entries
	vector< TreeNode< nodeHierarchicalSegmentation >* > rootNodes;
	rootNodes.reserve( hs->currentSegmentatioSupervoxel.size() );

	//trim all supervoxels
	for(size_t ii = 0; ii < hs->currentSegmentatioSupervoxel.size(); ii++ )
	{
		hs->currentSegmentatioSupervoxel[ii].trimSupervoxel<unsigned short int>();
	}

	for(size_t ii = 0; ii < hs->currentSegmentatioSupervoxel.size(); ii++ )
	{
		//go all the way up by merging
		TreeNode< nodeHierarchicalSegmentation >* auxNode = hs->currentSegmentationNodes[ii], *rootMerge = hs->currentSegmentationNodes[ii];
		supervoxel auxSv;
		supervoxel rootMergeSv = hs->currentSegmentatioSupervoxel[ii];
		float score = 1.0f;

		while( score > 0.0f )
		{			
			auxSv = rootMergeSv;
			auxNode = rootMerge;
			score = hs->suggestMerge<unsigned short int>( auxNode, auxSv, &rootMerge, rootMergeSv );			
		}

		//check if node was already done
		bool isDone = false;
		for(size_t jj = 0; jj < rootNodes.size(); jj++ )
		{
			if( rootNodes[jj] == auxNode )
			{
				isDone = true;
				break;
			}
		}
		if( isDone == true )
			continue;

		rootNodes.push_back( auxNode );

		//traverse all the ways down by splitting starting at auxSv
		//generate a new lineage from this root
		lht.lineagesList.push_back( lineage() );
		list<lineage>::iterator listLineageIter = ((++ ( lht.lineagesList.rbegin() ) ).base());//iterator for the last element in the list

		queue< TreeNode<nodeHierarchicalSegmentation>* > qHS;
		queue< TreeNode<ChildrenTypeLineage>* > qLHT;//stores parent in LHT binary tree
		qHS.push( auxNode );
		qLHT.push( NULL );
		TreeNode<nodeHierarchicalSegmentation>* rootSplit[2];
		supervoxel rootSplitSv[2];

		TreeNode<ChildrenTypeLineage>* auxLHT = NULL;
		int TM = 0;
		void* dataPtr = hs->basicRegionsVec[0].dataPtr;
		supervoxel sv;
		while( qHS.empty() == false )
		{
			//retrieve information
			auxNode = qHS.front();
			qHS.pop();
			auxLHT = qLHT.front();
			qLHT.pop();

			if( auxLHT == NULL )
				TM = 0;//root of dendrogram
			else
				TM = auxLHT->data->TM + 1;

			//generate supervoxel and trim it
			hs->supervoxelAtTreeNode(auxNode, sv);
			sv.TM = TM;
			sv.dataPtr = dataPtr;
			sv.trimSupervoxel<unsigned short int>();			


			//add supervoxel to lht
			lht.supervoxelsList[ TM ].push_back( sv );
			ChildrenTypeNucleus iterS = ((++ ( lht.supervoxelsList[ TM ].rbegin() ) ).base());//iterator for the last element in the list
			ParentTypeSupervoxel iterN =  lht.addNucleusFromSupervoxel(TM, iterS);//creates a nucleus from a single supervoxel and updates the hypergraph properly. Returns an iterator to the created element
			

			//add nucleus to lineage				
			iterN->treeNode.setParent(listLineageIter);
			if( auxLHT != NULL )
				listLineageIter->bt.SetCurrent( auxLHT );
			else //root
				listLineageIter->bt.SetMainRootToNULL();

			iterN->treeNodePtr = listLineageIter->bt.insert(iterN);
			if(iterN->treeNodePtr == NULL)
			{
				cout<<"ERROR: at parseHierarchicalSegmentation2LineageHyperTree: inserting node into binary tree"<<endl;
				return 3;
			}

			/*
			//debugging
			if( TM == 0 && lht.supervoxelsList[ TM ].size() == 2363 )
			{
				iterS->weightedCentroid<unsigned short int>();
				cout<<"ii = "<<ii<<";supervoxel = "<<(*iterS)<<";par tauThr = "<<auxNode->parent->data.thrTau<<endl;
				exit(3);
			}
			*/

			//calculate split and update queues to keep descending in the lineage
			hs->suggestSplit<unsigned short int>(auxNode, sv, rootSplit, rootSplitSv);
			if( rootSplit[0] != NULL )
			{
				qHS.push( rootSplit[0] );
				qLHT.push( iterN->treeNodePtr );
			}
			if( rootSplit[1] != NULL )
			{
				qHS.push( rootSplit[1] );
				qLHT.push( iterN->treeNodePtr );
			}
			
		}		
	}

	
	return 0;
}

//=====================================================================
TreeNode< ChildrenTypeLineage >* addSupervoxelsPointersFromLineage(std::vector< vector<supervoxel*> > &svIniVec, int iniTM, int endTM, TreeNode< ChildrenTypeLineage >* node)
{
	//add elements for node
	TreeNode< ChildrenTypeLineage >* auxNode = node;
	while( auxNode->parent != NULL && auxNode->data->TM > iniTM )
		auxNode = auxNode->parent;
	//traverse the node downstream
	queue< TreeNode< ChildrenTypeLineage >* > q;
	q.push( auxNode );
	TreeNode< ChildrenTypeLineage >* root = auxNode;
	while( q.empty() == false )
	{
		auxNode = q.front();
		q.pop();
		if( auxNode->left != NULL && auxNode->left->data->TM <= endTM )
			q.push( auxNode->left );
		if( auxNode->right != NULL && auxNode->right->data->TM <= endTM )
			q.push( auxNode->right );

		//add all supervoxels
		int offsetTM = auxNode->data->TM - iniTM;
		for( vector< ChildrenTypeNucleus >::iterator iterS = auxNode->data->treeNode.getChildren().begin(); iterS != auxNode->data->treeNode.getChildren().end(); ++iterS )
		{
			svIniVec[ offsetTM ].push_back( &(*(*iterS)) );
		}
	}

	return root;
}


//=======================================================================
void parseImagePath(string& imgRawPath, int frame)
{	

	size_t found=imgRawPath.find_first_of("?");
	while(found != string::npos)
	{
		int intPrecision = 0;
		while ((imgRawPath[found] == '?') && found != string::npos)
		{
			intPrecision++;
			found++;
			if( found >= imgRawPath.size() )
				break;

		}

		
		char bufferTM[16];
		switch( intPrecision )
		{
		case 2:
			sprintf(bufferTM,"%.2d",frame);
			break;
		case 3:
			sprintf(bufferTM,"%.3d",frame);
			break;
		case 4:
			sprintf(bufferTM,"%.4d",frame);
			break;
		case 5:
			sprintf(bufferTM,"%.5d",frame);
			break;
		case 6:
			sprintf(bufferTM,"%.6d",frame);
			break;
		}
		string itoaTM(bufferTM);
		
		found=imgRawPath.find_first_of("?");
		imgRawPath.replace(found, intPrecision,itoaTM);
		

		//find next ???
		found=imgRawPath.find_first_of("?");
	}
	
}

//===========================================================
void transposeStackUINT16(mylib::Array *img)
{
	//cout<<"WARNING: transposing each slice to agree with Matlab convention"<<endl;
	if(img->type != mylib::UINT16_TYPE)
	{
		cout<<"ERROR: at transposeStackUINT16: array has to be uint16"<<endl;
		exit(3);
	}
	if(img->ndims != 3)
	{
		cout<<"ERROR: at transposeStackUINT16: code only ready for 3D arrays"<<endl;
		exit(3);
	}

	mylib::Size_Type sliceSize = img->dims[0]*img->dims[1];
	mylib::uint16* imgPtr = (mylib::uint16*) (img->data);
	mylib::uint16* imgCpyPtr = new mylib::uint16[sliceSize]; 

	mylib::Size_Type imgIdx = 0;
	for(mylib::Dimn_Type zz = 0; zz<img->dims[2]; zz++)
	{
		//copy plane
		memcpy(imgCpyPtr,&(imgPtr[imgIdx]),sizeof(mylib::uint16)*sliceSize);

		for(mylib::Dimn_Type xx = 0; xx<img->dims[0]; xx++)		
		{
			mylib::Size_Type offset2 = xx;
			for(mylib::Dimn_Type yy = 0; yy<img->dims[1]; yy++)
			{
				imgPtr[imgIdx++] = imgCpyPtr[offset2];
				offset2 += img->dims[0];
			}
		}
	}

	//change array dimensions
	mylib::Dimn_Type aux = img->dims[0];
	img->dims[0] = img->dims[1];
	img->dims[1] = aux;

	delete[] imgCpyPtr;
}

//==============================================================
int extendDeadNucleiAtTMwithHS(lineageHyperTree &lht, hierarchicalSegmentation* hsForward, int TM, int& numExtensions, int &numDeaths)
{
	
	numExtensions = 0;
	numDeaths = 0;

	if( TM < 0 || TM >= (int) ( lht.getMaxTM()) )
		return 0;

	TreeNode<ChildrenTypeLineage>* aux;	
	for(list<nucleus>::iterator iterN = lht.nucleiList[TM].begin(); iterN != lht.nucleiList[TM].end(); ++iterN)
	{
		if( iterN->treeNodePtr->getNumChildren() == 0)
		{
			numDeaths++;
			numExtensions += extendDeadNucleiWithHS(lht, hsForward, iterN->treeNodePtr);
		}
	}

	return 0;
}

//=============================================================================================================================
int extendDeadNucleiWithHS(lineageHyperTree &lht, hierarchicalSegmentation* hsForward, TreeNode<ChildrenTypeLineage>* rootDead)
{
	if(rootDead == NULL || rootDead->getNumChildren() != 0)
		return 0;//rootDead is not a dead split so we cannnot do anything

	//try to find the most obvious continuation

	//1.-Generate a super-supervoxel by merging all the supervoxel belonging to the nucleus
	vector<ChildrenTypeNucleus>::iterator iterS = rootDead->data->treeNode.getChildren().begin();
	supervoxel supervoxelFromNucleus( *(*iterS) );
	++iterS;
	vector< supervoxel* > auxVecS;
	for(;  iterS != rootDead->data->treeNode.getChildren().end(); ++iterS)
	{
		auxVecS.push_back( &(*(*iterS)) );
	}
	supervoxelFromNucleus.mergeSupervoxels(auxVecS);//we add all the components at once


	//2.-find candidate with largest intersection
	uint64 intersectionSize = 0, auxI;
	SibilingTypeSupervoxel intersectionS;
	for(iterS = rootDead->data->treeNode.getChildren().begin();  iterS != rootDead->data->treeNode.getChildren().end(); ++iterS)
	{
		for(vector< SibilingTypeSupervoxel >::iterator iterS2 = (*iterS)->nearestNeighborsInTimeForward.begin(); iterS2 != (*iterS)->nearestNeighborsInTimeForward.end(); ++iterS2)
		{
			auxI = supervoxelFromNucleus.intersectionSize( *(*(iterS2)) );
			if(  auxI > intersectionSize )
			{
				intersectionSize = auxI;
				intersectionS = (*iterS2);
			}
		}
	}
	
	if( intersectionSize == 0 )
		return 0;//no clear option for extending death

	//3.find the nuclei that "owns" the supervoxel
	list< nucleus >::iterator iterNucOwner, iterNucOwnerDaughterL, iterNucOwnerDaughterR, iterNucNew;
	int intIsCellDivision = 0x000;//0->no children;0x0001->left children;0x0010->right children;0x0011->both children
	if( intersectionS->treeNode.hasParent() == false )//the supervoxel with highest intersection has not been claimed by anybody->just take it
	{

		iterNucNew = lht.addNucleusFromSupervoxel(rootDead->data->TM + 1, intersectionS );//returns iterator to newly created nucleus

		//update lineage-nucleus hypergraph
		iterNucNew->treeNode.setParent( rootDead->data->treeNode.getParent() );
		rootDead->data->treeNode.getParent()->bt.SetCurrent( rootDead );
		iterNucNew->treeNodePtr = rootDead->data->treeNode.getParent()->bt.insert( iterNucNew );
		if( iterNucNew->treeNodePtr == NULL )
			exit(3);

		return 1;//we have added one nucleus
	}else{
		 iterNucOwner = intersectionS->treeNode.getParent();
		 if( iterNucOwner->treeNodePtr->parent != NULL )//we were one time step ahead
		 {
			 iterNucOwner = iterNucOwner->treeNodePtr->parent->data;
			 
			 if( iterNucOwner->treeNodePtr->getNumChildren() > 1)
			 {
				 intIsCellDivision = 0x0011;
				 iterNucOwnerDaughterR = iterNucOwner->treeNodePtr->right->data;
				 iterNucOwnerDaughterL = iterNucOwner->treeNodePtr->left->data;
			 }else{
				 if ( iterNucOwner->treeNodePtr->left !=NULL )
				 {
					 intIsCellDivision = 0x0001;
					 iterNucOwnerDaughterL = iterNucOwner->treeNodePtr->left->data;
					 iterNucOwnerDaughterR = iterNucOwnerDaughterL;
				 }else{
					 intIsCellDivision = 0x0010;
					 iterNucOwnerDaughterR = iterNucOwner->treeNodePtr->right->data;
					 iterNucOwnerDaughterL = iterNucOwnerDaughterR;
				 }
			 }			 

		 }else{
			 return 0;//there is no parent
		 }
	}

	if( iterNucOwner->TM != rootDead->data->TM)
	{
		cout<<"ERROR: lineageHyperTree::extendDeadNuclei: TM does not agree between two candidate nucleus"<<endl;
		exit(5);
	}

	//4.-run a small Hungarian algorithm in order to decide what is the best matching to solve this issue.
	//We setup a list of cancidates in time point t
	list< supervoxel > svListT0;	
	for(iterS = rootDead->data->treeNode.getChildren().begin();  iterS != rootDead->data->treeNode.getChildren().end(); ++iterS)
	{
		svListT0.push_back( *(*iterS) );
	}
	for(iterS = iterNucOwner->treeNode.getChildren().begin();  iterS != iterNucOwner->treeNode.getChildren().end(); ++iterS)
	{
		svListT0.push_back( *(*iterS) );
	}


	//4.1-Addition with respect to original code in lineageHypertree class
	//Check in supervoxel cancidates in t+1 if splitting them should help. If it does ->do it before calculating Hungarian algorithm

	float ddNoSplit;
	TreeNode< nodeHierarchicalSegmentation >* rootSplit[2];
	supervoxel rootSplitSv[2];
	float ddSplit[2];
	//items needed to know correspondence between hsForward->currentSegmentation and lht.supervoxelsList[rootDead->data->TM + 1]
	int TMforward = rootDead->data->TM + 1;
	vector<ChildrenTypeNucleus> svListIterVec;
	lht.getSupervoxelListIteratorsAtTM( svListIterVec, TMforward );
	size_t countNN = 0;//to keep id
	for(vector<ChildrenTypeNucleus>::iterator iterNN = svListIterVec.begin(); iterNN != svListIterVec.end(); ++iterNN, countNN++)
		(*iterNN)->tempWildcard = countNN;

	for(list< supervoxel >::iterator iter = svListT0.begin(); iter != svListT0.end(); ++iter)
	{
		size_t sizeVec = iter->nearestNeighborsInTimeForward.size();
		for( size_t cc = 0; cc < sizeVec; cc++ )//we need to access using indexes because we push_back into iter->nearestNeighborsInTimeForward if a split is done
		{
			SibilingTypeSupervoxel iterNeigh = iter->nearestNeighborsInTimeForward[cc];
			ddNoSplit = supervoxelFromNucleus.JaccardDistance(*iterNeigh);//calculate distance without splitting
			float ddThr = ddNoSplit - 0.1;//it has to be significantly better

			//propose split
			float prob = hsForward->suggestSplit<float>((*iterNeigh).nodeHSptr, (*iterNeigh), rootSplit, rootSplitSv);//probability of the split
			if( prob < 1e-3 )
				continue;
			//calculate Jaccard distance
			ddSplit[0] = supervoxelFromNucleus.JaccardDistance(rootSplitSv[0]);
			ddSplit[1] = supervoxelFromNucleus.JaccardDistance(rootSplitSv[1]);
			
			if( ddSplit[0] < ddThr || ddSplit[1] < ddThr )//incorporate split into solution space
			{
				
				
				//if matching is significantly better->proceed with to split supervoxels and incorporate them in the lineage
				//calculate Gauss stats
				rootSplitSv[0].weightedGaussianStatistics<float>(true);
				rootSplitSv[1].weightedGaussianStatistics<float>(true);

				//update sv-nucleus info				
				rootSplitSv[0].treeNode.setParent((*iterNeigh).treeNode.getParent());
				rootSplitSv[1].treeNode.setParent((*iterNeigh).treeNode.getParent());


				//lht->supervoxelsList[TM] was created as a copy of hs->currentSegmentation for backwards compatibility (even if it is redundant). So we just need tokeep that structure								
				int aa = (int) ((*iterNeigh).tempWildcard);
				hsForward->currentSegmentatioSupervoxel[aa] = rootSplitSv[0];//root split has the correct nodeHSptr
				hsForward->currentSegmentatioSupervoxel.push_back( rootSplitSv[1] );//this could dynamically reallocate memory and make all the pointers nodeHSptr->data.svPtr invalid (the crash would be obvious). So I reserve when creating a partition.
				
				hsForward->currentSegmentationNodes[aa] = rootSplit[0];
				hsForward->currentSegmentationNodes.push_back( rootSplit[1] );
			

				//update nucleus-sv info (remember: lht->supervoxelsList[TM] was created as a copy of hs->currentSegmentation for backwards compatibility);
				//I actually only need to change nucleus-supervoxel by adding children to nucleus since it was a split;
				(*(svListIterVec[aa])) = rootSplitSv[0];
				(svListIterVec[aa])->tempWildcard = aa;
				lht.supervoxelsList[TMforward].push_back(rootSplitSv[1]);
				SibilingTypeSupervoxel iterSadded = (++ ( lht.supervoxelsList[TMforward].rbegin() ) ).base();//get iterator to added element
				iterSadded->tempWildcard = lht.supervoxelsList[TMforward].size() - 1;
				(*iterNeigh).treeNode.getParent()->treeNode.addChild( iterSadded );


				//add new supervoxels in iter->NEARESTneighbors list for Hungarian algorithm;
				iter->nearestNeighborsInTimeForward.push_back( iterSadded );
			}
		}
	}

	list< supervoxel> nullAssignmentList;//temporary supervoxel list to simulate null assignment
	nullAssignmentList.push_back(supervoxel());
	SibilingTypeSupervoxel nullAssignment = nullAssignmentList.begin();
	nullAssignment->centroid[0] = -1e32f;//characteristic to find out no assignment
	vector<SibilingTypeSupervoxel> assignmentId;
	int err = calculateTrackletsWithSparseHungarianAlgorithm(svListT0, 0, 0.9, assignmentId, &nullAssignment);
	if( err > 0)
		exit(err);

	
	//5.-Parse results and modify assignment accordingly
	int extendedLineages = 0;//1->we have extended it
	list< supervoxel >::iterator svListT0iter = svListT0.begin();
	int count =0;
	for(iterS = rootDead->data->treeNode.getChildren().begin();  iterS != rootDead->data->treeNode.getChildren().end(); ++iterS, ++svListT0iter, ++count)
	{
		if( assignmentId[count]->centroid[0] < 0.0f )
			continue;//not assigned to anything

		if( assignmentId[count]->treeNode.hasParent() == false )//the assigned element has no parent-> we can claim it directly
		{
			if( extendedLineages == 0)//we need to create new nucleus in the list
			{
				lht.nucleiList[ rootDead->data->TM + 1 ].push_back( nucleus(rootDead->data->TM + 1, assignmentId[count]->centroid) );
				iterNucNew = (++ ( lht.nucleiList[ rootDead->data->TM + 1 ].rbegin() ) ).base();//iterator to last added nucleus
				
				//update lineage-nucleus hypergraph
				iterNucNew->treeNode.setParent( rootDead->data->treeNode.getParent() );
				rootDead->data->treeNode.getParent()->bt.SetCurrent( rootDead );
				iterNucNew->treeNodePtr = rootDead->data->treeNode.getParent()->bt.insert( iterNucNew );
				if( iterNucNew->treeNodePtr == NULL )
					exit(3);
				//update supervoxel-nucleus hypergraph
				iterNucNew->addSupervoxelToNucleus( assignmentId[count] );
				assignmentId[count]->treeNode.setParent( iterNucNew );

				extendedLineages++;
				//iterNucNew->confidence = 4;//to analyze extended elements
			}
		}
		else if( ( assignmentId[count]->treeNode.getParent() == iterNucOwnerDaughterL ) || ( assignmentId[count]->treeNode.getParent() == iterNucOwnerDaughterR ) )//to confirm it is not null assignment && we are "stealing" a supervoxel from iterNucOwner and not from anothe nuclei
		{
			if( extendedLineages == 0)//we need to create new nucleus in the list
			{
				lht.nucleiList[ rootDead->data->TM + 1 ].push_back( nucleus(rootDead->data->TM + 1, assignmentId[count]->centroid) );
				iterNucNew = (++ ( lht.nucleiList[ rootDead->data->TM + 1 ].rbegin() ) ).base();//iterator to last added nucleus
				
				//update lineage-nucleus hypergraph
				iterNucNew->treeNode.setParent( rootDead->data->treeNode.getParent() );
				rootDead->data->treeNode.getParent()->bt.SetCurrent( rootDead );
				iterNucNew->treeNodePtr = rootDead->data->treeNode.getParent()->bt.insert( iterNucNew );
				if( iterNucNew->treeNodePtr == NULL )
					exit(3);

				extendedLineages++;
				//iterNucNew->confidence = 4;//to analyze extended elements
			}
			//update supervoxel-nucleus hypergraph
			if ( assignmentId[count]->treeNode.getParent() == iterNucOwnerDaughterL )
			{
				iterNucOwnerDaughterL->removeSupervoxelFromNucleus( assignmentId[count] );
				//if( ret > 0 )
				//	cout<<"WARNING: lineageHyperTree::extendDeadNuclei: supervoxel not found to be removed from nucleus"<<endl;
			}
			else{
				iterNucOwnerDaughterR->removeSupervoxelFromNucleus( assignmentId[count] );
				//if( ret > 0 )
				//	cout<<"WARNING: lineageHyperTree::extendDeadNuclei: supervoxel not found to be removed from nucleus"<<endl;
			}
			iterNucNew->addSupervoxelToNucleus( assignmentId[count] );
			assignmentId[count]->treeNode.setParent( iterNucNew );
		}
	}


	//make sure original nuclei still has some supervoxels associated
	if( ( (intIsCellDivision & 0x0001) != 0 ) && ( iterNucOwnerDaughterL->treeNode.getNumChildren() == 0 ) )
	{
		int TMaux = iterNucOwnerDaughterL->TM;
		delete iterNucOwnerDaughterL->treeNodePtr;
		iterNucOwner->treeNodePtr->left = NULL;
		lht.nucleiList[ TMaux ].erase( iterNucOwnerDaughterL );
	}
	if(  ( (intIsCellDivision & 0x0010) != 0 ) && ( iterNucOwnerDaughterR->treeNode.getNumChildren() == 0 ) )
	{
		int TMaux = iterNucOwnerDaughterR->TM;
		delete iterNucOwnerDaughterR->treeNodePtr;
		iterNucOwner->treeNodePtr->right = NULL;
		lht.nucleiList[ TMaux ].erase( iterNucOwnerDaughterR );
	}	


	return extendedLineages;//returns 1 if extension was achieved
}


//============================================================================
template int parseNucleiList2TGMM<float>(std::vector<GaussianMixtureModel*> &vecGM, lineageHyperTree &lht,int frame, bool regularizeW, float thrDist2LargeDisplacement);
template int parseNucleiList2TGMM<unsigned short int>(std::vector<GaussianMixtureModel*> &vecGM, lineageHyperTree &lht,int frame, bool regularizeW, float thrDist2LargeDisplacement);
template int parseNucleiList2TGMM<unsigned char>(std::vector<GaussianMixtureModel*> &vecGM, lineageHyperTree &lht,int frame, bool regularizeW, float thrDist2LargeDisplacement);
