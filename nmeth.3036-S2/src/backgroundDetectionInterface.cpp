/*
 *
 * \brief Interface to backgroundDetection lib to find incoherent tracks that look like background
 *        
 *
 */

#include <iostream>
#include <unordered_map>
#include <list>

#if (!defined(_WIN32)) && (!defined(_WIN64))
#include <sys/stat.h>
#endif

#include "backgroundDetectionInterface.h"
#include "backgroundClassifier.h"
#include "gentleBoost/gentleBoost.h"
#include "GaussianMixtureModel_Redux.h"

using namespace std;




//======================================================================================================
int setProbBackgroundTracksAtTM(lineageHyperTree &lht, int TM, int temporalWindowSizeForBackground, string &classifierFilename, int devCUDA)
{
	//features: have to match the values used during training in mainBackgroundDetection.cpp::mainExtractFeatures()
	//nearest neighbors features ("social" features)
	unsigned int KmaxNumNNsupervoxel = 10;
	float KmaxDistKNNsupervoxel = 1e32;
	


	//set parameters
	backgroundDetectionFeatures::setTemporalWindowSize( temporalWindowSizeForBackground );

	//read classifier
	vector<vector< treeStump > > classifier;
	int err = loadClassifier(classifier, classifierFilename);
	if( err > 0) 
		return err;


	//run classifier for each nuclei
	backgroundDetectionFeatures f;
	unsigned long long int numFeatures = backgroundDetectionFeatures::getNumFeatures();
	unsigned long long int numSamples = 1;
	unsigned int debugC = 0;
	for(list<nucleus>::iterator iterN = lht.nucleiList[TM].begin(); iterN != lht.nucleiList[TM].end(); ++iterN, debugC++ )
	{
		if(  f.calculateFeatures(iterN->treeNodePtr, devCUDA) == 0 ) //we have good features
		{	
			boostingTreeClassifier(&(f.f[0]),&(iterN->probBackground), classifier, numSamples, numFeatures);			
		}else{//undetermined: we always initialize it to zero, so we do not need to anything
			//iterN->probBackground = 0.0;
			//forward propagation
			if( iterN->treeNodePtr->parent != NULL )
			{
				iterN->probBackground = iterN->treeNodePtr->parent->data->probBackground;
			}else{//at the beginning of the lineage
				iterN->probBackground = -1.0f;//to indicate we did not calculate it here
			}
		}		
	}		
	

	return 0;
}

//========================================================================================================
int applyProbBackgroundMinMaxRulePerBranch(const string &basenameTGMMresult, int iniFrame, int endFrame, const string& outputFolder, float thrBackground)
{
	#if defined(_WIN32) || defined(_WIN64)
	if (GetFileAttributes(outputFolder.c_str()) == INVALID_FILE_ATTRIBUTES)
	{
		string cmd=string("mkdir " + outputFolder);
		int error=system(cmd.c_str());
		if(error>0)
		{
			cout<<"ERROR ("<<error<<"): generating debug path "<<outputFolder<<endl;
			cout<<"Wtih command "<<cmd<<endl;
			return error;
		}
	}

#else
	struct stat St;
	if (stat( outputFolder.c_str(), &St ) != 0)//check if folder exists
	{
		string cmd=string("mkdir " + outputFolder);
		int error=system(cmd.c_str());
		if(error>0)
		{
			cout<<"ERROR ("<<error<<"): generating debug path "<<outputFolder<<endl;
			cout<<"Wtih command "<<cmd<<endl;
			return error;
		}
	}
#endif

	//I use list instead of vector because when it resizes dynamically it does not copy all the binary trees objects. Thus, changing all the pointer references.
	list< BinaryTree<float> > btBckg;//for each element in a lineage we store the background probability score

	GaussianMixtureModelRedux auxGM;	
	TreeNode<float> *auxNode;
	vector< vector< TreeNode<float>* > >  mapFrameBlob2bt(endFrame+1);//stores all the pointers between (frameId,blobId) -> binary tree, so I can resave later
	for(int frame = iniFrame; frame <= endFrame; frame++)
	{
		cout<<"Background filter: Reading frame "<<frame<<" for forward-backward pass"<<endl;
		//read XML file
		char buffer[256];
		sprintf(buffer,"%.4d",frame);
		string itoa(buffer);
		string GMxmlFilename = string(basenameTGMMresult + itoa + ".xml");
		
		XMLNode xMainNode = XMLNode::openFileHelper(GMxmlFilename.c_str(),"document");
		int n = xMainNode.nChildNode("GaussianMixtureModel");
		mapFrameBlob2bt[frame].resize(n);
		//iterate over each element
		for(int ii = 0; ii < n; ii++)
		{			
			auxGM = GaussianMixtureModelRedux(xMainNode,ii);
			TreeNode<float> *nodeCh = new TreeNode<float>();//by default everything is initialized to NULL
			nodeCh->data = auxGM.beta_o;//beta_o stores the background information			


			mapFrameBlob2bt[frame][ii] = nodeCh;			
			if( auxGM.parentId < 0 || frame == iniFrame)//new lineage
			{				
				btBckg.push_back( BinaryTree<float>() );
				btBckg.back().SetMainRoot( nodeCh );

			}else{//existing lineage
				
				auxNode = mapFrameBlob2bt[frame-1][ auxGM.parentId ];				
				nodeCh->parent = auxNode;


				if( auxNode->left == NULL )
				{
					auxNode->left = nodeCh;
				}else if( auxNode->right == NULL )
				{
					auxNode->right = nodeCh;
				}else{
					cout<<"ERROR: at applyProbBackgroundMinMaxRulePerBranch: node already has two children"<<endl;
					return 1;
				}

				//we CANNOT propagate forward information at this point because we do not have a complete lineage, so we do not know when branches start or end				
			}
		}		
	}


	//forward-backward pass: find all the leaves and splitting points and traverse that branch backward
	cout<<"Background filter: forward-backward pass for "<<btBckg.size()<<" lineages"<<endl;
	size_t countI = 0;
	for(list<BinaryTree<float> >::iterator iter = btBckg.begin(); iter != btBckg.end(); ++iter, countI++ )
	{				
		queue< TreeNode<float>* > q;
		q.push( iter->pointer_mainRoot() );

		while( q.empty() == false )
		{
			auxNode = q.front();
			q.pop();
			
			if( auxNode->left != NULL )
				q.push( auxNode->left );
			if( auxNode->right != NULL )
				q.push( auxNode->right );

			//forward pass			
			if( auxNode->parent != NULL && auxNode->parent->getNumChildren() == 1 )//they are part of the same branch
			{
				auxNode->data = std::max(auxNode->data, auxNode->parent->data);//propagate max forward
			}

			if( auxNode->getNumChildren() != 1 )//leave or branch->traverse that branch backward to propagate information
			{
				while( auxNode->parent != NULL && auxNode->parent->getNumChildren() < 2 )
				{
					auxNode->parent->data = std::max(auxNode->data, auxNode->parent->data);//propagate max backwards
					//update backwards
					auxNode = auxNode->parent;
				}
			}
		}

	}


	//now all the elements in a branch have the same probability (the maximum)->delete all the branches that do not satisfy that criteria
	long long unsigned int totalBlobs = 0, totalBlobsBackground = 0;
	vector< int > mapParentId, mapParentIdOld;
	for(int frame = iniFrame; frame <= endFrame; frame++)
	{
		cout<<"Background filter: updating frame "<<frame<<" after forward/backward pass"<<endl;
		//read XML file
		char buffer[256];
		sprintf(buffer,"%.4d",frame);
		string itoa(buffer);
		string GMxmlFilename = string(basenameTGMMresult + itoa + ".xml");
		
		XMLNode xMainNode = XMLNode::openFileHelper(GMxmlFilename.c_str(),"document");
		int n = xMainNode.nChildNode("GaussianMixtureModel");
		mapParentId.resize(n);
		int countB = 0;//to keep count of the number of blobs written

		//prepare XML ooutput file
		string XMLout(outputFolder + "GMEMfinalResult_frame" + itoa + ".xml");
		ofstream outXML(XMLout.c_str());
		if( outXML.is_open() == false )
		{
			cout<<"ERROR: at applyProbBackgroundMinMaxRulePerBranch: opening output file "<<XMLout<<endl;
			//release memory
			for(list<BinaryTree<float> >::iterator iter = btBckg.begin(); iter != btBckg.end(); ++iter )
				iter->clear();
			return 2;
		}
		GaussianMixtureModelRedux::writeXMLheader(outXML);


		for(int ii = 0; ii < n; ii++)
		{
			auxGM = GaussianMixtureModelRedux(xMainNode,ii);
			//------------debugging---------------------------------------
			//hola: uncomment this and delete extra code for debugging
			//auxGM.beta_o = mapFrameBlob2bt[frame][ii]->data;
			//if(1)
			//----------------------------------------------
			if( mapFrameBlob2bt[frame][ii]->data < thrBackground )//this blob needs to be written
			{
				auxGM.id = countB;
				if( auxGM.parentId >= 0 )
					auxGM.parentId = mapParentIdOld[ auxGM.parentId ];

				//write out blob
				//write solution
				auxGM.writeXML(outXML);
				
				//update variables
				mapParentId[ii] = countB;
				countB++;
			}else{
				mapParentId[ii] = -1;
				totalBlobsBackground++;
			}
		}

		//close XML file for this frame
		GaussianMixtureModelRedux::writeXMLfooter(outXML);
		outXML.close();

		//update variables
		totalBlobs += (long long unsigned int)(n);
		mapParentIdOld = mapParentId;
		mapParentId.clear();
	}

	//release memory
	for(list<BinaryTree<float> >::iterator iter = btBckg.begin(); iter != btBckg.end(); ++iter )
		iter->clear();

	cout<<"Deleted "<<totalBlobsBackground<<" out of "<<totalBlobs<<" with background thr = "<<thrBackground<<endl;

	return 0;
}


//========================================================================================================
int applyProbBackgroundAvgRulePerBranch(const string &basenameTGMMresult, int iniFrame, int endFrame, const string& outputFolder, float thrBackground)
{
	#if defined(_WIN32) || defined(_WIN64)
	if (GetFileAttributes(outputFolder.c_str()) == INVALID_FILE_ATTRIBUTES)
	{
		string cmd=string("mkdir " + outputFolder);
		int error=system(cmd.c_str());
		if(error>0)
		{
			cout<<"ERROR ("<<error<<"): generating debug path "<<outputFolder<<endl;
			cout<<"Wtih command "<<cmd<<endl;
			return error;
		}
	}

#else
	struct stat St;
	if (stat( outputFolder.c_str(), &St ) != 0)//check if folder exists
	{
		string cmd=string("mkdir " + outputFolder);
		int error=system(cmd.c_str());
		if(error>0)
		{
			cout<<"ERROR ("<<error<<"): generating debug path "<<outputFolder<<endl;
			cout<<"Wtih command "<<cmd<<endl;
			return error;
		}
	}
#endif

	//I use list instead of vector because when it resizes dynamically it does not copy all the binary trees objects. Thus, changing all the pointer references.
	list< BinaryTree<float> > btBckg;//for each element in a lineage we store the background probability score

	GaussianMixtureModelRedux auxGM;	
	TreeNode<float> *auxNode;
	vector< vector< TreeNode<float>* > >  mapFrameBlob2bt(endFrame+1);//stores all the pointers between (frameId,blobId) -> binary tree, so I can resave later
	for(int frame = iniFrame; frame <= endFrame; frame++)
	{
		cout<<"Background filter: Reading frame "<<frame<<" for forward pass"<<endl;
		//read XML file
		char buffer[256];
		sprintf(buffer,"%.4d",frame);
		string itoa(buffer);
		string GMxmlFilename = string(basenameTGMMresult + itoa + ".xml");
		
		XMLNode xMainNode = XMLNode::openFileHelper(GMxmlFilename.c_str(),"document");
		int n = xMainNode.nChildNode("GaussianMixtureModel");
		mapFrameBlob2bt[frame].resize(n);
		//iterate over each element
		for(int ii = 0; ii < n; ii++)
		{			
			auxGM = GaussianMixtureModelRedux(xMainNode,ii);
			TreeNode<float> *nodeCh = new TreeNode<float>();//by default everything is initialized to NULL
			nodeCh->data = auxGM.beta_o;//beta_o stores the background information			


			mapFrameBlob2bt[frame][ii] = nodeCh;			
			if( auxGM.parentId < 0 || frame == iniFrame)//new lineage
			{				
				btBckg.push_back( BinaryTree<float>() );
				btBckg.back().SetMainRoot( nodeCh );

			}else{//existing lineage
				
				auxNode = mapFrameBlob2bt[frame-1][ auxGM.parentId ];				
				nodeCh->parent = auxNode;


				if( auxNode->left == NULL )
				{
					auxNode->left = nodeCh;
				}else if( auxNode->right == NULL )
				{
					auxNode->right = nodeCh;
				}else{
					cout<<"ERROR: at applyProbBackgroundMinMaxRulePerBranch: node already has two children"<<endl;
					return 1;
				}
			}
		}		
	}

	//backward pass: find all the leaves and splitting points and traverse that branch backward
	cout<<"Background filter: backward pass for "<<btBckg.size()<<" lineages"<<endl;
	size_t countI = 0;
	for(list<BinaryTree<float> >::iterator iter = btBckg.begin(); iter != btBckg.end(); ++iter, countI++ )
	{				
		queue< TreeNode<float>* > q;
		q.push( iter->pointer_mainRoot() );

		while( q.empty() == false )
		{
			auxNode = q.front();
			q.pop();
			
			if( auxNode->left != NULL )
				q.push( auxNode->left );
			if( auxNode->right != NULL )
				q.push( auxNode->right );

			

			if( auxNode->parent != NULL && auxNode->parent->getNumChildren() == 1 )//this is not the beginning of a branch->propagate average forward
			{
				auxNode->data += auxNode->parent->data;
			}//else: it is a starting point, so leave the current prob

			if( auxNode->getNumChildren() != 1 )//leave or branch->traverse that branch backward
			{
				TreeNode<float>* auxNodeOrig = auxNode;
				//first count branch length
				int count = 1;
				while( auxNode->parent != NULL && auxNode->parent->getNumChildren() < 2 )
				{
					count++;
					//update backwards
					auxNode = auxNode->parent;
				}


				auxNode = auxNodeOrig;
				float avg = auxNode->data / ((float) count);
				while( auxNode->parent != NULL && auxNode->parent->getNumChildren() < 2 )
				{
					auxNode->data = avg;
					//update backwards
					auxNode = auxNode->parent;
				}
			}
		}

	}


	//now all the elements in a branch have the same probability (the maximum)->delete all the branches that do not satisfy that criteria
	long long unsigned int totalBlobs = 0, totalBlobsBackground = 0;
	vector< int > mapParentId, mapParentIdOld;
	for(int frame = iniFrame; frame <= endFrame; frame++)
	{
		cout<<"Background filter: updating frame "<<frame<<" after forward/backward pass"<<endl;
		//read XML file
		char buffer[256];
		sprintf(buffer,"%.4d",frame);
		string itoa(buffer);
		string GMxmlFilename = string(basenameTGMMresult + itoa + ".xml");
		
		XMLNode xMainNode = XMLNode::openFileHelper(GMxmlFilename.c_str(),"document");
		int n = xMainNode.nChildNode("GaussianMixtureModel");
		mapParentId.resize(n);
		int countB = 0;//to keep count of the number of blobs written

		//prepare XML ooutput file
		string XMLout(outputFolder + "GMEMfinalResult_frame" + itoa + ".xml");
		ofstream outXML(XMLout.c_str());
		if( outXML.is_open() == false )
		{
			cout<<"ERROR: at applyProbBackgroundMinMaxRulePerBranch: opening output file "<<XMLout<<endl;
			//release memory
			for(list<BinaryTree<float> >::iterator iter = btBckg.begin(); iter != btBckg.end(); ++iter )
				iter->clear();
			return 2;
		}
		GaussianMixtureModelRedux::writeXMLheader(outXML);


		for(int ii = 0; ii < n; ii++)
		{
			auxGM = GaussianMixtureModelRedux(xMainNode,ii);
			if( mapFrameBlob2bt[frame][ii]->data < thrBackground )//this blob needs to be written
			{
				auxGM.id = countB;
				if( auxGM.parentId >= 0 )
					auxGM.parentId = mapParentIdOld[ auxGM.parentId ];

				//write out blob
				//write solution
				auxGM.writeXML(outXML);
				
				//update variables
				mapParentId[ii] = countB;
				countB++;
			}else{
				mapParentId[ii] = -1;
				totalBlobsBackground++;
			}
		}

		//close XML file for this frame
		GaussianMixtureModelRedux::writeXMLfooter(outXML);
		outXML.close();

		//update variables
		totalBlobs += (long long unsigned int)(n);
		mapParentIdOld = mapParentId;
		mapParentId.clear();
	}

	//release memory
	for(list<BinaryTree<float> >::iterator iter = btBckg.begin(); iter != btBckg.end(); ++iter )
		iter->clear();

	cout<<"Deleted "<<totalBlobsBackground<<" out of "<<totalBlobs<<" with background thr = "<<thrBackground<<endl;

	return 0;
}


//========================================================================================================
int applyProbBackgroundHysteresisRulePerBranch(const string &basenameTGMMresult, int iniFrame, int endFrame, const string& outputFolder, float thrLow, float thrHigh)
{

#if defined(_WIN32) || defined(_WIN64)
	if (GetFileAttributes(outputFolder.c_str()) == INVALID_FILE_ATTRIBUTES)
	{
		string cmd=string("mkdir " + outputFolder);
		int error=system(cmd.c_str());
		if(error>0)
		{
			cout<<"ERROR ("<<error<<"): generating debug path "<<outputFolder<<endl;
			cout<<"Wtih command "<<cmd<<endl;
			return error;
		}
	}

#else
	struct stat St;
	if (stat( outputFolder.c_str(), &St ) != 0)//check if folder exists
	{
		string cmd=string("mkdir " + outputFolder);
		int error=system(cmd.c_str());
		if(error>0)
		{
			cout<<"ERROR ("<<error<<"): generating debug path "<<outputFolder<<endl;
			cout<<"Wtih command "<<cmd<<endl;
			return error;
		}
	}
#endif


	//I use list instead of vector because when it resizes dynamically it does not copy all the binary trees objects. Thus, changing all the pointer references.
	list< BinaryTree<float> > btBckg;//for each element in a lineage we store the background probability score

	GaussianMixtureModelRedux auxGM;	
	TreeNode<float> *auxNode;
	vector< vector< TreeNode<float>* > >  mapFrameBlob2bt(endFrame+1);//stores all the pointers between (frameId,blobId) -> binary tree, so I can resave later
	for(int frame = iniFrame; frame <= endFrame; frame++)
	{
		cout<<"Background filter: Reading frame "<<frame<<" for forward-backward pass"<<endl;
		//read XML file
		char buffer[256];
		sprintf(buffer,"%.4d",frame);
		string itoa(buffer);
		string GMxmlFilename = string(basenameTGMMresult + itoa + ".xml");
		
		XMLNode xMainNode = XMLNode::openFileHelper(GMxmlFilename.c_str(),"document");
		int n = xMainNode.nChildNode("GaussianMixtureModel");
		mapFrameBlob2bt[frame].resize(n);
		//iterate over each element
		for(int ii = 0; ii < n; ii++)
		{			
			auxGM = GaussianMixtureModelRedux(xMainNode,ii);
			TreeNode<float> *nodeCh = new TreeNode<float>();//by default everything is initialized to NULL
			nodeCh->data = auxGM.beta_o;//beta_o stores the background information			


			mapFrameBlob2bt[frame][ii] = nodeCh;			
			if( auxGM.parentId < 0 || frame == iniFrame)//new lineage
			{				
				btBckg.push_back( BinaryTree<float>() );
				btBckg.back().SetMainRoot( nodeCh );

			}else{//existing lineage
				
				auxNode = mapFrameBlob2bt[frame-1][ auxGM.parentId ];				
				nodeCh->parent = auxNode;


				if( auxNode->left == NULL )
				{
					auxNode->left = nodeCh;
				}else if( auxNode->right == NULL )
				{
					auxNode->right = nodeCh;
				}else{
					cout<<"ERROR: at applyProbBackgroundMinMaxRulePerBranch: node already has two children"<<endl;
					return 1;
				}

				//we CANNOT propagate forward information at this point because we do not have a complete lineage, so we do not know when branches start or end				
			}
		}		
	}

	//backward pass to remove any -1 remaining in the prob background
	cout<<"Background filter: backward pass for interpolation"<<endl;
	for(list<BinaryTree<float> >::iterator iter = btBckg.begin(); iter != btBckg.end(); ++iter )
	{	

		if( iter->pointer_mainRoot()->data >= 0)
			continue;//no need to check further

		vector< TreeNode<float>* > lastNode;
		queue< TreeNode<float>* > q;
		q.push( iter->pointer_mainRoot() );

		while( q.empty() == false )
		{
			auxNode = q.front();
			q.pop();
			

			if( auxNode->data >= 0.0f )//we do not need to check further
			{
				lastNode.push_back( auxNode );//first node  with prob background
				continue;
			}

			if( auxNode->left != NULL )
				q.push( auxNode->left );
			if( auxNode->right != NULL )
				q.push( auxNode->right );						
		}

		for(size_t aa = 0; aa < lastNode.size(); aa++)
		{
			auxNode = lastNode[aa];
			while( auxNode->parent != NULL )
			{
				auxNode->parent->data = auxNode->data;
				auxNode = auxNode->parent;
			}
		}

	}	

	//forward-backward pass: find all the leaves and splitting points and traverse that branch backward
	cout<<"Background filter: forward-backward pass for "<<btBckg.size()<<" lineages"<<endl;
	size_t countI = 0;
	for(list<BinaryTree<float> >::iterator iter = btBckg.begin(); iter != btBckg.end(); ++iter, countI++ )
	{				
		queue< TreeNode<float>* > q;
		q.push( iter->pointer_mainRoot() );

		while( q.empty() == false )
		{
			auxNode = q.front();
			q.pop();
			
			if( auxNode->left != NULL )
				q.push( auxNode->left );
			if( auxNode->right != NULL )
				q.push( auxNode->right );
			

			//forward pass			
			if( auxNode->parent != NULL && auxNode->parent->getNumChildren() == 1 )//they are part of the same branch
			{
				if( auxNode->parent->data > thrHigh && auxNode->data > thrLow ) //we are within the hysteresis process
					auxNode->data = std::max(auxNode->data, auxNode->parent->data);//propagate max forward
			}

			if( auxNode->getNumChildren() != 1 )//leave or branch->traverse that branch backward to propagate information
			{
				while( auxNode->parent != NULL && auxNode->parent->getNumChildren() < 2 )
				{
					if( auxNode->data > thrHigh && auxNode->parent->data > thrLow ) //we are within the hysteresis process
						auxNode->parent->data = std::max(auxNode->data, auxNode->parent->data);//propagate max backwards
					//update backwards
					auxNode = auxNode->parent;
				}
			}
		}

	}	

	//now all the elements in a branch have the same probability (the maximum)->delete all the branches that do not satisfy that criteria
	long long unsigned int totalBlobs = 0, totalBlobsBackground = 0;
	vector< int > mapParentId, mapParentIdOld;
	for(int frame = iniFrame; frame <= endFrame; frame++)
	{
		cout<<"Background filter: updating frame "<<frame<<" after forward/backward pass"<<endl;
		//read XML file
		char buffer[256];
		sprintf(buffer,"%.4d",frame);
		string itoa(buffer);
		string GMxmlFilename = string(basenameTGMMresult + itoa + ".xml");
		
		XMLNode xMainNode = XMLNode::openFileHelper(GMxmlFilename.c_str(),"document");
		int n = xMainNode.nChildNode("GaussianMixtureModel");
		mapParentId.resize(n);
		int countB = 0;//to keep count of the number of blobs written

		//prepare XML ooutput file
		string XMLout(outputFolder + "GMEMfinalResult_frame" + itoa + ".xml");
		ofstream outXML(XMLout.c_str());
		if( outXML.is_open() == false )
		{
			cout<<"ERROR: at applyProbBackgroundMinMaxRulePerBranch: opening output file "<<XMLout<<endl;
			//release memory
			for(list<BinaryTree<float> >::iterator iter = btBckg.begin(); iter != btBckg.end(); ++iter )
				iter->clear();
			return 2;
		}
		GaussianMixtureModelRedux::writeXMLheader(outXML);


		for(int ii = 0; ii < n; ii++)
		{
			auxGM = GaussianMixtureModelRedux(xMainNode,ii);
			auxGM.beta_o = mapFrameBlob2bt[frame][ii]->data;//we apply the changes
			//---------------debugging------------------------------------
			//hola: uncomment this and delete extra code for debugging
			//auxGM.beta_o = mapFrameBlob2bt[frame][ii]->data;
			//if(1)
			//----------------------------------------------
			if( mapFrameBlob2bt[frame][ii]->data < thrHigh )//this blob needs to be written
			{
				auxGM.id = countB;
				if( auxGM.parentId >= 0 )
					auxGM.parentId = mapParentIdOld[ auxGM.parentId ];

				//write out blob
				//write solution
				auxGM.writeXML(outXML);
				
				//update variables
				mapParentId[ii] = countB;
				countB++;
			}else{
				mapParentId[ii] = -1;
				totalBlobsBackground++;
			}
		}

		//close XML file for this frame
		GaussianMixtureModelRedux::writeXMLfooter(outXML);
		outXML.close();

		//update variables
		totalBlobs += (long long unsigned int)(n);
		mapParentIdOld = mapParentId;
		mapParentId.clear();
	}

	//release memory
	for(list<BinaryTree<float> >::iterator iter = btBckg.begin(); iter != btBckg.end(); ++iter )
		iter->clear();

	cout<<"Deleted "<<totalBlobsBackground<<" out of "<<totalBlobs<<" with background thrHigh = "<<thrHigh<<"; thrLow="<<thrLow<<endl;
	return 0;
}

