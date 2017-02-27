/*
 * testKnnResults.cpp
 *
 *  Created on: Jul 18, 2011
 *      Author: amatf
 */
#include "testKnnResults.h"
#include "../kdtree.cpp"
#include <vector>
#include "../GaussianMixtureModel.h"



bool testKnnResults(float *ref,float *query,int *ind,int ref_nb,int query_nb,int dimsImage,int maxGaussiansPerVoxel)
{


		//-------------run kdtree to check results-----------------------------------------------------------
		vector<GaussianMixtureModel*> vecGM;
		float scale[3]={1.0f,1.0f,1.0f};
		for(int kk=0;kk<3*ref_nb;kk+=3)
		{
			GaussianMixtureModel *auxGM=new GaussianMixtureModel(vecGM.size(),scale);
			auxGM->m_k(0)=ref[kk];
			auxGM->m_k(1)=ref[kk+1];
			auxGM->m_k(2)=ref[kk+2];
			vecGM.push_back(auxGM);
		}
		KDTree<GaussianMixtureModel> kdtreeGaussian(dimsImage);
		kdtreeGaussian.balanceTree(vecGM);//this function rearranges the order of elements in vecGM
		for(unsigned int kk=0;kk<vecGM.size();kk++) vecGM[kk]->updateCenter();
		//--------------------------------------------------------------------------------------------------

		float cc[3];
		float radius;
		int ss=0;
		GaussianMixtureModel *GM;
		for(int kk=0;kk<query_nb;kk++)
		{
			printf("==================Checking results for query point %d==============\n",kk);
			cc[0]=query[3*kk];cc[1]=query[3*kk];cc[2]=query[3*kk+2];
			kdtreeGaussian.findNearest(vecGM,cc,maxGaussiansPerVoxel,1e32,&radius);
			while(!kdtreeGaussian.nearest_queue.empty())
			{
				GM=kdtreeGaussian.nearest_queue.top();
				printf("Selected neigh id=%d;CUDA id=%d\n",GM->id,ind[ss]);
				kdtreeGaussian.nearest_queue.pop();
				ss++;
			}
		}
		for(unsigned int kk=0;kk<vecGM.size();kk++) delete vecGM[kk];

		return true;
}

