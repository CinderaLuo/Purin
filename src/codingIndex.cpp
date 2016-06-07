#if __GNUC__>2
#include <ext/hash_map>
using namespace __gnu_cxx;
#else
#include <hash_map>
using namespace stdext;
#endif

#include <stdio.h>
#include <string.h>
#include "graph.h"

/*
Name: CODE
Copyright: 
Author: Xuan Luo
Date: 19/01/16  09:37
Description: Index of vertices should be changed before its values assigned to gpu.
             g[i]->vertex_id[m]=k  record the comparison table of vertex index.
             It means that vertex k becomes vertex m in gpu i.
*/

/* valueVertices d[i]->values[m] represents the values of the vertex (ID is i+1) in one gpu   */
/* valueVertices d[i]->partition_id[m] reprents the id of gpu */
struct valueVertices
{

	int *values;
	int *partitition_id;
};
void print_error() { 
	printf("The ID of edge list is error. Please check the preprocessing!\n"); 
		exit(1); 
}
/*1.change the id in g  2. the record the id table in table and m_value*/

void coding(Graph **g, int gpu_num)
{
	int vertex_num_part;
	int edge_outer_num_part,edge_inner_num_part;
	hash_map<int ,int> map_id;
	hash_map<int, int>::iterator ite_map;
    typedef pair <int, int> Int_Pair;	
	int tmp_id;
	for (int i = 0; i <gpu_num ; ++i)
	{
		map_id.clear();
		printf("gpu %d is coding...\n",i);
		vertex_num_part=g[i]->vertex_num;
		edge_outer_num_part=g[i]->edge_outer_num;
		edge_inner_num_part=g[i]->edge_num - edge_outer_num_part;
		/* malloc table*/

		//pragma omp parallel for
		for(int j=0;j<vertex_num_part;j++)
		{
			//map_id[j]=g[i]->vertex_id[j];
			map_id.insert(Int_Pair(g[i]->vertex_id[j],j));
		}
		//pragma omp parallel for
		for(int j=0;j<edge_outer_num_part;j++)
		{
			/*src vertex*/
			tmp_id=g[i]->edge_outer_src[j];
			ite_map=map_id.find(tmp_id);
			if(ite_map==map_id.end()) 
				print_error();
			g[i]->edge_outer_src[j]=ite_map->second;

			/* destination vertex */
			tmp_id=g[i]->edge_outer_dst[j];
			ite_map=map_id.find(tmp_id);
			if(ite_map==map_id.end()) 
				print_error();
			g[i]->edge_outer_dst[j]=ite_map->second;

		}
		//pragma omp parallel for
		for(int j=0;j<edge_inner_num_part;j++)
		{
			/*src vertex*/
			tmp_id=g[i]->edge_inner_src[j];
			ite_map=map_id.find(tmp_id);
			if(ite_map==map_id.end()) 
				print_error();
			g[i]->edge_inner_src[j]=ite_map->second;

			/* destination vertex */
			tmp_id=g[i]->edge_inner_dst[j];
			ite_map=map_id.find(tmp_id);
			if(ite_map==map_id.end()) 
				print_error();
			g[i]->edge_inner_dst[j]=ite_map->second;
		}
	}
}

/* position_id[vertex_id][gpu_id]=new_vertex_id */
/* record position of every vertex*/
int ** recordPosition(Graph **g, int gpu_num, int vertex_num)
{
	int ** position_id;
	int vertex_id;
	printf("malloc *position_id\n");
	position_id=(int **)malloc(sizeof(int *)*vertex_num);
	//omp parallel for 
	printf("malloc position_id[i]\n");
	for (int i = 0; i < vertex_num; ++i)
	{
		position_id[i]=(int*)malloc(sizeof(int)*gpu_num);
		memset(position_id[i],MAX_POSITION, sizeof(int)*gpu_num);
	}
	printf("malloc is ok!\n");
	for (int i = 0; i < gpu_num; ++i)
	{
		for(int j=0; j < g[i]->vertex_num;++j)
		{
			vertex_id=g[i]->vertex_id[j];
			vertex_id=vertex_id-1;
            position_id[vertex_id][i]=j;
		}
	}
	/* print */
	printf("Position_id\n");
	printf("vertex_id ( partition_id, p_id)\n");
	for (int i = 0; i < vertex_num; ++i)
	{
		for (int j = 0; j < gpu_num; ++j)
		{
			printf("%d ( %d , %d )\n",i+1,j,position_id[i][j]);
		}
	}
	return position_id;
}
