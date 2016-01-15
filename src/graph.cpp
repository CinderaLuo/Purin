#pragma   once 

#include <stdio.h>
#include <malloc.h>
/* The head file of perror(), atoi() */
#include <stdlib.h> 
/* The head file of strstr()*/
#include <string.h>

/* The function and the graph data structure is defined in the "graph.h" */
#include "graph.h"

/*
Name: GRAPH_H
Copyright: 2016
Author: Xuan Luo
Date: 13/01/16 16:24
Description: Using the VGP to partition the graph.
This file is used to read the result of VGP.

VGP£ºA software package for one-pass Vertex-cut balanced Graph Partitioning.
"F. Petroni, L. Querzoni, G. Iacoboni, K. Daudjee and S. Kamali: "Hdrf: Efficient stream-based partitioning for power-law graphs". CIKM, 2015." 
HDRF has been integrated in GraphLab PowerGraph!             
*/

/*Check if a is bigger than b, if yes, then exit program */
#define LT(a,b) if(a>b){\
	printf("The result of graph partition is error! \n "); \
	exit(1);\
}
/* Allocate the memory for each machine to store the graph data */ 
/* Add: use size to record the graph data informatition */
Graph ** Initiate_graph (int gpu_num, DataSize *size )
{
	Graph **g=(Graph **)malloc(sizeof(Graph*)*gpu_num);
	for(int i=0;i<gpu_num;i++)
	{
		/* Easy to forget */
		g[i]=(Graph *)malloc(sizeof(Graph)*gpu_num);
		g[i]->edge_num=0;
		g[i]->vertex_num=0;
		g[i]->edge_outer_num=0;
		g[i]->vertex_outer_num=0;

		/*Allocte the memory to the array in graph_h*/
		g[i]->edge_src=(int *)malloc(sizeof(int)*(size->max_part_edge_num));
		g[i]->edge_dst=(int *)malloc(sizeof(int)*(size->max_part_edge_num));
		g[i]->vertex_id=(int *)malloc(sizeof(int)*(size->max_part_vertex_num));
		g[i]->vertex_outer_id=(int *)malloc(sizeof(int)*(size->max_part_vertex_num));
	}
	printf("Malloc Finished!\n");
	return g;
}

/* Read the file from [output-name].vertices which is the partition result of vertice */
/* Add : copy_num[vertex_id-1] is the copy number of vertex_id in all gpus */
/*       In file, the partition ID from 0 */
void read_graph_vertices(char *  filename,Graph **g,int  gpu_num,int *copy_num)
{
	char line[1024]; 
	char *loc=line;
	int vertex_id;
	int partition_id;
	/* vertex_outer_num[i] record the number of OUTER in gpu(i)*/
	int *vertex_outer_num;
	/* vertex_num[i] record the number of vertice in gpu(i)*/
	int *vertex_num;
	/* check whether the vertex is OUTER or not in each line. If OUTER, flag is true */ 
	bool flag=true;
	int tmp_num=0;
	int cp_num=0;
	FILE *f=NULL;


	/* try to open the file */
	f=fopen(filename,"r");
	if(f==NULL)
	{
		fprintf(stderr,"File open failed : %s ", filename);
		perror("");
		exit(1);           
	}
	printf("Reading.....\n");
	int i=0;
	while(fgets(line,1024,f)!=NULL)
	{
         
		/* process each line in file */
		cp_num=0;
		flag=true;

		loc=line;
		/* first number*/
		vertex_id=(int)atoi(line);
		
		/* second number */
		loc=strstr(line," ");
		//loc=loc+1;
		partition_id=(int)atoi(loc);
		LT(partition_id,gpu_num);
		tmp_num=g[partition_id]->vertex_num;
		g[partition_id]->vertex_id[tmp_num++]=vertex_id;
		g[partition_id]->vertex_num=tmp_num;
		cp_num++;
        

		/* third number and later */
		loc=strstr(loc+1," ");
		if(loc==NULL) flag=false;
		while(loc!=NULL)
		{
			/* record the OUTER */
			tmp_num=g[partition_id]->vertex_outer_num;
			g[partition_id]->vertex_outer_id[tmp_num++]=vertex_id;
			g[partition_id]->vertex_outer_num=tmp_num;

			partition_id=(int)atoi(loc);
			LT(partition_id,gpu_num);
			tmp_num=g[partition_id]->vertex_num;
			g[partition_id]->vertex_id[tmp_num++]=vertex_id;
			g[partition_id]->vertex_num=tmp_num;
			loc=strstr(loc+1," ");
			cp_num++;   
		}
		if(flag==true)
		{
			tmp_num=g[partition_id]->vertex_outer_num;
			g[partition_id]->vertex_outer_id[tmp_num++]=vertex_id;
			g[partition_id]->vertex_outer_num=tmp_num;
		}
		copy_num[vertex_id-1]=cp_num;           
	}   
}
/* Read the file from [output-name].edges which is the partition result of edge list */
void read_graph_edges(char * filename,Graph *g, int gpu_num,int *copy_num)
{
}
