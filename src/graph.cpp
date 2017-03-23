//#define PRINT_CHECK_G
#include <stdio.h>
#include <malloc.h>
/* The head file of perror(), atoi() */
#include <stdlib.h> 
/* The head file of strstr()*/
#include <string.h>
#include <math.h>

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
Graph**  Initiate_graph (int gpu_num, DataSize *size )
{
	Graph **g=(Graph **)malloc(sizeof(Graph*)*gpu_num);
	for(int i=0;i<gpu_num;i++)
	{
		/* Easy to forget */
		g[i]=(Graph *)malloc(sizeof(Graph));
		g[i]->edge_num=0;
		g[i]->vertex_num=0;
		g[i]->edge_outer_num=0;
		g[i]->vertex_outer_num=0;

		/*Allocte the memory to the array in graph_h*/
		g[i]->edge_outer_src=(int *)malloc(sizeof(int)*(size->max_part_edge_num));
		g[i]->edge_outer_dst=(int *)malloc(sizeof(int)*(size->max_part_edge_num));
		g[i]->edge_inner_src=(int *)malloc(sizeof(int)*(size->max_part_edge_num));
		g[i]->edge_inner_dst=(int *)malloc(sizeof(int)*(size->max_part_edge_num));
		g[i]->vertex_id=(int *)malloc(sizeof(int)*(size->max_part_vertex_num));
		g[i]->vertex_outer_id=(int *)malloc(sizeof(int)*(size->max_part_vertex_num));
	}
	printf("Malloc Finished!\n");
	return g;
}

/* Read the file from [output-name].vertices which is the partition result of vertice */
/* Add : copy_num[vertex_id-1] is the copy number of vertex_id in all gpus */
/*       In file, the partition ID from 0 */
void read_graph_vertices(char *  filename, Graph ** g,int  gpu_num,int *copy_num)
{
	char line[1024]; 
	char *loc=line;
	int vertex_id;
	int partition_id;
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
	printf("Reading  %s.....\n",filename);
	
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

/* add map_copy */
void read_graph_vertices_m(char *  filename, Graph **g,int  gpu_num,int *copy_num,int *map_copy)
{
	char line[1024]; 
	char *loc=line;
	int vertex_id;
	int partition_id;
	/* check whether the vertex is OUTER or not in each line. If OUTER, flag is true */ 
	bool flag=true;
	int tmp_num=0;
	int cp_num=0;
	int mp_copy=0;
	FILE *f=NULL;
	int tmp_size=log(gpu_num)/log(2);
	int size=0;
	if(pow(2,tmp_size)!=gpu_num)
          size=tmp_size+1;
      else
      	size=tmp_size;

	/* try to open the file */
	f=fopen(filename,"r");
	if(f==NULL)
	{
		fprintf(stderr,"File open failed : %s ", filename);
		perror("");
		exit(1);           
	}
	printf("Reading  %s.....\n",filename);
	
	while(fgets(line,1024,f)!=NULL)
	{
         
		/* process each line in file */
		cp_num=0;
		mp_copy=0;
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
		mp_copy=partition_id;
        

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
			mp_copy=mp_copy<<size;
			mp_copy=mp_copy||partition_id; 
		}
		if(flag==true)
		{
			tmp_num=g[partition_id]->vertex_outer_num;
			g[partition_id]->vertex_outer_id[tmp_num++]=vertex_id;
			g[partition_id]->vertex_outer_num=tmp_num;
		}
		copy_num[vertex_id-1]=cp_num;
		map_copy[vertex_id-1]=mp_copy;           
	}   
}


/* Read the file from [output-name].edges which is the partition result of edge list */
void read_graph_edges(char * filename,Graph **g, int gpu_num,int *copy_num)
{

	char line[1024];
	char *loc=line;
	int  edge_src,edge_dst;
	int partition_id;
	
	/* record the idx of edge_inner_src[] and edge_inner_dst[] */
	int *edge_inner_num;
	int tmp_num;
	FILE *f=NULL;

	f=fopen(filename,"r");
	edge_inner_num=(int *)malloc(sizeof(int)*gpu_num);
	memset(edge_inner_num,0,sizeof(int)*gpu_num);

	if(f==NULL)
	{
		fprintf(stderr,"File open failed : %s ", filename);
		perror("");
		exit(1);           
	}
    printf("Reading  %s.....\n",filename);
    while(fgets(line,1024,f)!=NULL)
    {
         /* process each line in file, each line is a edge list */
    	loc=line;

    	edge_src=(int)atoi(line);
    	loc=strstr(line,",");
    	edge_dst=(int)atoi(loc+1);
    	loc=strstr(loc+1," ");
    	partition_id=(int)atoi(loc);
    	LT(partition_id,gpu_num);
         /* Importation :  decide which edges should be processed firstly */
         // Anthor Method:
    	 // check whether edge_dst is included in g[partition_id]->vertice_outer_id[]
    	 // change the type of g[partition_id]->vertice_outer_id[]? vector?
    	if(copy_num[edge_dst-1]>1)
    	{
    		/* edge_dest is outer */
            tmp_num=g[partition_id]->edge_outer_num;
    		g[partition_id]->edge_outer_src[tmp_num]=edge_src;
    		g[partition_id]->edge_outer_dst[tmp_num]=edge_dst;
    		g[partition_id]->edge_outer_num=tmp_num+1;
    	}
    	else
    	{
    		/* edge_dest is inner */
            tmp_num=edge_inner_num[partition_id];
            g[partition_id]->edge_inner_src[tmp_num]=edge_src;
            g[partition_id]->edge_inner_dst[tmp_num]=edge_dst;
            edge_inner_num[partition_id]=tmp_num+1;
    	}
    	tmp_num=g[partition_id]->edge_num;
    	g[partition_id]->edge_num=tmp_num+1;
    }
}

void read_graph_size(Graph **g, DataSize *dsize, int gpu_num)
{

	int max_vertex_num=0;
	int max_edge_num=0;
   	for (int i = 0; i < gpu_num; ++i)
	{
		if(max_vertex_num<g[i]->vertex_num)
			max_vertex_num=g[i]->vertex_num;
		if (max_edge_num<g[i]->edge_num)
		    max_edge_num=g[i]->edge_num;
	}
	dsize->max_part_vertex_num=max_vertex_num;
	dsize->max_part_edge_num=max_edge_num;
}

/* Record the max size of outer edge lsits in GPUs which is used for determining the block size */
int max_num_outer_edge(Graph **g, int gpu_num)
{
	int max=0;
    for (int i = 0; i < gpu_num; ++i)
    {
    	if(max < g[i]->edge_outer_num)
    		max=g[i]->edge_outer_num;
    }
    return max;
}

/* Record the min size of outer edge lsits in GPUs which is used for determining the block size */
int min_num_outer_edge(Graph **g, int gpu_num)
{
	int min=g[0]->edge_outer_num;
	for (int i = 0; i < gpu_num; ++i)
	{
		if (min>g[i]->edge_outer_num)
		{
			min=g[i]->edge_outer_num;
		}
	}
	return min;
}

/* Do not think about preprocesing time */
/* The following functions just are used to [algorithm]_cpu() to check the correctness. */

/* In fact, read the [input].edges again. TODO: Integrate into read_graph_edges() */
Graph_cpu * read_graph_edges_again(char * filename, int edge_num)
{
    Graph_cpu *g=(Graph_cpu *)malloc(sizeof(Graph_cpu));
    g->edge_src=(int *)malloc(sizeof(int)*edge_num);
    g->edge_dst=(int *)malloc(sizeof(int)*edge_num);
    if(g==NULL)
    {
    	perror("Out of memory");
    	exit(1);
    }
    char line[1024];
    char *loc=line;
    int edge_src,edge_dst;

    FILE *f=NULL;
    f=fopen(filename,"r");
    if(f==NULL)
    {
       	fprintf(stderr,"File open failed : %s ", filename);
		perror("");
		exit(1);       
    }
    int index=0;
    printf("Reading  %s again .....\n",filename);
    while(fgets(line,1024,f)!=NULL)
    {
    	loc=line;
    	edge_src=(int)atoi(line);
    	loc=strstr(line,",");
    	edge_dst=(int)atoi(loc+1);
        g->edge_src[index]=edge_src;
        g->edge_dst[index]=edge_dst;
        index++;
        if (index>edge_num)
        {
        	printf("Edge_num is wrong!\n");
        	perror("");
        	exit(1);
        }
    }
    return g;
}

/* parse the edgelist in Graph_cpu to csr structure in Graph_cpu*/
void edgelsit_to_csr( Graph_cpu *g, int vertex_num, int edge_num)
{
   g->vertex_begin=(int *)malloc(sizeof(int)*(vertex_num+1));
   g->vertex_dst=(int *)malloc(sizeof(int)*edge_num);
   int index=0;
   int vertex_id=0;
   int count=0;
   int begin_id=0;
   g->vertex_begin[begin_id++]=index;
   printf("start\n");
   while(vertex_id<vertex_num)
   {
   	 for (int i = 0; i < edge_num; ++i)
   	 {
   		if (g->edge_src[i]==vertex_id)
   		{
   			g->vertex_dst[count++]= g->edge_dst[i];
   			index++;
   		}
   	 }
   	 g->vertex_begin[begin_id++]=index;
   	 vertex_id++;
   }
  printf("finish edgelist\n");
 #ifdef PRINT_CHECK_G
   printf("edge_src\n");
   for (int i = 0; i < edge_num; ++i)
   {
   	   printf("%d\t",g->edge_src[i]);
   }
   printf("\nedge_dst\n");
   for (int i = 0; i < edge_num; ++i)
   {
   	   printf("%d\t",g->edge_dst[i]);
   }
   printf("\nvertex_begin\n");
   for (int i = 0; i < begin_id; ++i)
   {
   	  printf("%d\t",g->vertex_begin[i]);
   }
   printf("\nvertex_dst\n");
   for (int i = 0; i < count; ++i)
   {
   	  printf("%d\t",g->vertex_dst[i] );
   }
   printf("\n");
 #endif 
}
