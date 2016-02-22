
#include <stdio.h>
#include <malloc.h>
#include <stdlib.h> 
/* The head of strcat()*/
#include <string.h>

#include "graph.h"
#include "timer.h"
#include "algorithm.h"


/*
Name: MAIN
Copyright: 
Author: Xuan Luo
Date: 14/01/16 10:43
Description: Test the function in the graph.c
*/


int main(int argc, char *argv[])
{

	/* declarations */
	Graph ** g;
	/* check the algorithm running on cpu */
	Graph_cpu * origin_g;
	DataSize * dsize;
	char *filename;
	/* Record the path to [input].vertices */
	char *filename_v;
	/* Record the path to [input].edges */
	char *filename_e;
	int gpu_num;
	/* CPU need to known all the copy number of vertices in the gpus */
	int *copy_num;
	int **position_id;
    int *value_cpu;
    int *value_gpu;

	if(argc<8)
	{
		printf("Input Command is Error!\n");
		printf("Description:\n");
		printf("Input filename1 Input filename2 vertex_num  edge_num   max_part_vertex_num  max_part_edge_num  gpu_num \n ");
		printf("eg: \/home\/xxling\/amazon.vertices \/hofme\/xxling\/amazon.edges 735322 5158012 356275 880813 4 \n"); 
		printf("Note:\n");
		/* TODO print detailed informatition about input command. */
		printf("....to be continued ...\n");
		return 0;        
	}

	/* malloc*/
	dsize=(DataSize *)malloc(sizeof(DataSize));

	filename_v=argv[1];
	filename_e=argv[2];

	dsize->vertex_num=atoi(argv[3]);
	dsize->edge_num=atoi(argv[4]);
	dsize->max_part_vertex_num=atoi(argv[5]);
	dsize->max_part_edge_num=atoi(argv[6]);
	gpu_num=atoi(argv[7]);

	copy_num=(int *)malloc(sizeof(int)*(dsize->vertex_num));
	memset(copy_num,0,sizeof(int)*(dsize->vertex_num));

	g=Initiate_graph (gpu_num,dsize);
	read_graph_vertices(filename_v,g,gpu_num,copy_num);
	read_graph_edges(filename_e,g,gpu_num,copy_num);

    
    int edge_num=dsize->edge_num;
    origin_g=read_graph_edges_again(filename_e,edge_num);
	   
    /* renumber the index of vertex in GPU */
    //coding(g,gpu_num);

    checkGraphvalue(g,dsize,gpu_num);

    coding(g,gpu_num);
    position_id=recordPosition(g,gpu_num,dsize->vertex_num);

    printf("After coding ......\n");
    checkGraphvalue(g,dsize,gpu_num);
	printf("\ncopy_num :\n");
	checkvalue_s(copy_num,dsize->vertex_num);


    int vertex_num=dsize->vertex_num;
    int first_vertex=3;
    value_cpu=(int *)malloc(sizeof(int)*vertex_num);
    value_gpu=(int *)malloc(sizeof(int)*vertex_num);
    bfs_cpu(origin_g,value_cpu,dsize,first_vertex);
    print_bfs_values(value_cpu,vertex_num);
    free(origin_g);
    
    //bfs_gpu(g,gpu_num,value_gpu,dsize,first_vertex,copy_num);
    //check_value_s(value_cpu,value_gpu);
    
	return 0; 
}

