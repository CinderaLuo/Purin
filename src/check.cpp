
#pragma once 
#include <stdio.h>
#include "graph.h"

/*
Name: CHECK
Copyright: 
Author: Xuan Luo 
Date: 14/01/16 16:50
Description:check whether the value is right or not 
*/


/* print the value of array g */
void checkvalue(int * g, int size)
{
	for(int i=0;i<size;i++)
		printf("%d\t",g[i]);
	printf("\n");
}

/* check graph data structure */
void checkGraphvalue(Graph ** g, DataSize * size,int gpu_num)
{
	for(int i=0;i<gpu_num;i++)
	{
		printf("******GPU %d Information**********\n",i);
		printf("vertex_num: %d\n",g[i]->vertex_num);
		printf("vertex_outer_num: %d\n", g[i]->vertex_outer_num);
		//printf("edge_num: %d\n",g[i]->edge_num);
		//printf("edge_outer_num: %d\n",g[i]->edge_outer_num);
		printf("vertex_id:\n");
		checkvalue(g[i]->vertex_id,g[i]->vertex_num);
		printf("vertex_outer_id:\n");
		checkvalue(g[i]->vertex_outer_id,g[i]->vertex_outer_num);           
	}
}
