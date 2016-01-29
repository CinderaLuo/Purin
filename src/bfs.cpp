//header file of memset()
#include<string.h>
#include<malloc.h>
#include<stdio.h>

#include "graph.h"
#include "timer.h"
//#include "cuda_runtime.h"

void bfs_cpu(Graph_cpu *g,int *value_cpu,DataSize *dsize,int first_vertex)
{
	timer_start();
	int vertex_num=dsize->vertex_num;
	int edge_num=dsize->edge_num;
	int edge_src,edge_dst;
	memset(value_cpu,0,vertex_num*sizeof(int));
	value_cpu[first_vertex]=1;

	int step=1;
	int flag=0;

	while(!flag)
	{
		flag=0;
		//#pragma omp parallel for
		for(int i=0;i<edge_num;i++)
		{
			edge_src=g->edge_src[i];
			edge_dst=g->edge_dst[i];
			if(value_cpu[edge_src-1]==step && value_cpu[edge_dst-1]==0)
			{
				value_cpu[edge_dst-1]=step+1;
				flag=1;
			}
		}
		step++;
	}
}

// print info about bfs values
void print_bfs_values(const int * const values, int const size) {
	int visited = 0;
	int step = 0;
	int first = 0;
	// get the max step and count the visited
	for (int i = 0; i < size; i++) {
		if (values[i] != 0) {
			visited++;
			if (values[i] > step) step = values[i];
			if (values[i] == 1) first = i;
		}
	}
	// count vertices of each step
	if (step == 0) return;
	int * m = (int *) malloc((step + 1)*sizeof(int));
	memset(m,0,sizeof(int)*(step+1));
	for (int i = 0; i < size; i++) {
		m[values[i]]++;
	}
	// print result info
	printf("\tSource = %d, Step = %d, Visited = %d\n", first, step, visited);
	printf("\tstep\tvisit\n");
	for (int i = 1; i <= step; i++) {
		printf("\t%d\t%d\n", i, m[i]);
	}
	/*
	//Add to check
	printf("\tindex\tvalues\n");
	for(int i=0;i<size;i++)
	{
	printf("\t%d\t%d\n",i,values[i]);
	}
	*/
	free(m);
}
