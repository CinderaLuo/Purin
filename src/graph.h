#ifndef GRAPH_H_INCLUDED
#define GRAPH_H_INCLUDED

/*
Name: GRAPH_H_DEFINE
Copyright: 2016
Author: Xuan Luo
Date: 13/01/16 11:00
Description: This file defines the data structures for Graph
Note: IDs for vertice start from 1
*/

/* The graph data structure is an EDGE LIST. Each machine have one. */
struct graph_h{
	/*
	*   The brief graph data of each machine 
	*/ 
	int vertex_num;
	int edge_num;
	/* edge_src[e] is the source vertex of Edge e */
	int *edge_src;
	/* edge_dst[e] is the destination vertex of Edge e */
	int *edge_dst;
	/* vertex_id  record the id set of vertex which be processed in this machine */
	int *vertex_id;

	/*
	*   Divide the vertex into two sets.One which has copy is called OUTER, Other called INNER.
	*   Record the information of OUTER.
	*/
	int vertex_outer_num;
	/*  The edge list whose source or destination vertex is OUTER should be processed firstly */
	int edge_outer_num;
	/*  The vertex_inner_id record the id set of OUTER vertex which be processed in this machine */
	int *vertex_outer_id;          
};
typedef struct graph_h Graph;

struct dataSize_h{
	int vertex_num;
	int edge_num;
	/* Max partition size of vertex number which can get from the [input].info file */
	int max_part_vertex_num;
	/* Max partition size of edge number which can get from the [input].info file */
	int max_part_edge_num;
};
typedef struct dataSize_h  DataSize;

/* Allocate the memory for each machine to store the graph data */ 
/* Add: use size to record the graph data informatition */
Graph ** Initiate_graph (int gpu_num,DataSize *size );

/* Read the file from [output-name].vertices which is the partition result of vertice */
/* Add : copy_num[vertex_id-1] is the copy number of vertex_id in all gpus */
void read_graph_vertices( char *  filename, Graph **g,int  gpu_num,int *copy_num);

/* Read the file from [output-name].edges which is the partition result of edge list */
void read_graph_edges(char *  filename, Graph *g,int gpu_num,int *copy_num);


/*check.cpp*/
void checkvalue(int * g, int size);
void checkGraphvalue(Graph ** g, DataSize * size,int gpu_num);

#endif // #ifndef GRAPH_H_INCLUDED
