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
// >> vertex_num 
#define MAX_POSITION  -1
/* The graph data structure is an EDGE LIST. Each machine have one. */
struct graph_h{
	/*
	*   The brief graph data of each machine 
	*/ 
	int vertex_num;
	int edge_num;
	/* vertex_id  record the id set of vertex which be processed in this machine */
	/* TODO: change the type to hash map*/
	int *vertex_id;

	/*
	*   Divide the vertex into two sets.One which has copy is called OUTER, Other called INNER.
	*   Record the information of OUTER.
	*/
	/* edge_outer_src[e] is the source vertex of Edge e, when the destination vertex is outer*/
	int *edge_outer_src;
	/* edge_outer_dst[e] is the destination vertex of Edge e, when the destination vertex is outer*/
	int *edge_outer_dst;
	/* edge_inner_src[e] is the source vertex of Edge e, when the destination vertex is inner*/
	int *edge_inner_src;
	/* edge_inner_dst[e] is the destition vertex of Edge e, when the destination vertex is inner*/
	int *edge_inner_dst;
	int vertex_outer_num;
	/*  The edge list whose source or destination vertex is OUTER should be processed firstly */
	int edge_outer_num;
	/*  The vertex_inner_id record the id set of OUTER vertex which be processed in this machine */
	/* No need */
	int *vertex_outer_id;          
};
typedef struct graph_h Graph;

struct graph_d
{
	int * d_edge_outer_src;
	int * d_edge_outer_dst;
	int * d_edge_inner_src;
	int * d_edge_inner_dst;
	int * d_value;
	int * d_flag;
};

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
void read_graph_vertices( char *  filename, Graph ** g,int  gpu_num,int *copy_num);
void read_graph_vertices_m( char *  filename, Graph ** g,int  gpu_num,int *copy_num, int *map_copy);

/* Read the file from [output-name].edges which is the partition result of edge list */
void read_graph_edges(char *  filename, Graph ** g, int gpu_num,int *copy_num);

/* Read Graph **g to store the max_part_vertex_num and max_part_edge_num which be needed in DataSize *dsize*/
void read_graph_size(Graph **g, DataSize *dsize, int gpu_num);

/* Record the max size of outer edge lists in GPUs which is used for determining the block size */
int max_num_outer_edge(Graph **g, int gpu_num);

/* Record the min size of outer edge lsits in GPUs which is used for determining the block size */
int min_num_outer_edge(Graph **g, int gpu_num);

/* Do not think about preprocesing time */
/* The following functions just are used to [algorithm]_cpu() to check the correctness. */
struct graph_cpu{
	int *edge_src;
	int *edge_dst;
	int *vertex_begin;
	int *vertex_dst;
};
typedef struct graph_cpu Graph_cpu;
/* In fact, read the [input].edges again. TODO: Integrate into read_graph_edges() */
Graph_cpu * read_graph_edges_again(char * filename, int edge_num);
void edgelsit_to_csr( Graph_cpu *g, int vertex_num, int edge_num);


/*check.cpp*/
/* print the value of array g */
void checkvalue_s(int * g, int size);
/* print the value of array g and m*/
void checkvalue_d(int * g, int *m,int size);
/* check graph data structure */
void checkGraphvalue(Graph ** g, DataSize * size,int gpu_num);
void checkResult(int *g, int *m, int num);

/* codingIndex.cpp */
void coding(Graph ** g,int gpu_num);
/* record position of every vertex*/
int ** recordPosition(Graph **g, int gpu_num, int vertex_num);

#endif // #ifndef GRAPH_H_INCLUDED
