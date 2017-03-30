#ifndef A_H_INCLUDED
#define A_H_INCLUDED

#include "graph.h"
/*
Name: ALGORITHM_H_DEFINE
Copyright: 2016
Author: Xuan Luo
Date: 29/01/16 11:30
Description: This file defines the algorithm
*/

void bfs_cpu(Graph_cpu *g,int *value_cpu,DataSize *dsize,int first_vertex);

// print info about bfs values
void print_bfs_values(const int * const values, int const size);

void bfs_gpu(Graph **g,int gpu_num,int *value_gpu,DataSize *dsize, int first_vertex, int *copy_num, int **position_id);

void pr_gpu(Graph **g,int gpu_num,float *value_gpu,DataSize *dsize, int* out_degree, int *copy_num, int **position_id);

#endif
