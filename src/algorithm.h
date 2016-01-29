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

#endif
