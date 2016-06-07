/*
 * Author: Xuan Luo
 * Data  : 2016-01-13
 * Function : change edgelist without weigt to the edgelist with weight
 * -----------------------------------
 *                         | tab |
 * eg:  1 2     weight=0   2   3
 *      1 3   ---------->  2   4
 *      1 4                2   5 
 * --------------------------------
 *                         | tab |
 * eg:  1 2     weight=n   2   3  n
 *      1 3   ---------->  2   4  n
 *      1 4                2   5  n             
 *-------------------------------------
 *
 */
#include "stdlib.h"
#include "stdio.h"
#include "string"

using namespace std;

int main(int argc, char *argv[])
{
	std::string file = argv[1], out = file + "_vgp";
	FILE *fin = fopen(file.c_str(), "r");
	FILE *fout = fopen(out.c_str(), "w");

	if(argc != 5)
	{
		printf("no use input...\n");
		printf("Usage: to_gt_file file_name number_of_nodes number_of_edges with_weight\n");
		return 0;
	}

	int with_weight = atoi(argv[4]);
	unsigned int nodes = atoi(argv[2]), edges = atoi(argv[3]);
	unsigned int src, dst, total = 0, weight;

	if(with_weight == 0)
	{
		while(fscanf(fin, "%u\t%u\n", &src, &dst))
		{
			fprintf(fout, "%u\t%u\n", src+1, dst+1);
			total++;
			if(total == edges)	break;
		}
	}
	else
	{
		while(fscanf(fin, "%u\t%u\t%u\n", &src, &dst, &weight))
		{
			fprintf(fout, "%u\t%u %u\n", src+1, dst+1, weight);
			total++;
			if(total == edges)  break;
		}
	}
	return 0;
}
