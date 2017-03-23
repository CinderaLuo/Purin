#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

struct graph_t
{
    /* number of vertices & edges */
    int vertex_num;
    int edge_num;
    /* Vertex v has outgoing edge from
        index vertex_begin[v] (inclusive) to
        index vertex_begin[v + 1] (exclusive) */
    int * vertex_begin;
    /* edge_dest[e] is the destination vertex of Edge e */
    int * edge_dest;
    /* edge_src[e] is the source vertex of Edge e */
    int * edge_src;
    /* vertex_begin[v] == vertex_begin[v + 1]) */
    int * vertex_end;
};
typedef struct graph_t Graph;

Graph * allocate_graph(const int vertex_num, const int edge_num) {
    Graph * g = (Graph *) calloc(1, sizeof(Graph));
    int * vertex_begin = (int *) malloc(sizeof(int) * (vertex_num + 1));
    int * edge_src = (int *) malloc(sizeof(int) * edge_num);
    int * edge_dest = (int *) malloc(sizeof(int) * edge_num);
    if (g == NULL || vertex_begin == NULL || edge_src == NULL || edge_dest == NULL) {
        perror("Out of memory for graph");
        exit(1);
    } else {
        g->vertex_num = vertex_num;
        g->edge_num = edge_num;
        g->vertex_begin = vertex_begin;
        g->vertex_end = vertex_begin + 1;
        g->edge_src = edge_src;
        g->edge_dest = edge_dest;
    }
    return g;
}
Graph * read_random_edge_list(const char * const filename)
{
    /* try to open the file */
    FILE * f = fopen(filename, "r");
    if (f == NULL) {
        fprintf(stderr, "File open failed: %s ", filename);
        perror("");
        /* exit program when fail */
        exit(1);
    }

    /* naive method: scan the entire file to get edge_num & vertex_num (max vertex id) */
    int c;
    int vertex_num = -1;
    int edge_num = 0;
    while ((c = fgetc(f)) != EOF) {
        if (isdigit(c)) {
            /* the line starts with digit should be an edge (of two numbers) */
            ungetc(c, f);
            fscanf(f, "%d", &c);
            if (c > vertex_num) vertex_num = c;
            fscanf(f, "%d", &c);
            if (c > vertex_num) vertex_num = c;
            edge_num++;
        }
        /* the next line (also skip the line starts with '#') */
        while ((c = fgetc(f)) != EOF && c != '\n');
    }
    if (vertex_num < 0) {
        fclose(f);
        fprintf(stderr, "Unknown file format: '%s'\n", filename);
        exit(1);
    }
    /* vertex number = max vertex id + 1 */
    vertex_num++;
	//luo  totem error when the vertex_num==maax_vertex_ID
	if(c==vertex_num)
		vertex_num++;
    /* allocate memory for the graph */
    Graph * ret = allocate_graph(vertex_num, edge_num);
    int * vertex_begin = ret->vertex_begin;
    int * edge_src = ret->edge_src;
    int * edge_dest = ret->edge_dest;
    //add 
	int *tmp_vertex=(int *)malloc(sizeof(int)*(vertex_num+1));
	memset(vertex_begin, 0, sizeof(int) * (vertex_num + 1));
	memset(tmp_vertex, 0, sizeof(int) * (vertex_num + 1));
    /* naive method: scan the entire file to count edge for each vertex */
    fseek(f, 0, SEEK_SET);
    while ((c = fgetc(f)) != EOF) {
        if (isdigit(c)) {
            /* read the source vertex, then increase the counter */
            ungetc(c, f);
            fscanf(f, "%d", &c);
            tmp_vertex[c]++;
        }
        /* the next line (also skip the line starts with '#') */
        while ((c = fgetc(f)) != EOF && c != '\n');
	}

	/* calculate the edge index for each vertex */
    for (int i = 1; i < vertex_num; i++) {
        //error
		/*
		vertex_begin[i + 1] += vertex_begin[i];
        vertex_begin[i] = vertex_begin[i - 1];
		*/
		vertex_begin[i]=vertex_begin[i-1]+tmp_vertex[i-1];
    }
    vertex_begin[0] = 0;
    vertex_begin[vertex_num] = edge_num;

    /* scan the file for data */
    fseek(f, 0, SEEK_SET);
    while ((c = fgetc(f)) != EOF) {
        if (isdigit(c)) {
            /* read the edge */
            ungetc(c, f);
            int src, dest;
            fscanf(f, "%d %d", &src, &dest);
            int k = vertex_begin[src];
            edge_src[k] = src;
            edge_dest[k] = dest;
            vertex_begin[src]++;
        }
        /* next line */
        while ((c = fgetc(f)) != EOF && c != '\n');
    }
    fclose(f);
    /* reset edge indices */
    for (int i = vertex_num - 1; i > 0; i--)
        vertex_begin[i] = vertex_begin[i - 1];
    vertex_begin[0] = 0;

    return ret;
}

void write_edge_list(const Graph * const g, const char * const filename)
{
    FILE * f = fopen(filename, "wb");
    if (f != NULL) {
       for (int i = 0; i < g->edge_num; ++i)
       {
          fprintf(f, "%d	%d\n",g->edge_src[i]+1,g->edge_dest[i]+1);
       }
    }
    fclose(f);
}

int main(int argc, char ** argv) {
	Graph *g;
	if (argc > 2) {
		printf("reading file '%s' ... \n", argv[1]);
		g = read_random_edge_list(argv[1]);
		if (g != NULL) {
			printf("saving file '%s' ... \n", argv[2]);
			write_edge_list(g, argv[2]);
		}
	} 
	return 0;
}
