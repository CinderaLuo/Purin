//#define PRINT_CHECK

//header file of memset()
#include<string.h>
#include<malloc.h>
#include<stdio.h>
#include<omp.h>

#include "graph.h"
#include "timer.h"
//#include "algorithm.h"
#include "cuda_runtime.h"

#define ITERATE_IN_OUTER 1

#ifdef __CUDA_RUNTIME_H__
#define HANDLE_ERROR(err) if (err != cudaSuccess) {	\
	printf("CUDA Error in %s at line %d: %s\n", \
			__FILE__, __LINE__, cudaGetErrorString(cudaGetLastError()));\
	exit(1);\
}
#endif  // #ifdef __CUDA_RUNTIME_H__    


void bfs_cpu(Graph_cpu *g,int *value_cpu,DataSize *dsize,int first_vertex)
{
	printf("BFS is running on CPU...............\n");
	timer_start();
	int vertex_num=dsize->vertex_num;
	int edge_num=dsize->edge_num;
	int edge_src,edge_dst;
	int *queue=(int *)malloc(sizeof(int)*vertex_num);
	memset(value_cpu,0,vertex_num*sizeof(int));
	value_cpu[first_vertex]=1;
	if(queue==NULL)
	{
		perror("Out of memory");
		exit(1);
	}

	int step=1;
	int incount=0;
	int outcount=0;
	queue[incount++]=first_vertex;

	while(incount > outcount)
	{
		int vertex_id=queue[outcount++];
		for (int i = g->vertex_begin[vertex_id]; i < g->vertex_begin[vertex_id+1]; ++i)
		{
			int dst_id=g->vertex_dst[i];
			step=value_cpu[vertex_id];
			if (value_cpu[dst_id]==0)
			{
				value_cpu[dst_id]=step+1;
				queue[incount++]=dst_id;
			}
		}

#ifdef PRINT_CHECK
		printf("\n");
		for (int i = 0; i < 15 && i<vertex_num+1; ++i)
		{
			printf("%d\t", value_cpu[i]);
		}
		printf("\n");
#endif

	}
	double total_time=timer_stop();
	printf("Total time of bfs_cpu is %.3fms\n",total_time);
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
	printf("\n");
	free(m);
}

static __global__ void  bfs_kernel_outer(  
		const int edge_num,
		const int * const edge_src,
		const int * const edge_dest,
		int * const values,
		const int step)
{
	// total thread number & thread index of this thread
	int n = blockDim.x * gridDim.x;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// step counter
	int curStep = step;
	int nextStep = curStep + 1;
	// proceeding loop
	for (int i = index; i < edge_num; i +=n) {
		if (values[edge_src[i]] == curStep && values[edge_dest[i]] == 0) {
			values[edge_dest[i]] = nextStep;
		}
	}
}
static __global__ void bfs_kernel_inner(  
		const int edge_num,
		const int * const edge_src,
		const int * const edge_dest,
		int * const values,
		const int step,
		int * const continue_flag)
{

	// total thread number & thread index of this thread
	int n = blockDim.x * gridDim.x;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// continue flag for each thread
	int flag = 0;

	// proceeding loop
	for (int i = index; i < edge_num; i +=n) {		
		if(values[edge_src[i]]==step && values[edge_dest[i]]==0)	
		{
			values[edge_dest[i]]=step+1;
			flag = 1;	
		}	
	}
	// update flag
	if (flag == 1) *continue_flag = 1;
}

int no_zero_num(int ** a, int gpu_num, int vertex_id, int step)
{
	int count=0;
	for (int i = 0; i < gpu_num; ++i)
	{
		if (a[i][vertex_id]==step+1)
		{
			count++;
		}
	}
	return count;
}

static omp_lock_t lock;

void bfs_gather_cpu(
		Graph **g,
		DataSize *dsize,
		int * copy_num,
		int **h_value,
		int *value_gpu,
		int gpu_num,
		int step,
		int &flag )
{
	int vertex_num=dsize->vertex_num;
	int nozeronum;
	flag=0;
	//omp_init_lock(&lock);
	//#pragma omp parallel for
	for (int i = 0; i < vertex_num+1; ++i)
	{
		// #pragma omp parallel for 
		for (int j = 0; j < gpu_num; ++j)
		{
			if (h_value[j][i]==step+1)
			{
				//omp_set_lock(&lock);
				value_gpu[i]=step+1;
				flag=1;
				//omp_unset_lock(&lock);
			}
		}
	}
	
#pragma omp parallel for
	for (int i = 0; i < gpu_num; ++i)
	{
#pragma omp parallel for
		for (int j = 0; j < vertex_num+1; ++j)
		{
			h_value[i][j]=value_gpu[j];
		}
	}
}

/* BFS algorithm on GPU */
void bfs_gpu(Graph **g,int gpu_num,int *value_gpu,DataSize *dsize, int first_vertex, int *copy_num, int **position_id)
{
	printf("BFS is running on GPU...............\n");
	printf("Start malloc edgelist...\n");
	/* TODO : edgelsit store twices */
	/* Inite value*/
	value_gpu[first_vertex]=1;
	int **h_value=(int **)malloc(sizeof(int *)* gpu_num);
	int **h_flag=(int **)malloc(sizeof(int *)*gpu_num);
	int vertex_num=dsize->vertex_num;
	int **d_edge_inner_src=(int **)malloc(sizeof(int *)*gpu_num);
	int **d_edge_inner_dst=(int **)malloc(sizeof(int *)*gpu_num);
	int **d_edge_outer_src=(int **)malloc(sizeof(int *)*gpu_num);
	int **d_edge_outer_dst=(int **)malloc(sizeof(int *)*gpu_num);
	int **d_value=(int **)malloc(sizeof(int *)*gpu_num);
	int **d_flag=(int **)malloc(sizeof(int *)*gpu_num);

	/* determine the size of outer vertex in one process*/
	int tmp_per_size = min_num_outer_edge(g,gpu_num);
	int outer_per_size=tmp_per_size/ITERATE_IN_OUTER;
	int iterate_in_outer=ITERATE_IN_OUTER+1;
	int *last_outer_per_size=(int *)malloc(sizeof(int)*gpu_num);
	memset(last_outer_per_size,0,sizeof(int)*gpu_num);
	//int last_outer_per_size=0;

	int flag=0;
	int step=1;
	int inner_edge_num=0;

	/* Malloc stream*/
	cudaStream_t **stream;
	cudaEvent_t tmp_start,tmp_stop;
	stream=(cudaStream_t **)malloc(gpu_num*sizeof(cudaStream_t*));

	cudaEvent_t * start_outer,*stop_outer,*start_inner,*stop_inner,*start_asyn,*stop_asyn,*start,*stop;
	start_outer=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));
	stop_outer=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));
	start_inner=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));
	stop_inner=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));
	start_asyn=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));
	stop_asyn=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));
	start=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));
	stop=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));

	for (int i = 0; i < gpu_num; ++i)
	{
		cudaSetDevice(i);
		stream[i]=(cudaStream_t *)malloc((iterate_in_outer+1)*sizeof(cudaStream_t));
		HANDLE_ERROR(cudaEventCreate(&start_outer[i],0));
		HANDLE_ERROR(cudaEventCreate(&stop_outer[i],0));
		HANDLE_ERROR(cudaEventCreate(&start_inner[i],0));
		HANDLE_ERROR(cudaEventCreate(&stop_inner[i],0));  
		HANDLE_ERROR(cudaEventCreate(&start_asyn[i],0));
		HANDLE_ERROR(cudaEventCreate(&stop_asyn[i],0));
		HANDLE_ERROR(cudaEventCreate(&start[i],0));
		HANDLE_ERROR(cudaEventCreate(&stop[i],0));
		HANDLE_ERROR(cudaEventCreate(&tmp_start,0));
		HANDLE_ERROR(cudaEventCreate(&tmp_stop,0));

		for (int j = 0; j <= iterate_in_outer; ++j)
		{
			HANDLE_ERROR(cudaStreamCreate(&stream[i][j]));
		}

#ifdef PRINT_CHECK
		HANDLE_ERROR(cudaMemcpy(h_value[i],d_value[i],sizeof(int)*(vertex_num+1),cudaMemcpyDeviceToHost));
		printf("Before while --> check value\n");
		printf("value:\n");
		for (int j = 0; j < vertex_num+1; ++j)
		{
			printf("%d\t", h_value[i][j]);
		}
		printf("\nOuter_edgelist\n");
		for (int j = 0; j < g[i]->edge_outer_num ; ++j)
		{
			printf("( %d, %d )\t",g[i]->edge_outer_src[j],g[i]->edge_outer_dst[j]);
		}
		printf("\nInner_edgelist\n");
		for (int j = 0; j < g[i]->edge_num- g[i]->edge_outer_num; ++j)
		{
			printf("( %d, %d )\t",g[i]->edge_inner_src[j],g[i]->edge_inner_dst[j]);
		}
		printf("\n");
#endif
	}


	for (int i = 0; i < gpu_num; ++i)
	{
		h_value[i]=(int *)malloc(sizeof(int)*(vertex_num+1));
		memset(h_value[i],0,sizeof(int)*(vertex_num+1));
		h_value[i][first_vertex]=1;
		h_flag[i]=(int *)malloc(sizeof(int));
	}

	for (int i = 0; i < gpu_num; ++i)
	{
		cudaSetDevice(i);
		int out_size=g[i]->edge_outer_num;
		int inner_size=g[i]->edge_num - out_size;

		HANDLE_ERROR(cudaMalloc((void **)&d_edge_outer_src[i],sizeof(int)*out_size));
		HANDLE_ERROR(cudaMalloc((void **)&d_edge_outer_dst[i],sizeof(int)*out_size));
		if (outer_per_size!=0 && outer_per_size < out_size)
		{
			for (int j = 1; j < iterate_in_outer; ++j)
			{
				HANDLE_ERROR(cudaMemcpyAsync((void *)(d_edge_outer_src[i]+(j-1)*outer_per_size),(void *)(g[i]->edge_outer_src+(j-1)*outer_per_size),sizeof(int)*outer_per_size,cudaMemcpyHostToDevice, stream[i][j-1]));
				HANDLE_ERROR(cudaMemcpyAsync((void *)(d_edge_outer_dst[i]+(j-1)*outer_per_size),(void *)(g[i]->edge_outer_dst+(j-1)*outer_per_size),sizeof(int)*outer_per_size,cudaMemcpyHostToDevice, stream[i][j-1]));
			}
		}

		last_outer_per_size[i]=g[i]->edge_outer_num-outer_per_size * (iterate_in_outer-1);           
		if (last_outer_per_size[i]>0 && iterate_in_outer>1 )
		{
			HANDLE_ERROR(cudaMemcpyAsync((void *)(d_edge_outer_src[i]+(iterate_in_outer-1)*outer_per_size),(void *)(g[i]->edge_outer_src+(iterate_in_outer-1)*outer_per_size),sizeof(int)*last_outer_per_size[i],cudaMemcpyHostToDevice, stream[i][iterate_in_outer-1]));
			HANDLE_ERROR(cudaMemcpyAsync((void *)(d_edge_outer_dst[i]+(iterate_in_outer-1)*outer_per_size),(void *)(g[i]->edge_outer_dst+(iterate_in_outer-1)*outer_per_size),sizeof(int)*last_outer_per_size[i],cudaMemcpyHostToDevice, stream[i][iterate_in_outer-1]));
		}

		HANDLE_ERROR(cudaMalloc((void **)&d_edge_inner_src[i],sizeof(int)*inner_size));
		HANDLE_ERROR(cudaMalloc((void **)&d_edge_inner_dst[i],sizeof(int)*inner_size));
		HANDLE_ERROR(cudaMemcpyAsync((void *)d_edge_inner_src[i],(void *)g[i]->edge_inner_src,sizeof(int)*inner_size,cudaMemcpyHostToDevice,stream[i][iterate_in_outer]));
		HANDLE_ERROR(cudaMemcpyAsync((void *)d_edge_inner_dst[i],(void *)g[i]->edge_inner_dst,sizeof(int)*inner_size,cudaMemcpyHostToDevice,stream[i][iterate_in_outer]));

		HANDLE_ERROR(cudaMalloc((void **)&d_value[i],sizeof(int)*(vertex_num+1)));
		HANDLE_ERROR(cudaMemcpyAsync((void *)d_value[i],(void *)h_value[i],sizeof(int)*(vertex_num+1),cudaMemcpyHostToDevice, stream[i][0]));
		HANDLE_ERROR(cudaMalloc((void **)&d_flag[i],sizeof(int)));
	}

	printf("Malloc is finished!\n");
	/* Time Inite*/
	float *outer_compute_time,*inner_compute_time,*compute_time,*total_compute_time,*communicate_time;
	float gather_time=0.0;
	float cpu_gather_time=0.0;
	float total_time=0.0;
	float record_time=0.0;
	outer_compute_time=(float *)malloc(sizeof(float)*gpu_num);
	inner_compute_time=(float *)malloc(sizeof(float)*gpu_num);
	compute_time=(float *)malloc(sizeof(float)*gpu_num);
	total_compute_time=(float *)malloc(sizeof(float)*gpu_num);
	communicate_time=(float *)malloc(sizeof(float)*gpu_num);

	memset(outer_compute_time,0,sizeof(float)*gpu_num);
	memset(inner_compute_time,0,sizeof(float)*gpu_num);
	memset(compute_time,0,sizeof(float)*gpu_num);

	/* one iteration */
	do
	{
		flag=0;
		//HANDLE_ERROR(cudaMemset(d_flag,0,sizeof(int)*gpu_num));
		for (int i = 0; i <gpu_num; ++i)
		{		
			memset(h_flag[i],0,sizeof(int));
			cudaSetDevice(i);
			HANDLE_ERROR(cudaMemcpyAsync(d_flag[i],h_flag[i],sizeof(int),cudaMemcpyHostToDevice,stream[i][0]));
			// outer
			//cudaMemcpy(d_value[i],h_value[i],sizeof(int)*(g[i]->vertex_num+1),cudaMemcpyHostToDevice);
			HANDLE_ERROR(cudaEventRecord(start[i],0));
			HANDLE_ERROR(cudaEventRecord(start_outer[i],stream[i][0] )); 
			if (outer_per_size!=0 && outer_per_size < g[i]->edge_outer_num)
			{
				for (int j = 1; j < iterate_in_outer; ++j)
				{
					//printf("%d outer_per_size %d outer_num %d\n",j,outer_per_size,g[i]->edge_outer_num);
					bfs_kernel_outer<<<208,128,0,stream[i][j-1]>>>(
							outer_per_size,
							d_edge_outer_src[i]+(j-1)*outer_per_size,
							d_edge_outer_dst[i]+(j-1)*outer_per_size,
							d_value[i],
							step);
				}
			}
			last_outer_per_size[i]=g[i]->edge_outer_num-outer_per_size * (iterate_in_outer-1);           
			if (last_outer_per_size[i]>0 && iterate_in_outer>1  )
			{
				// The size of edge list in last block is different in every gpu
				bfs_kernel_outer<<<208,128,0,stream[i][iterate_in_outer-1]>>>(
						last_outer_per_size[i],
						d_edge_outer_src[i]+(iterate_in_outer-1)*outer_per_size,
						d_edge_outer_dst[i]+(iterate_in_outer-1)*outer_per_size,
						d_value[i],
						step);
				//printf("iterate_in_outer  %d, last_outer_per_size %d, g[i]->edge_src %d, outer_per_size  %d\n", iterate_in_outer, last_outer_per_size[i],g[i]->edge_num, outer_per_size);
			}
			HANDLE_ERROR(cudaEventRecord(stop_outer[i], stream[i][0]));


			cudaEventRecord(start_inner[i], stream[i][iterate_in_outer]);
			//inner+flag
			inner_edge_num=g[i]->edge_num-g[i]->edge_outer_num;
			bfs_kernel_inner<<<208,128,0,stream[i][iterate_in_outer]>>>(
					inner_edge_num,
					d_edge_inner_src[i],
					d_edge_inner_dst[i],
					d_value[i],
					step,
					d_flag[i]);
			cudaEventRecord(stop_inner[i], stream[i][iterate_in_outer]);


			cudaEventRecord(start_asyn[i],stream[i][0]);
			HANDLE_ERROR(cudaMemcpyAsync(h_value[i],d_value[i],sizeof(int)*(vertex_num+1),cudaMemcpyDeviceToHost,stream[i][0]));
			HANDLE_ERROR(cudaMemcpyAsync(h_flag[i], d_flag[i],sizeof(int),cudaMemcpyDeviceToHost,stream[i][0]));
			cudaEventRecord(stop_asyn[i],stream[i][0]);


			HANDLE_ERROR(cudaEventRecord(stop[i],0));

		}


		timer_start();
		bfs_gather_cpu(g,dsize,copy_num,h_value,value_gpu,gpu_num,step,flag);
		record_time=timer_stop();
		cpu_gather_time+=record_time;
		gather_time+=record_time;

		cudaEventRecord(tmp_start, 0);
		for (int i = 0; i < gpu_num; ++i)
		{
			HANDLE_ERROR(cudaMemcpyAsync(d_value[i], h_value[i], sizeof(int)*(vertex_num+1),cudaMemcpyHostToDevice,stream[i][0]));
		}

#ifdef PRINT_CHECK
		printf("After bfs_gather_cpu\n%d\n\n",flag);
#endif
		cudaEventRecord(tmp_stop, 0);

		cudaDeviceSynchronize();

		//cudaEventSynchronize(tmp_stop);
		cudaEventElapsedTime(&record_time, tmp_start, tmp_stop);
		gather_time+=record_time;

		for (int i = 0; i < gpu_num; ++i)
		{
			flag=flag||h_flag[i][0];
		}
		step++;

		//collect time  different stream
		for (int i = 0; i < gpu_num; ++i)
		{
			//HANDLE_ERROR(cudaEventSynchronize(stop_outer[i]));
			HANDLE_ERROR(cudaEventElapsedTime(&record_time, start_outer[i], stop_outer[i]));
			outer_compute_time[i]+=record_time;
			HANDLE_ERROR(cudaEventElapsedTime(&record_time, start_inner[i], stop_inner[i]));  
			inner_compute_time[i]+=record_time;
            HANDLE_ERROR(cudaEventElapsedTime(&record_time, start_asyn[i], stop_asyn[i]));  
			communicate_time[i]+=record_time;
			HANDLE_ERROR(cudaEventElapsedTime(&record_time, start[i], stop[i]));
			total_compute_time[i]+=record_time;
		}


	}while(flag);

	//collect the information of time 
	float total_time_n=0.0;
	for (int i = 0; i < gpu_num; ++i)
	{
		compute_time[i]=outer_compute_time[i]+inner_compute_time[i];
		if (total_time_n<compute_time[i])
		{
			//max
			total_time_n=compute_time[i];
		}
	}
	total_time=total_time_n+gather_time;
	printf("Total time of bfs_cpu is %.3fms\n",total_time);
	printf("Detail:\n");
	printf("\n");
	for (int i = 0; i < gpu_num; ++i)
	{
		printf("GPU %d\n",i);
		printf("Outer_Compute_Time:  %.3fms\n", outer_compute_time[i]);
		printf("Inner_Compute_Time:  %.3fms\n", inner_compute_time[i]);
		printf("Compute_Time:        %.3fms\n", compute_time[i]);
		printf("Communicate_Time     %.3fms\n", communicate_time[i]);
		printf("Total Compute_Time   %.3fms\n", total_compute_time[i]);
	}
	printf("\n");
	printf("Gather_Time:          %.3fms\n", gather_time);
	printf("CPU_Gather_Time:      %.3fms\n", cpu_gather_time);

	//clean

	for (int i = 0; i < gpu_num; ++i)
	{
		cudaSetDevice(i);
		HANDLE_ERROR(cudaEventDestroy(start[i]));
		HANDLE_ERROR(cudaEventDestroy(stop[i]));
		HANDLE_ERROR(cudaFree(d_edge_outer_src[i]));
		HANDLE_ERROR(cudaFree(d_edge_outer_dst[i]));
		HANDLE_ERROR(cudaFree(d_edge_inner_src[i]));
		HANDLE_ERROR(cudaFree(d_edge_inner_dst[i]));
		HANDLE_ERROR(cudaFree(d_value[i]));
		HANDLE_ERROR(cudaFree(d_flag[i]));

		HANDLE_ERROR(cudaDeviceReset());
		free(h_value[i]);
		free(h_flag[i]);
		free(stream[i]);
	}

	free(outer_compute_time);
	free(inner_compute_time);
	free(compute_time);

}

