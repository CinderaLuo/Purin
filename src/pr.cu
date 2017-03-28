#include<string.h>
#include<malloc.h>
#include<stdio.h>
#include<omp.h>

#include "graph.h"
#include "timer.h"
//#include "algorithm.h"
#include "cuda_runtime.h"

// The number of partitioning the outer chunk must be greater or equal to 1
#define ITERATE_IN_OUTER 1
#define NUM_THREADS 4

#ifdef __CUDA_RUNTIME_H__
#define HANDLE_ERROR(err) if (err != cudaSuccess) {	\
	printf("CUDA Error in %s at line %d: %s\n", \
			__FILE__, __LINE__, cudaGetErrorString(cudaGetLastError()));\
	exit(1);\
}
#endif  // #ifdef __CUDA_RUNTIME_H__    


static __global__ void  pr_kernel_outer(  
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

static __global__ void pr_kernel_inner(  
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
	int curStep = step;
	int nextStep = curStep + 1;

	for (int i = index; i < edge_num; i +=n) {		
		if(values[edge_src[i]]==curStep && values[edge_dest[i]]==0)	
		{
			values[edge_dest[i]]=nextStep;
			flag = 1;
		}	
	}
	// update flag
	if (flag == 1) *continue_flag = 1;
}

/* PageRank algorithm on GPU */
void pr_gpu(Graph **g,int gpu_num,int *value_gpu,DataSize *dsize, int first_vertex, int *copy_num, int **position_id)
{
   	printf("PageRank is running on GPU...............\n");
	printf("Start malloc edgelist...\n");
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

	for (int i = 0; i < gpu_num; ++i)
	{
		h_value[i]=(int *)malloc(sizeof(int)*(vertex_num+1));
		memset(h_value[i],0,sizeof(int)*(vertex_num+1));
		h_value[i][first_vertex]=1;
		h_flag[i]=(int *)malloc(sizeof(int));
	}
	   /*Cuda Malloc*/
	/* Malloc stream*/
	cudaStream_t **stream;
	cudaEvent_t tmp_start,tmp_stop;
	stream=(cudaStream_t **)malloc(gpu_num*sizeof(cudaStream_t*));

	cudaEvent_t * start_outer,*stop_outer,*start_inner,*stop_inner,*start_asyn,*stop_asyn;
	start_outer=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));
	stop_outer=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));
	start_inner=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));
	stop_inner=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));
	start_asyn=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));
	stop_asyn=(cudaEvent_t *)malloc(gpu_num*sizeof(cudaEvent_t));

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


		for (int j = 0; j <= iterate_in_outer; ++j)
		{
			HANDLE_ERROR(cudaStreamCreate(&stream[i][j]));
		}
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
		HANDLE_ERROR(cudaMemcpyAsync((void *)d_value[i],(void *)h_value[i],sizeof(int)*(vertex_num+1),cudaMemcpyHostToDevice,stream[i][0]));

		HANDLE_ERROR(cudaMalloc((void **)&d_flag[i],sizeof(int)));


	}
	printf("Malloc is finished!\n");

	/* Before While: Time Initialization */
	float *outer_compute_time,*inner_compute_time,*compute_time,*total_compute_time,*extract_bitmap_time;
	float gather_time=0.0;
	float cpu_gather_time=0.0;
	float total_time=0.0;
	float record_time=0.0;
	outer_compute_time=(float *)malloc(sizeof(float)*gpu_num);
	inner_compute_time=(float *)malloc(sizeof(float)*gpu_num);
	compute_time=(float *)malloc(sizeof(float)*gpu_num);
	total_compute_time=(float *)malloc(sizeof(float)*gpu_num);
	extract_bitmap_time=(float *)malloc(sizeof(float)*gpu_num);

	memset(outer_compute_time,0,sizeof(float)*gpu_num);
	memset(inner_compute_time,0,sizeof(float)*gpu_num);
	memset(compute_time,0,sizeof(float)*gpu_num);

	
    /* Before While: Variable Initialization */
	int flag=0;
	int step=1;
	int inner_edge_num=0;

	printf("Computing......\n");
	do
	{
		flag=0;
		for (int i = 0; i <gpu_num; ++i)
		{		
			memset(h_flag[i],0,sizeof(int));
			cudaSetDevice(i);
			HANDLE_ERROR(cudaMemcpyAsync(d_flag[i],h_flag[i],sizeof(int),cudaMemcpyHostToDevice,stream[i][0]));
	
            HANDLE_ERROR(cudaEventRecord(start_outer[i], stream[i][0]));
			//kernel of outer edgelist
			if (outer_per_size!=0 && outer_per_size < g[i]->edge_outer_num)
			{
				for (int j = 1; j < iterate_in_outer; ++j)
				{				
					pr_kernel_outer<<<208,128,0,stream[i][j-1]>>>(
							outer_per_size,
							d_edge_outer_src[i]+(j-1)*outer_per_size,
							d_edge_outer_dst[i]+(j-1)*outer_per_size,
							d_value[i],
							step);
					HANDLE_ERROR(cudaMemcpyAsync((void *)(h_bitmap[i]+(j-1)*bitmap_len),(void *)(d_bitmap[i]+(j-1)*bitmap_len),sizeof(int)*(bitmap_len),cudaMemcpyDeviceToHost,stream[i][j-1]));
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
				kernel_make_bitmap<<<208,128,0,stream[i][iterate_in_outer-1]>>>(
						vertex_num,
						d_value[i],
						d_bitmap[i]+(iterate_in_outer-1)*bitmap_len,
						step+1
						);
				HANDLE_ERROR(cudaMemcpyAsync((void *)(h_bitmap[i]+(iterate_in_outer-1)*bitmap_len),(void *)(d_bitmap[i]+(iterate_in_outer-1)*bitmap_len),sizeof(int)*(bitmap_len),cudaMemcpyDeviceToHost,stream[i][iterate_in_outer-1]));
			}
			HANDLE_ERROR(cudaEventRecord(stop_outer[i], stream[i][iterate_in_outer-1]));


			HANDLE_ERROR(cudaEventRecord(start_inner[i], stream[i][iterate_in_outer]));
			//inner+flag
			inner_edge_num=g[i]->edge_num-g[i]->edge_outer_num;
			if (inner_edge_num>0)
			{
				bfs_kernel_inner<<<208,128,0,stream[i][iterate_in_outer]>>>(
						inner_edge_num,
						d_edge_inner_src[i],
						d_edge_inner_dst[i],
						d_value[i],
						step,
						d_flag[i]);			
				HANDLE_ERROR(cudaMemcpyAsync(h_flag[i], d_flag[i],sizeof(int),cudaMemcpyDeviceToHost,stream[i][iterate_in_outer]));	    
			}
			HANDLE_ERROR(cudaEventRecord(stop_inner[i],stream[i][iterate_in_outer]));
		}


		//merge bitmap on gpu
		double t1=omp_get_wtime();
		merge_bitmap_on_cpu(bitmap_len, gpu_num, h_bitmap, buff_bitmap);
		double t2=omp_get_wtime();
		record_time=(t2-t1)*1000;
		gather_time+=record_time;


		for (int i = 0; i < gpu_num; ++i)
		{
			cudaSetDevice(i);
			//extract bitmap to the value
			HANDLE_ERROR(cudaMemcpyAsync(d_bitmap[i], buff_bitmap,sizeof(int)*bitmap_len,cudaMemcpyHostToDevice,stream[i][0]));
			HANDLE_ERROR(cudaEventRecord(start_asyn[i], stream[i][0]));
			kernel_extract_bitmap<<<256,108,0,stream[i][0]>>>
				(  
				 vertex_num,
				 d_bitmap[i],
				 d_value[i],
				 step+1
				);		
			HANDLE_ERROR(cudaEventRecord(stop_asyn[i], stream[i][0]));
			HANDLE_ERROR(cudaMemset(d_bitmap[i],0,sizeof(int)*(bitmap_len*iterate_in_outer)));	
		}

		for (int i = 0; i < gpu_num; ++i)
		{
			flag=flag||h_flag[i][0];
		}
		step++;

		//collect time  different stream
		for (int i = 0; i < gpu_num; ++i)
		{
			cudaSetDevice(i);
			HANDLE_ERROR(cudaEventSynchronize(stop_outer[i]));
            HANDLE_ERROR(cudaEventSynchronize(stop_inner[i]));
            HANDLE_ERROR(cudaEventSynchronize(stop_asyn[i]));

			HANDLE_ERROR(cudaEventElapsedTime(&record_time, start_outer[i], stop_outer[i]));
			outer_compute_time[i]+=record_time;
			HANDLE_ERROR(cudaEventElapsedTime(&record_time, start_inner[i], stop_inner[i]));  
			inner_compute_time[i]+=record_time;
			HANDLE_ERROR(cudaEventElapsedTime(&record_time, start_asyn[i], stop_asyn[i]));  
			extract_bitmap_time[i]+=record_time;
            total_compute_time[i]=outer_compute_time[i]+extract_bitmap_time[i]-inner_compute_time[i]>0?(outer_compute_time[i]+extract_bitmap_time[i]):inner_compute_time[i];
		}		
	}while(flag);


}