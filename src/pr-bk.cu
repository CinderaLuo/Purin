#include<string.h>
#include<malloc.h>
#include<stdio.h>
#include<omp.h>

#include "graph.h"
#include "timer.h"
//#include "algorithm.h"
#include "cuda_runtime.h"

// The number of partitioning the outer chunk must be greater or equal to 1
#define ITERATE_IN_OUTER 2
#define NUM_THREADS 2

#define PAGERANK_COEFFICIENT  0.85f
#define PAGERANK_THRESHOLD  0.005f



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
		const int * const out_degree,
		const float * const values,
		float * const add_values)
{
	// total thread number & thread index of this thread
	int n = blockDim.x * gridDim.x;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	float sum=0.0f;
	for (int i = index; i < edge_num; i+=n)
	{
		int src=edge_src[i];
		int dest=edge_dest[i];
		if (out_degree[src])
		{
			sum=values[src]/out_degree[src];
		    atomicAdd(&add_values[dest],sum);
		}		
	}
}



static __global__ void pr_kernel_inner(  
		const int edge_num,
		const int * const edge_src,
		const int * const edge_dest,
		const int * const out_degree,
		const float * const values,
		float * const add_values,
		int * const continue_flag)
{

	// total thread number & thread index of this thread
	int n = blockDim.x * gridDim.x;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int flag=0;
	float sum=0.0f;
	for (int i = index; i < edge_num; i+=n)
	{
		int src=edge_src[i];
		int dest=edge_dest[i];
		if (out_degree[src])
		{
			sum=values[src]/out_degree[src];
		    atomicAdd(&add_values[dest],sum);
		}
	}
	__syncthreads();
	//check
	float new_value=0.0f;
	for (int i = index; i < edge_num; i+=n)
	{
		new_value=add_values[edge_dest[i]]*PAGERANK_COEFFICIENT+1.0f - PAGERANK_COEFFICIENT;
		if (fabs(new_value-values[edge_dest[i]])>PAGERANK_THRESHOLD)
		{
			flag=1;
		}
	}
	if (flag==1)  *continue_flag=1;
}


static __global__ void kernel_extract_values(
		int const edge_num,
		int const vertex_num,
		int * const edge_dest,
		float * const add_value,
		float * const value
		)
{
	int n = blockDim.x * gridDim.x;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = index; i < edge_num; i+=n)
	{
		int dest=edge_dest[i];
		if(dest <= vertex_num )
		{
		value[dest]=add_value[dest];
		add_value[dest]=0.0;
		}
	}  
}

void merge_value_on_cpu(
		int const vertex_num, 
		int const gpu_num, 
		float * const  *h_add_value, 
		float * const value_gpu , 
		int *copy_num, 
		int flag)
{
	int i,id;
	float new_value=0.0f;
	omp_set_num_threads(NUM_THREADS);	
#pragma omp parallel private(i)
	{
		id=omp_get_thread_num(); 
		for (i = id; i < vertex_num; i=i+NUM_THREADS)
		{
			if (copy_num[i]>1)
			{
				new_value=0.0f;
				for (int j = 0; j < gpu_num; ++j)
				{
					new_value+=h_add_value[j][i];  
				}
				new_value=PAGERANK_COEFFICIENT*new_value+1.0 - PAGERANK_COEFFICIENT;
				if(fabs(new_value- value_gpu[i]>PAGERANK_THRESHOLD))
					flag=1;
				value_gpu[i]=new_value;
			}		
		}

	}
}

void Gather_result_pr(
		int const vertex_num, 
		int const gpu_num, 
		int * const copy_num,
		float * const  *h_add_value,  
		float * const value_gpu
		)
{
	int i,id;
	float new_value=0.0f;
	omp_set_num_threads(NUM_THREADS);	
#pragma omp parallel private(i)
	{
		id=omp_get_thread_num(); 
		for (i = id; i < vertex_num; i=i+NUM_THREADS)
		{
			if (copy_num[i]>1)
			{
				new_value=0.0f;
				for (int j = 0; j < gpu_num; ++j)
				{
					new_value+=h_add_value[j][i];  
				}
				new_value=PAGERANK_COEFFICIENT*new_value+1.0 - PAGERANK_COEFFICIENT;
				value_gpu[i]=new_value;	
			}
		}
	}
}

/* PageRank algorithm on GPU */
void pr_gpu(Graph **g,int gpu_num,float *value_gpu,DataSize *dsize, int* out_degree, int *copy_num, int **position_id)
{
	printf("PageRank is running on GPU...............\n");
	printf("Start malloc edgelist...\n");

	int **h_flag=(int **)malloc(sizeof(int *)*gpu_num);
	int vertex_num=dsize->vertex_num;
	int **d_edge_inner_src=(int **)malloc(sizeof(int *)*gpu_num);
	int **d_edge_inner_dst=(int **)malloc(sizeof(int *)*gpu_num);
	int **d_edge_outer_src=(int **)malloc(sizeof(int *)*gpu_num);
	int **d_edge_outer_dst=(int **)malloc(sizeof(int *)*gpu_num);
	float **h_value=(float **)malloc(sizeof(float *)* gpu_num);
	float **h_add_value=(float **)malloc(sizeof(float *)*gpu_num);

	float **d_value=(float **)malloc(sizeof(float *)*gpu_num);
	//pr different
	//float **d_tem_value=(float **)malloc(sizeof(float *)*gpu_num);
	float **d_add_value=(float **)malloc(sizeof(float *)*gpu_num);
	int **d_outdegree=(int **)malloc(sizeof(int *)*gpu_num);

	int **d_flag=(int **)malloc(sizeof(int *)*gpu_num);

	/* determine the size of outer vertex in one process*/
	int tmp_per_size = min_num_outer_edge(g,gpu_num);
	int outer_per_size=tmp_per_size/ITERATE_IN_OUTER;
	int iterate_in_outer=ITERATE_IN_OUTER+1;
	int *last_outer_per_size=(int *)malloc(sizeof(int)*gpu_num);
	memset(last_outer_per_size,0,sizeof(int)*gpu_num);

	for (int i = 0; i < gpu_num; ++i)
	{
		h_value[i]=(float *)malloc(sizeof(float)*(vertex_num+1));
		h_add_value[i]=(float *)malloc(sizeof(float)*(vertex_num+1));
		//memset 0.0 or 1.0 
		memset(h_value[i],0.0,sizeof(float)*(vertex_num+1));
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

		HANDLE_ERROR(cudaMalloc((void **)&d_value[i],sizeof(float)*(vertex_num+1)));
		HANDLE_ERROR(cudaMemcpyAsync((void *)d_value[i],(void *)h_value[i],sizeof(float)*(vertex_num+1),cudaMemcpyHostToDevice,stream[i][0]));
		//pr different
		HANDLE_ERROR(cudaMalloc((void **)&d_add_value[i],sizeof(float)*(vertex_num+1)));
		//"memset only works for bytes. If you're using the runtime API, you can use thrust::fill() instead"
		//HANDLE_ERROR(cudaMemset((void **)&d_add_value[i],0,sizeof(float)*(vertex_num+1)));

		//HANDLE_ERROR(cudaMalloc((void **)&d_tem_value[i],sizeof(float)*(vertex_num+1)));
		//HANDLE_ERROR(cudaMalloc((void **)&d_tem_value[i],sizeof(float)*(vertex_num+1)));
		HANDLE_ERROR(cudaMalloc((void **)&d_outdegree[i],sizeof(int)*(vertex_num+1)));
		HANDLE_ERROR(cudaMemcpyAsync(d_outdegree[i],out_degree, sizeof(int)*(vertex_num+1),cudaMemcpyHostToDevice,stream[i][0]));

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
	int step=0;
	int flag=0;
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
							d_outdegree[i],
							d_value[i],
							d_add_value[i]);
					//TODO didn't not realize overlap
					HANDLE_ERROR(cudaMemcpyAsync((void *)(h_add_value[i]),(void *)(d_add_value[i]),sizeof(float)*(vertex_num+1),cudaMemcpyDeviceToHost,stream[i][j-1]));
				}
			}

			last_outer_per_size[i]=g[i]->edge_outer_num-outer_per_size * (iterate_in_outer-1);           
			if (last_outer_per_size[i]>0 && iterate_in_outer>1  )
			{
				pr_kernel_outer<<<208,128,0,stream[i][iterate_in_outer-1]>>>(
						last_outer_per_size[i],
						d_edge_outer_src[i]+(iterate_in_outer-1)*outer_per_size,
						d_edge_outer_dst[i]+(iterate_in_outer-1)*outer_per_size,
						d_outdegree[i],
						d_value[i],
						d_add_value[i]);
				//TODO didn't not realize 
				HANDLE_ERROR(cudaMemcpyAsync((void *)(h_add_value[i]),(void *)(d_add_value[i]),sizeof(float)*(vertex_num+1),cudaMemcpyDeviceToHost,stream[i][iterate_in_outer-1]));
			}
			HANDLE_ERROR(cudaEventRecord(stop_outer[i], stream[i][iterate_in_outer-1]));


			HANDLE_ERROR(cudaEventRecord(start_inner[i], stream[i][iterate_in_outer]));
			//inner+flag
			inner_edge_num=g[i]->edge_num-g[i]->edge_outer_num;
			if (inner_edge_num>0)
			{
				pr_kernel_inner<<<208,128,0,stream[i][iterate_in_outer]>>>(
						inner_edge_num,
						d_edge_inner_src[i],
						d_edge_inner_dst[i],
						d_outdegree[i],
						d_value[i],
						d_add_value[i],
						d_flag[i]);			
				HANDLE_ERROR(cudaMemcpyAsync(h_flag[i], d_flag[i],sizeof(int),cudaMemcpyDeviceToHost,stream[i][iterate_in_outer]));	    
			}
			HANDLE_ERROR(cudaEventRecord(stop_inner[i],stream[i][iterate_in_outer]));
		}


		//merge bitmap on gpu
		double t1=omp_get_wtime();
		merge_value_on_cpu(vertex_num, gpu_num, h_add_value, value_gpu, copy_num, flag);
		double t2=omp_get_wtime();
		record_time=(t2-t1)*1000;
		gather_time+=record_time;


		for (int i = 0; i < gpu_num; ++i)
		{
			cudaSetDevice(i);
			//extract bitmap to the value
			HANDLE_ERROR(cudaMemcpyAsync(d_add_value[i], value_gpu,sizeof(float)*(vertex_num+1),cudaMemcpyHostToDevice,stream[i][0]));
			HANDLE_ERROR(cudaEventRecord(start_asyn[i], stream[i][0]));
			// d_value copy to the value of outer vertices
			kernel_extract_values<<<208,128,0,stream[i][0]>>>
				(  
				 g[i]->edge_outer_num,
				 vertex_num,
				 d_edge_outer_dst[i],
				 d_add_value[i],
				 d_value[i]
				);		
			HANDLE_ERROR(cudaEventRecord(stop_asyn[i], stream[i][0]));
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
	}while(flag && step<10);


	//Todo to get the true value of inner vertice and outer vertice
	for (int i = 0; i < gpu_num; ++i)
	{
		cudaSetDevice(i);
		cudaMemcpyAsync((void *)h_value[i],(void *)d_value[i],sizeof(int)*(vertex_num+1),cudaMemcpyDeviceToHost,stream[i][0]);
	}

	printf("Gather result on cpu....\n");
	Gather_result_pr(vertex_num,gpu_num,copy_num,h_add_value,value_gpu);

	printf("Time print\n");

	//collect the information of time 
	float total_time_n=0.0;
	for (int i = 0; i < gpu_num; ++i)
	{
		if(total_time_n<total_compute_time[i])
			total_time_n=total_compute_time[i];
	}
	total_time=total_time_n>gather_time?total_time_n:gather_time;

	printf("Total time of pr_gpu is %.3f ms\n",total_time);
	printf("Elapsed time of pr_gpu is %.3f ms\n", total_time/step);
	printf("-------------------------------------------------------\n");
	printf("Detail:\n");
	printf("\n");
	for (int i = 0; i < gpu_num; ++i)
	{
		printf("GPU %d\n",i);
		printf("Outer_Compute_Time(include pre-stage):  %.3f ms\n", outer_compute_time[i]);
		printf("Inner_Compute_Time:                     %.3f ms\n", inner_compute_time[i]);
		printf("Total Compute_Time                      %.3f ms\n", total_compute_time[i]);
		printf("Extract_Bitmap_Time                     %.3f ms\n", extract_bitmap_time[i]);
	}
	printf("CPU \n");
	printf("CPU_Gather_Time:                            %.3f ms\n", gather_time);
	printf("--------------------------------------------------------\n");

	//clean
	for (int i = 0; i < gpu_num; ++i)
	{
		cudaSetDevice(i);
		//HANDLE_ERROR(cudaEventDestroy(start[i]));
		//HANDLE_ERROR(cudaEventDestroy(stop[i]));
		HANDLE_ERROR(cudaFree(d_edge_outer_src[i]));
		HANDLE_ERROR(cudaFree(d_edge_outer_dst[i]));
		HANDLE_ERROR(cudaFree(d_edge_inner_src[i]));
		HANDLE_ERROR(cudaFree(d_edge_inner_dst[i]));
		HANDLE_ERROR(cudaFree(d_value[i]));
		HANDLE_ERROR(cudaFree(d_flag[i]));

		HANDLE_ERROR(cudaDeviceReset());
		//error 
		//free(h_value[i]);
		free(h_flag[i]);
		free(stream[i]);
	}
	free(outer_compute_time);
	free(inner_compute_time);
	free(compute_time);
}
