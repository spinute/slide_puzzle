#define CUDA_CHECK(call) do { \
	const cudaError_t e = call; \
	if (e != cudaSuccess) { \
		printf("Error: %s:%d ", __FILE__, __LINE__); \
		printf("code:%d, reason: %s\n", e, cudaGetErrorString(e)); \
	} \
	exit(1); \
} while(0);
