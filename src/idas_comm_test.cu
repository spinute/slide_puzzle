#include <stdbool.h>
#include <stdio.h>

#define N_CORE 48*32
#define N_BLOCK 48

typedef unsigned char uchar;

#define STACK_SIZE_BYTES 64
#define STACK_BUF_BYTES (STACK_SIZE_BYTES - sizeof(uchar))
#define STACK_DIR_BITS 2
#define STACK_DIR_MASK ((1 << STACK_DIR_BITS) - 1)
#define PLAN_LEN_MAX ((1 << STACK_DIR_BITS) * STACK_BUF_BYTES)

typedef uchar Direction;
#define dir_reverse(dir) ((Direction)(3 - (dir)))
#define DIR_N 4
#define DIR_FIRST 0
#define DIR_UP 0
#define DIR_RIGHT 1
#define DIR_LEFT 2
#define DIR_DOWN 3

/* stack implementation */

__device__ __shared__ static struct dir_stack_tag
{
    uchar i;
    uchar buf[STACK_BUF_BYTES];
} stack;

#define stack_byte(i) (stack.buf[(i) >> STACK_DIR_BITS])
#define stack_ofs(i) ((i & STACK_DIR_MASK) << 1)
#define stack_get(i)                                                           \
    ((stack_byte(i) & (STACK_DIR_MASK << stack_ofs(i))) >> stack_ofs(i))
__device__ static inline void
stack_put(Direction dir)
{
    stack_byte(stack.i) &= ~(STACK_DIR_MASK << stack_ofs(stack.i));
    stack_byte(stack.i) |= dir << stack_ofs(stack.i);
    ++stack.i;
}
__device__ static inline bool
stack_is_empty(void)
{
    return stack.i == 0;
    /* how about !stack.i */
}
__device__ static inline Direction
stack_pop(void)
{
    --stack.i;
    return stack_get(stack.i);
}
__device__ static inline Direction
stack_peak(void)
{
    return stack_get(stack.i - 1);
}

/* state implementation */

#define STATE_EMPTY 0
#define STATE_WIDTH 4
#define STATE_N STATE_WIDTH *STATE_WIDTH
#define STATE_TILE_BITS 4
#define STATE_TILE_MASK ((1ull << STATE_TILE_BITS) - 1)

#define POS_X(pos) ((pos) % STATE_WIDTH)
#define POS_Y(pos) ((pos) / STATE_WIDTH)

/*
 * goal: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
 */

__device__ __shared__ static struct state_tag
{
    unsigned long long tile; /* packed representation label(4bit)*16pos */
    uchar              empty;
    uchar              h_value; /* ub of h_value is 6*16 */
} state;

#define state_tile_ofs(i) (i << 2)
#define state_tile_get(i)                                                      \
    ((state.tile & (STATE_TILE_MASK << state_tile_ofs(i))) >> state_tile_ofs(i))
#define state_tile_set(i, val)                                                 \
    do                                                                         \
    {                                                                          \
        state.tile &= ~((STATE_TILE_MASK) << state_tile_ofs(i));               \
        state.tile |= ((unsigned long long) val) << state_tile_ofs(i);         \
    } while (0)

__device__ static uchar inline distance(uchar i, uchar j)
{
    return i > j ? i - j : j - i;
}

#define H_DIFF(opponent, empty, empty_dir)                                     \
	h_diff_table_shared[opponent][empty][empty_dir]
__device__ static signed char h_diff_table[STATE_N][STATE_N][DIR_N];
__device__ __shared__ static signed char h_diff_table_shared[STATE_N][STATE_N][DIR_N];

#define H_DIFF_HOST(opponent, empty, empty_dir)                                     \
	h_diff_table_host[opponent][empty][empty_dir]
static signed char h_diff_table_host[STATE_N][STATE_N][DIR_N];

	__host__ static void
init_mdist(void)
{
	for (int opponent = 0; opponent < STATE_N; ++opponent)
	{
		int goal_x = POS_X(opponent), goal_y = POS_Y(opponent);

		for (int i = 0; i < STATE_N; ++i)
		{
			int from_x = POS_X(i), from_y = POS_Y(i);
			for (uchar dir = 0; dir < DIR_N; ++dir)
			{
				if (dir == DIR_LEFT)
					H_DIFF_HOST(opponent, i, dir) = goal_x > from_x ? -1 : 1;
				if (dir == DIR_RIGHT)
					H_DIFF_HOST(opponent, i, dir) = goal_x < from_x ? -1 : 1;
				if (dir == DIR_UP)
					H_DIFF_HOST(opponent, i, dir) = goal_y > from_y ? -1 : 1;
				if (dir == DIR_DOWN)
					H_DIFF_HOST(opponent, i, dir) = goal_y < from_y ? -1 : 1;
			}
		}
	}
}

	__device__ static inline void
state_init_hvalue(void)
{
	uchar from_x[STATE_N], from_y[STATE_N];

	for (int i = 0; i < STATE_N; ++i)
	{
		from_x[state_tile_get(i)] = POS_X(i);
		from_y[state_tile_get(i)] = POS_Y(i);
	}
	for (int i = 1; i < STATE_N; ++i)
	{
		state.h_value += distance(from_x[i], POS_X(i));
		state.h_value += distance(from_y[i], POS_Y(i));
	}
}

	__device__ static void
state_tile_fill(const uchar v_list[STATE_WIDTH * STATE_WIDTH])
{
	for (int i = 0; i < STATE_N; ++i)
	{
		if (v_list[i] == STATE_EMPTY)
			state.empty = i;
		state_tile_set(i, v_list[i]);
	}
}

	__device__ static inline bool
state_is_goal(void)
{
	return state.h_value == 0;
}

__device__ static char assert_direction2
[DIR_UP == 0 && DIR_RIGHT == 1 && DIR_LEFT == 2 && DIR_DOWN == 3 ? 1 : -1];
__device__ static bool movable_table[STATE_N][DIR_N];
__device__ __shared__ static bool movable_table_shared[STATE_N][DIR_N];
static bool movable_table_host[STATE_N][DIR_N];

	__host__ static void
init_movable_table(void)
{
	for (int i = 0; i < STATE_N; ++i)
		for (unsigned int d = 0; d < DIR_N; ++d)
		{
			if (d == DIR_RIGHT)
				movable_table_host[i][d] = (POS_X(i) < STATE_WIDTH - 1);
			else if (d == DIR_LEFT)
				movable_table_host[i][d] = (POS_X(i) > 0);
			else if (d == DIR_DOWN)
				movable_table_host[i][d] = (POS_Y(i) < STATE_WIDTH - 1);
			else if (d == DIR_UP)
				movable_table_host[i][d] = (POS_Y(i) > 0);
		}
}
	__device__ static inline bool
state_movable(Direction dir)
{
	return movable_table_shared[state.empty][dir];
}

__device__ static char assert_direction
[DIR_UP == 0 && DIR_RIGHT == 1 && DIR_LEFT == 2 && DIR_DOWN == 3 ? 1 : -1];
__device__ __constant__ const static int pos_diff_table[DIR_N] = {-STATE_WIDTH, 1, -1,
	+STATE_WIDTH};

	__device__ static inline bool
state_move_with_limit(Direction dir, unsigned int f_limit)
{
	int new_empty   = state.empty + pos_diff_table[dir];
	int opponent    = state_tile_get(new_empty);
	int new_h_value = state.h_value + H_DIFF(opponent, new_empty, dir);

	if (stack.i + 1 + new_h_value > f_limit)
		return false;

	state.h_value = new_h_value;
	state_tile_set(state.empty, opponent);
	state.empty = new_empty;

	return true;
}

	__device__ static inline void
state_move(Direction dir)
{
	int new_empty = state.empty + pos_diff_table[dir];
	int opponent  = state_tile_get(new_empty);

	state.h_value += H_DIFF(opponent, new_empty, dir);
	state_tile_set(state.empty, opponent);
	state.empty = new_empty;
}

/*
 * solver implementation
 */

	__device__ static bool
idas_internal(uchar f_limit)
{
	uchar dir = 0;

	for (;;)
	{
		if (state_is_goal())
			return true;

		if ((stack_is_empty() || stack_peak() != dir_reverse(dir)) &&
				state_movable(dir))
		{
			if (state_move_with_limit(dir, f_limit))
			{
				stack_put(dir);
				dir = 0;
				continue;
			}
		}

		while (++dir == DIR_N)
		{
			if (stack_is_empty())
				return false;

			dir = stack_pop();
			state_move(dir_reverse(dir));
		}
	}
}

#define NOT_SOLVED -1
#define WARP_SIZE 32
	__global__ void
idas_kernel(uchar *input, char *plan, int f_limit)
{
	int tid = threadIdx.x;
	int core_id = tid + blockIdx.x * blockDim.x;
	int t_ofs = core_id * PLAN_LEN_MAX;
	bool solved;

	for (int i = 0; i < STATE_N*STATE_N*DIR_N/WARP_SIZE; ++i)
	{
		int d1 = i % STATE_N;
		int d2 = tid / DIR_N + i / STATE_N * WARP_SIZE / DIR_N;
		int d3 = tid % DIR_N;
		h_diff_table_shared[d1][d2][d3] = h_diff_table[d1][d2][d3];
	}
	for (int i = 0; i < STATE_N*DIR_N/WARP_SIZE; ++i)
	{
		int d1 = tid / DIR_N + i * WARP_SIZE / DIR_N;
		int d2 = tid % DIR_N;
		movable_table_shared[d1][d2] = movable_table[d1][d2];
	}

	__syncthreads();

	state_tile_fill(input + t_ofs);
	state_init_hvalue();

	solved = idas_internal(f_limit);

	plan[t_ofs] = solved ? (int) stack.i : NOT_SOLVED; /* len of plan */
	for (uchar i = 0; i < stack.i; ++i)
		plan[i + 1 + t_ofs] = stack_get(i);
}

/* host implementation */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#define exit_failure(...)                                                      \
	do                                                                         \
{                                                                          \
	printf(__VA_ARGS__);                                                   \
	exit(EXIT_FAILURE);                                                    \
} while (0)

	static int
pop_int_from_str(const char *str, char **end_ptr)
{
	long int rv = strtol(str, end_ptr, 0);
	errno       = 0;

	if (errno != 0)
		exit_failure("%s: %s cannot be converted into long\n", __func__, str);
	else if (end_ptr && str == *end_ptr)
		exit_failure("%s: reach end of string", __func__);

	if (rv > INT_MAX || rv < INT_MIN)
		exit_failure("%s: too big number, %ld\n", __func__, rv);

	return (int) rv;
}

#define MAX_LINE_LEN 100
	static void
load_state_from_file(const char *fname, uchar *s)
{
	FILE *fp;
	char  str[MAX_LINE_LEN];
	char *str_ptr = str, *end_ptr;

	fp = fopen(fname, "r");
	if (!fp)
		exit_failure("%s: %s cannot be opened\n", __func__, fname);

	if (!fgets(str, MAX_LINE_LEN, fp))
		exit_failure("%s: fgets failed\n", __func__);

	for (int i = 0; i < STATE_N; ++i)
	{
		s[i]    = pop_int_from_str(str_ptr, &end_ptr);
		str_ptr = end_ptr;
	}

	fclose(fp);
}
#undef MAX_LINE_LEN

#define CUDA_CHECK(call)                                                       \
	do                                                                         \
{                                                                          \
	const cudaError_t e = call;                                            \
	if (e != cudaSuccess)                                                  \
	exit_failure("Error: %s:%d code:%d, reason: %s\n", __FILE__,       \
			__LINE__, e, cudaGetErrorString(e));                  \
} while (0)

__host__ static int
host_distance(int i, int j)
{
	return i > j ? i - j : j - i;
}

	__host__ static int
calc_hvalue(uchar s_list[])
{
	uchar from_x[STATE_N], from_y[STATE_N];
	int h_value = 0;

	for (int i = 0; i < STATE_N; ++i)
	{
		from_x[s_list[i]] = POS_X(i);
		from_y[s_list[i]] = POS_Y(i);
	}
	for (int i = 1; i < STATE_N; ++i)
	{
		h_value += host_distance(from_x[i], POS_X(i));
		h_value += host_distance(from_y[i], POS_Y(i));
	}
	return h_value;
}

	int
main(int argc, char *argv[])
{
	uchar  s_list[STATE_N * N_CORE];
	uchar *s_list_device;
	char  plan[PLAN_LEN_MAX * N_CORE];
	char *plan_device;
	int insize = sizeof(uchar) * STATE_N * N_CORE;
	int outsize = sizeof(char) * PLAN_LEN_MAX * N_CORE;
	int root_h_value = 0;

	if (argc < 2)
	{
		printf("usage: bin/cumain <ifname>\n");
		exit(EXIT_FAILURE);
	}

	load_state_from_file(argv[1], s_list);
	root_h_value = calc_hvalue(s_list);

	/* fill roots */
	for (int i = 0; i < N_CORE; ++i)
		s_list[i] = s_list[i%STATE_N];

	CUDA_CHECK(cudaMalloc((void **) &s_list_device, insize));
	CUDA_CHECK(cudaMalloc((void **) &plan_device, outsize));
	CUDA_CHECK(cudaMemcpy(s_list_device, s_list, insize,
				cudaMemcpyHostToDevice));

	init_mdist();
	init_movable_table();
	(void) assert_direction[0];
	(void) assert_direction2[0];
	CUDA_CHECK(cudaMemcpy(&movable_table, &movable_table_host,
				sizeof(bool) * STATE_N * DIR_N, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(&h_diff_table, &h_diff_table_host,
				sizeof(int) * STATE_N * STATE_N * DIR_N, cudaMemcpyHostToDevice));

	for (uchar f_limit = root_h_value;; ++f_limit)
	{
		CUDA_CHECK(cudaMemcpy(s_list_device, s_list, insize,
					cudaMemcpyHostToDevice));

		idas_kernel<<<N_BLOCK, N_CORE/N_BLOCK>>>(s_list_device, plan_device, f_limit);

		CUDA_CHECK(cudaMemcpy(plan, plan_device, outsize,
					cudaMemcpyDeviceToHost));

		for (int i = 0; i < N_CORE; ++i)
			if (plan[i] != NOT_SOLVED)
			{
				printf("len=%d: ", (int)plan[0]);
				for (int j = 0; j < plan[0]; ++j)
					printf("%d ", (int) plan[j+1]);
				putchar('\n');
				goto solution_found;
			}
	}
solution_found:

    CUDA_CHECK(cudaFree(s_list_device));
    CUDA_CHECK(cudaFree(plan_device));
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
