#include <stdbool.h>
#include <stdio.h>

typedef unsigned char uchar;

#define STACK_SIZE_BYTES 64
#define STACK_BUF_BYTES (STACK_SIZE_BYTES - sizeof(uchar))
#define STACK_DIR_BITS 2
#define STACK_DIR_MASK ((1 << STACK_DIR_BITS) - 1)
#define PLAN_LEN_MAX ((1 << STACK_DIR_BITS) * STACK_BUF_BYTES)

#define dir_reverse(dir) (3 - (dir))
typedef uchar Direction;
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

/*
 * goal: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
 */

__device__ __shared__ static struct state_tag
{
    unsigned long long tile;    /* packed representation label(4bit)*16pos */
    uchar              i, j;    /* pos of empty */
    uchar              h_value; /* ub of h_value is 6*16 */
} state;

#define state_pos(i, j) (((j) << 2) + (i))
#define state_tile_ofs(i, j) (state_pos((i), (j)) << 2)
#define state_tile_get(i, j)                                                   \
    ((state.tile & (STATE_TILE_MASK << state_tile_ofs((i), (j)))) >>           \
     state_tile_ofs((i), (j)))
#define state_tile_set(i, j, val)                                              \
    do                                                                         \
    {                                                                          \
        state.tile &= ~((STATE_TILE_MASK) << state_tile_ofs((i), (j)));        \
        state.tile |= ((unsigned long long) val) << state_tile_ofs((i), (j));  \
    } while (0)

__device__ static uchar inline distance(uchar i, uchar j)
{
    return i > j ? i - j : j - i;
}

__device__ static inline void
state_init_hvalue(void)
{
    uchar from_x[STATE_WIDTH * STATE_WIDTH], from_y[STATE_WIDTH * STATE_WIDTH];

    state.h_value = 0;

    for (uchar x = 0; x < STATE_WIDTH; ++x)
        for (uchar y = 0; y < STATE_WIDTH; ++y)
        {
            from_x[state_tile_get(x, y)] = x;
            from_y[state_tile_get(x, y)] = y;
        }

    for (uchar i = 1; i < STATE_WIDTH * STATE_WIDTH; ++i)
    {
        state.h_value += distance(from_x[i], i % STATE_WIDTH);
        state.h_value += distance(from_y[i], i / STATE_WIDTH);
    }
}

__device__ static void
state_tile_fill(const uchar v_list[STATE_WIDTH * STATE_WIDTH])
{
    int cnt = 0;

    for (uchar j = 0; j < STATE_WIDTH; ++j)
        for (uchar i = 0; i < STATE_WIDTH; ++i)
        {
            if (v_list[cnt] == STATE_EMPTY)
            {
                state.i = i;
                state.j = j;
            }
            state_tile_set(i, j, v_list[cnt]);
            ++cnt;
        }
}

__device__ static inline bool
state_is_goal(void)
{
    return state.h_value == 0;
}

__device__ inline static bool
state_left_movable(void)
{
    return state.i != 0;
}
__device__ inline static bool
state_down_movable(void)
{
    return state.j != STATE_WIDTH - 1;
}
__device__ inline static bool
state_right_movable(void)
{
    return state.i != STATE_WIDTH - 1;
}
__device__ inline static bool
state_up_movable(void)
{
    return state.j != 0;
}

__device__ static inline bool
state_movable(Direction dir)
{
    return (dir == DIR_LEFT && state_left_movable()) ||
           (dir == DIR_RIGHT && state_right_movable()) ||
           (dir == DIR_DOWN && state_down_movable()) ||
           (dir == DIR_UP && state_up_movable());
}

#define h_diff(dir)                                                            \
    (h_diff_table[(state_tile_get(state.i, state.j) << 6) + ((state.j) << 4) +     \
                  ((state.i) << 2) + (dir)])
__constant__ const static int h_diff_table[STATE_N * STATE_N * DIR_N] = {
    1,  1,  1,  1,  1,  1,  -1, 1,  1,  1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  1,
    1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  1,  1,  -1, 1,
    -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  -1,
    1,  -1, 1,  -1, 1,  -1, 1,  1,  -1, 1,  1,  1,  1,  1,  1,  1,  1,  -1, 1,
    1,  1,  -1, 1,  -1, -1, 1,  1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  -1, 1,  -1,
    1,  -1, -1, 1,  1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, -1,
    1,  1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  1,  -1, 1,  1,  1,
    -1, 1,  1,  1,  1,  1,  1,  1,  1,  -1, 1,  -1, -1, 1,  1,  -1, -1, 1,  1,
    -1, 1,  1,  1,  -1, 1,  -1, 1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, 1,  1,
    1,  -1, 1,  -1, 1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, 1,  1,  1,  -1, 1,
    -1, 1,  1,  -1, 1,  1,  1,  -1, 1,  1,  1,  -1, 1,  1,  1,  1,  1,  1,  -1,
    -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, 1,  1,  1,  -1, -1, 1,  1,
    -1, -1, 1,  1,  -1, -1, 1,  1,  -1, 1,  1,  1,  -1, -1, 1,  1,  -1, -1, 1,
    1,  -1, -1, 1,  1,  -1, 1,  1,  1,  1,  1,  1,  -1, 1,  1,  -1, -1, 1,  1,
    -1, -1, 1,  1,  -1, -1, 1,  1,  1,  1,  1,  1,  -1, 1,  1,  1,  -1, 1,  1,
    1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,
    -1, 1,  1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  1,  -1, 1,
    -1, 1,  1,  1,  -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  -1, 1,  1,  1,  1,
    1,  1,  1,  1,  -1, 1,  1,  1,  -1, 1,  -1, -1, 1,  1,  -1, 1,  1,  1,  -1,
    1,  -1, 1,  -1, 1,  -1, 1,  -1, -1, 1,  1,  -1, 1,  1,  1,  -1, 1,  -1, 1,
    -1, 1,  -1, 1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  1,  -1,
    -1, 1,  -1, 1,  1,  1,  -1, 1,  1,  1,  1,  1,  1,  1,  1,  -1, 1,  -1, -1,
    1,  1,  -1, -1, 1,  1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  -1, -1, 1,  1,  -1,
    -1, 1,  1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  1,  -1, 1,  -1, 1,  -1, 1,  -1,
    1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  1,  1,  -1, 1,
    1,  1,  1,  1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, 1,
    1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, 1,  1,  1,  1,
    1,  1,  -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,  1,  -1,
    1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,  1,  1,  1,  1,  -1,
    1,  1,  1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  -1, 1,
    -1, 1,  -1, 1,  -1, 1,  1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  1,  -1, -1, 1,
    1,  -1, -1, 1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  1,  -1, -1, 1,  1,  -1, -1,
    1,  -1, 1,  1,  1,  1,  1,  1,  1,  1,  -1, 1,  1,  1,  -1, 1,  -1, -1, 1,
    1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  1,  -1, 1,  -1, 1,  -1,
    1,  -1, 1,  1,  1,  -1, 1,  1,  -1, -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,
    1,  1,  -1, 1,  1,  -1, -1, 1,  -1, 1,  1,  1,  -1, 1,  1,  1,  1,  1,  1,
    1,  1,  -1, 1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, 1,  1,  1,  -1, 1,  -1,
    1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  -1,
    1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  1,  1,
    -1, 1,  1,  1,  -1, 1,  1,  1,  1,  1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,
    -1, -1, 1,  1,  -1, 1,  1,  1,  1,  1,  1,  -1, 1,  1,  -1, -1, 1,  1,  -1,
    -1, 1,  1,  -1, -1, 1,  1,  1,  -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,
    -1, -1, 1,  1,  1,  -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,
    1,  1,  1,  1,  1,  -1, 1,  1,  1,  -1, 1,  1,  1,  -1, 1,  1,  -1, 1,  -1,
    1,  1,  1,  -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  -1, 1,  -1, 1,  1,  1,
    -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  1,
    -1, -1, 1,  1,  -1, -1, 1,  -1, 1,  1,  1,  1,  1,  1,  1,  1,  -1, 1,  1,
    1,  -1, 1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  1,  -1, -1,
    1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  1,  -1, -1, 1,  -1, 1,
    -1, 1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  1,  -1, -1, 1,  -1, 1,  1,  1,  -1,
    1,  1,  1,  1,  1,  1,  1,  1,  -1, 1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,
    -1, 1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1,
    1,  1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  1,  1,
    -1, 1,  -1, 1,  1,  1,  -1, 1,  1,  1,  -1, 1,  1,  1,  1,  1,  1};

__device__ static inline int
calc_hdiff(uchar who, uchar i, uchar j, Direction dir)
{
    /* TODO: optimize? */
    return dir == DIR_LEFT
               ? (who % STATE_WIDTH < i ? -1 : 1)
               : dir == DIR_RIGHT
                     ? (who % STATE_WIDTH > i ? -1 : 1)
                     : dir == DIR_UP ? (who / STATE_WIDTH < j ? -1 : 1)
                                     : (who / STATE_WIDTH > j ? -1 : 1);
}

static char assert_direction
    [DIR_UP == 0 && DIR_RIGHT == 1 && DIR_LEFT == 2 && DIR_DOWN == 3 ? 1 : -1];
__device__ static void
state_move(Direction dir)
{
    int i_diff = (dir & 1u) - ((dir & 2u) >> 1),
        j_diff = (dir & 1u) + ((dir & 2u) >> 1) - 1;

    state_tile_set(state.i, state.j,
        state_tile_get(state.i + i_diff, state.j + j_diff));

    state.i += i_diff;
    state.j += j_diff;

    state.h_value += h_diff(dir_reverse(dir));
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
            state_movable((Direction) dir))
        {
            state_move((Direction) dir);

            if (stack.i + state.h_value > f_limit)
                state_move(dir_reverse(dir));
            else
            {
                stack_put((Direction) dir);
                dir = DIR_FIRST;
                continue;
            }
        }

        while (++dir == DIR_N)
        {
            if (stack_is_empty())
                return false;

            stack_pop();

            state_move(dir_reverse(dir));
        }
    }
}

__global__ void
idas_kernel(uchar *input, int *plan)
{
    state_tile_fill(input);
    state_init_hvalue();

    for (uchar f_limit = 1;; ++f_limit)
        if (idas_internal(f_limit))
            break;
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

int
main(int argc, char *argv[])
{
    uchar  s_list[STATE_N];
    uchar *s_list_device;
    int    plan[PLAN_LEN_MAX];
    int *  plan_device;

    if (argc < 2)
    {
        printf("usage: bin/cumain <ifname>\n");
        exit(EXIT_FAILURE);
    }

    load_state_from_file(argv[1], s_list);
    CUDA_CHECK(cudaMalloc((uchar **) &s_list_device, sizeof(uchar) * STATE_N));
    CUDA_CHECK(cudaMalloc((void **) &plan_device, sizeof(int) * PLAN_LEN_MAX));
    CUDA_CHECK(cudaMemcpy(s_list_device, s_list, sizeof(uchar) * STATE_N,
                          cudaMemcpyHostToDevice));

    idas_kernel<<<1, 1>>>(s_list_device, plan_device);

    CUDA_CHECK(cudaMemcpy(plan, plan_device, sizeof(int) * PLAN_LEN_MAX,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(s_list_device));
    CUDA_CHECK(cudaFree(plan_device));

    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
