#include <stdbool.h>

#define WARP_SIZE 32
#define N_THREADS 32
#define N_BLOCK 48
#define N_CORE N_BLOCK * N_THREADS
#define PLAN_LEN_MAX 255

typedef unsigned char uchar;
typedef uchar DirDev;
#define dir_reverse_dev(dir) ((DirDev)(3 - (dir)))
#define DIR_N 4
#define DIR_FIRST 0
#define DIR_UP 0
#define DIR_RIGHT 1
#define DIR_LEFT 2
#define DIR_DOWN 3

typedef struct search_stat_tag
{
	bool solved;
	int len;
	int nodes_expanded;
} search_stat;

/* stack implementation */

__device__ __shared__ static struct dir_stack_tag
{
    uchar i;
    uchar buf[PLAN_LEN_MAX];
} stack[N_THREADS];

#define STACK_I (stack[threadIdx.x].i)
#define stack_get(i) (stack[threadIdx.x].buf[i])
#define stack_set(i, val) (stack[threadIdx.x].buf[i] = (val))

__device__ static inline void
stack_init(void)
{
    STACK_I = 0;
}

__device__ static inline void
stack_put(DirDev dir)
{
	stack_set(STACK_I, dir);
    ++STACK_I;
}
__device__ static inline bool
stack_is_empty(void)
{
    return STACK_I == 0;
}
__device__ static inline DirDev
stack_pop(void)
{
    --STACK_I;
    return stack_get(STACK_I);
}
__device__ static inline DirDev
stack_peak(void)
{
    return stack_get(STACK_I - 1);
}

/* state implementation */

#define STATE_WIDTH 4
#define STATE_N (STATE_WIDTH*STATE_WIDTH)

static char assert_state_width_is_four[STATE_WIDTH==4 ? 1 : -1];
#define POS_X(pos) ((pos) & 3)
#define POS_Y(pos) ((pos) >> 2)

/*
 * goal: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
 */

__device__ __shared__ static struct state_tag
{
	uchar tile[STATE_N];
    uchar              empty;
    uchar              h_value; /* ub of h_value is 6*16 */
} state[N_THREADS];


#define STATE_TILE(i) (state[threadIdx.x].tile[(i)])
#define STATE_EMPTY (state[threadIdx.x].empty)
#define STATE_HVALUE (state[threadIdx.x].h_value)

__device__ static uchar inline distance(uchar i, uchar j)
{
    return i > j ? i - j : j - i;
}

#define H_DIFF(opponent, empty, empty_dir)                                     \
    h_diff_table_shared[opponent][empty][empty_dir]
__device__ __shared__ static signed char h_diff_table_shared[STATE_N][STATE_N]
                                                            [DIR_N];

__device__ static void
state_init_hvalue(void)
{
    uchar from_x[STATE_N], from_y[STATE_N];

    STATE_HVALUE = 0;

    for (int i = 0; i < STATE_N; ++i)
    {
        from_x[STATE_TILE(i)] = POS_X(i);
        from_y[STATE_TILE(i)] = POS_Y(i);
    }
    for (int i = 1; i < STATE_N; ++i)
    {
        STATE_HVALUE += distance(from_x[i], POS_X(i));
        STATE_HVALUE += distance(from_y[i], POS_Y(i));
    }
}

__device__ static void
state_tile_fill(const uchar v_list[STATE_WIDTH * STATE_WIDTH])
{
    for (int i = 0; i < STATE_N; ++i)
    {
        if (v_list[i] == 0)
            STATE_EMPTY = i;
        state_tile_set(i, v_list[i]);
    }
}

__device__ static inline bool
state_is_goal(void)
{
    return STATE_HVALUE == 0;
}

__device__ static char assert_direction2
    [DIR_UP == 0 && DIR_RIGHT == 1 && DIR_LEFT == 2 && DIR_DOWN == 3 ? 1 : -1];
__device__ __shared__ static bool movable_table_shared[STATE_N][DIR_N];

__device__ static inline bool
state_movable(DirDev dir)
{
    return movable_table_shared[STATE_EMPTY][dir];
}

__device__ static char assert_direction
    [DIR_UP == 0 && DIR_RIGHT == 1 && DIR_LEFT == 2 && DIR_DOWN == 3 ? 1 : -1];
__device__ __constant__ const static int pos_diff_table[DIR_N] = {
    -STATE_WIDTH, 1, -1, +STATE_WIDTH};

__device__ static inline bool
state_move_with_limit(DirDev dir, unsigned int f_limit)
{
    int new_empty = STATE_EMPTY + pos_diff_table[dir];
    int opponent  = STATE_TILE(new_empty);
    int new_h_value =
        STATE_HVALUE + H_DIFF(opponent, new_empty, dir);

    if (STACK_I + 1 + new_h_value > f_limit)
        return false;

    STATE_HVALUE = new_h_value;
    state_tile_set(STATE_EMPTY, opponent);
    STATE_EMPTY = new_empty;

    return true;
}

__device__ static inline void
state_move(DirDev dir)
{
    int new_empty = STATE_EMPTY + pos_diff_table[dir];
    int opponent  = STATE_TILE(new_empty);

    STATE_HVALUE += H_DIFF(opponent, new_empty, dir);
    state_tile_set(STATE_EMPTY, opponent);
    STATE_EMPTY = new_empty;
}

/*
 * solver implementation
 */

__device__ static bool
idas_internal(int f_limit, int *ret_nodes_expanded)
{
    uchar dir = 0;
	int nodes_expanded = 0;

    for (;;)
    {
        if (state_is_goal())
        {
			*ret_nodes_expanded = nodes_expanded;
            return true;
        }

        if ((stack_is_empty() || stack_peak() != dir_reverse_dev(dir)) &&
            state_movable(dir))
        {
			++nodes_expanded;

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
			{
				*ret_nodes_expanded = nodes_expanded;
                return false;
			}

            dir = stack_pop();
            state_move(dir_reverse_dev(dir));
        }
    }
}

__global__ void
idas_kernel(uchar *input, signed char *plan, search_stat *stat, int f_limit,
            signed char *h_diff_table, bool *movable_table)
{
	int nodes_expanded = 0;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int id  = tid + bid * blockDim.x;

    for (int dir = 0; dir < DIR_N; ++dir)
        if (tid < STATE_N)
            movable_table_shared[tid][dir] = movable_table[tid * DIR_N + dir];
    for (int i = 0; i < STATE_N * DIR_N; ++i)
        if (tid < STATE_N)
            h_diff_table_shared[tid][i / DIR_N][i % DIR_N] =
                h_diff_table[tid * STATE_N * DIR_N + i];

    __syncthreads();

    stack_init();
    state_tile_fill(input + id * STATE_N);
    state_init_hvalue();

    if (idas_internal(f_limit, &nodes_expanded))
    {
		stat[id].solved = true;
        stat[id].len = (int) STACK_I;
		stat[id].nodes_expanded = nodes_expanded;
		for (uchar i                        = 0; i < STACK_I; ++i)
            plan[i + 1 + id * PLAN_LEN_MAX] = stack_get(i);
    }
    else
		stat[id].solved = false;
}

/* host library implementation */

#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <stdlib.h>

void *
palloc(size_t size)
{
    void *ptr = malloc(size);
    if (!ptr)
        elog("malloc failed\n");

    return ptr;
}

void *
repalloc(void *old_ptr, size_t new_size)
{
    void *ptr = realloc(old_ptr, new_size);
    if (!ptr)
        elog("realloc failed\n");

    return ptr;
}

void
pfree(void *ptr)
{
    if (!ptr)
        elog("empty ptr\n");
    free(ptr);
}


#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned char idx_t;
/*
 *  [0,0] [1,0] [2,0] [3,0]
 *  [0,1] [1,1] [2,1] [3,1]
 *  [0,2] [1,2] [2,2] [3,2]
 *  [0,3] [1,3] [2,3] [3,3]
 */

/*
 * goal state is
 * [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
 */

struct state_tag
{
    int         depth; /* XXX: needed? */
    state_panel pos[STATE_WIDTH][STATE_WIDTH];
    idx_t       i, j; /* pos of empty */
    Direction   parent;
    int         h_value;
};

#define v(state, i, j) ((state)->pos[i][j])
#define ev(state) (v(state, state->i, state->j))
#define lv(state) (v(state, state->i - 1, state->j))
#define dv(state) (v(state, state->i, state->j + 1))
#define rv(state) (v(state, state->i + 1, state->j))
#define uv(state) (v(state, state->i, state->j - 1))

static state_panel from_x[STATE_WIDTH * STATE_WIDTH],
    from_y[STATE_WIDTH * STATE_WIDTH];

static inline int
distance(int i, int j)
{
    return i > j ? i - j : j - i;
}

static inline void
fill_from_xy(State from)
{
    for (idx_t x = 0; x < STATE_WIDTH; ++x)
        for (idx_t y = 0; y < STATE_WIDTH; ++y)
        {
            from_x[v(from, x, y)] = x;
            from_y[v(from, x, y)] = y;
        }
}

static inline int
heuristic_manhattan_distance(State from)
{
    int h_value = 0;

    fill_from_xy(from);

    for (idx_t i = 1; i < STATE_WIDTH * STATE_WIDTH; ++i)
    {
        h_value += distance(from_x[i], i % STATE_WIDTH);
        h_value += distance(from_y[i], i / STATE_WIDTH);
    }

    return h_value;
}

bool
state_is_goal(State state)
{
    return state->h_value == 0;
}

static inline State
state_alloc(void)
{
    return (State)palloc(sizeof(struct state_tag));
}

static inline void
state_free(State state)
{
    pfree(state);
}

State
state_init(state_panel v_list[STATE_WIDTH * STATE_WIDTH])
{
    State state = state_alloc();
    int   cnt   = 0;

    state->depth  = 0;
    state->parent = (Direction)-1;

    for (idx_t j = 0; j < STATE_WIDTH; ++j)
        for (idx_t i = 0; i < STATE_WIDTH; ++i)
        {
            if (v_list[cnt] == STATE_EMPTY)
            {
                state->i = i;
                state->j = j;
            }
            v(state, i, j) = v_list[cnt++];
        }

    state->h_value = heuristic_manhattan_distance(state);
    state_dump(state);

    return state;
}

void
state_fini(State state)
{
    state_free(state);
}

State
state_copy(State src)
{
    State dst = state_alloc();

    memcpy(dst, src, sizeof(*src));

    return dst;
}

static inline bool
state_left_movable(State state)
{
    return state->i != 0;
}
static inline bool
state_down_movable(State state)
{
    return state->j != STATE_WIDTH - 1;
}
static inline bool
state_right_movable(State state)
{
    return state->i != STATE_WIDTH - 1;
}
static inline bool
state_up_movable(State state)
{
    return state->j != 0;
}

bool
state_movable(State state, Direction dir)
{
    return (dir != LEFT || state_left_movable(state)) &&
           (dir != DOWN || state_down_movable(state)) &&
           (dir != RIGHT || state_right_movable(state)) &&
           (dir != UP || state_up_movable(state));
}

/*
static inline int
calc_h_diff(idx_t who, idx_t from_x, idx_t from_y, Direction rdir)
{
    idx_t right_x = who % STATE_WIDTH;
    idx_t right_y = who / STATE_WIDTH;

    switch (rdir)
    {
    case LEFT:
        return right_x > from_x ? -1 : 1;
    case RIGHT:
        return right_x < from_x ? -1 : 1;
    case UP:
        return right_y > from_y ? -1 : 1;
    case DOWN:
        return right_y < from_y ? -1 : 1;
    }
}
*/
#define h_diff(who, from_i, from_j, dir)                                       \
    (h_diff_table[((who) << 6) + ((from_j) << 4) + ((from_i) << 2) + (dir)])
static int h_diff_table[STATE_N * STATE_N * N_DIR] = {
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

void
state_move(State state, Direction dir)
{
    idx_t who;
    assert(state_movable(state, dir));

    switch (dir)
    {
    case LEFT:
        who = ev(state) = lv(state);
        state->i--;
        break;
    case DOWN:
        who = ev(state) = dv(state);
        state->j++;
        break;
    case RIGHT:
        who = ev(state) = rv(state);
        state->i++;
        break;
    case UP:
        who = ev(state) = uv(state);
        state->j--;
        break;
    default:
        elog("unexpected direction");
        assert(false);
    }

    state->h_value =
        state->h_value + h_diff(who, state->i, state->j, dir_reverse(dir));
    // state->h_value = state->h_value + calc_h_diff(who, state->i, state->j,
    // dir);
    state->parent = dir;
}

bool
state_pos_equal(State s1, State s2)
{
    for (idx_t i = 0; i < STATE_WIDTH; ++i)
        for (idx_t j = 0; j < STATE_WIDTH; ++j)
            if (v(s1, i, j) != v(s2, i, j))
                return false;

    return true;
}

size_t
state_hash(State state)
{
    /* FIXME: for A* */
    size_t hash_value = 0;
    for (idx_t i = 0; i < STATE_WIDTH; ++i)
        for (idx_t j = 0; j < STATE_WIDTH; ++j)
            hash_value ^= (v(state, i, j) << ((i * 3 + j) << 2));
    return hash_value;
}
int
state_get_hvalue(State state)
{
    return state->h_value;
}

int
state_get_depth(State state)
{
    return state->depth;
}

void
state_dump(State state)
{
    elog("%s: h_value=%d, (i,j)=(%u,%u)\n", __func__, state->h_value, state->i,
         state->j);

    for (idx_t j = 0; j < STATE_WIDTH; ++j)
    {
        for (idx_t i = 0; i < STATE_WIDTH; ++i)
            elog("%u ", i == state->i && j == state->j ? 0 : v(state, i, j));
        elog("\n");
    }
    elog("-----------\n");
}

void
state_fill_slist(State state, unsigned char slist[])
{
    for (int i   = 0; i < STATE_N; ++i)
        slist[i] = state->pos[i % STATE_WIDTH][i / STATE_WIDTH];
    slist[state->i + (state->j * STATE_WIDTH)] = 0;
}

#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* XXX: hash function for State should be surveyed */
inline static size_t
hashfunc(State key)
{
    return state_hash(key);
}

typedef struct ht_entry_tag *HTEntry;
struct ht_entry_tag
{
    HTEntry  next;
    State    key;
    ht_value value;
};

static HTEntry
ht_entry_init(State key)
{
    HTEntry entry = palloc(sizeof(*entry));

    entry->key  = state_copy(key);
    entry->next = NULL;

    return entry;
}

static void
ht_entry_fini(HTEntry entry)
{
    pfree(entry);
}

struct ht_tag
{
    size_t   n_bins;
    size_t   n_elems;
    HTEntry *bin;
};

static bool
ht_rehash_required(HT ht)
{
    return ht->n_bins <= ht->n_elems; /* TODO: local policy is also needed */
}

static size_t
calc_n_bins(size_t required)
{
    /* NOTE: n_bins is used for mask and hence it should be pow of 2, fon now */
    size_t size = 1;
    assert(required > 0);

    while (required > size)
        size <<= 1;

    return size;
}

HT
ht_init(size_t init_size_hint)
{
    size_t n_bins = calc_n_bins(init_size_hint);
    HT     ht     = palloc(sizeof(*ht));

    ht->n_bins  = n_bins;
    ht->n_elems = 0;

    assert(sizeof(*ht->bin) <= SIZE_MAX / n_bins);
    ht->bin = palloc(sizeof(*ht->bin) * n_bins);
    memset(ht->bin, 0, sizeof(*ht->bin) * n_bins);

    return ht;
}

static void
ht_rehash(HT ht)
{
    HTEntry *new_bin;
    size_t   new_size = ht->n_bins << 1;

    assert(ht->n_bins<SIZE_MAX>> 1);

    new_bin = palloc(sizeof(*new_bin) * new_size);
    memset(new_bin, 0, sizeof(*new_bin) * new_size);

    for (size_t i = 0; i < ht->n_bins; ++i)
    {
        HTEntry entry = ht->bin[i];

        while (entry)
        {
            HTEntry next = entry->next;

            size_t idx   = hashfunc(entry->key) & (new_size - 1);
            entry->next  = new_bin[idx];
            new_bin[idx] = entry;

            entry = next;
        }
    }

    pfree(ht->bin);
    ht->n_bins = new_size;
    ht->bin    = new_bin;
}

void
ht_fini(HT ht)
{
    for (size_t i = 0; i < ht->n_bins; ++i)
    {
        HTEntry entry = ht->bin[i];
        while (entry)
        {
            HTEntry next = entry->next;
            state_fini(entry->key);
            ht_entry_fini(entry);
            entry = next;
        }
    }

    pfree(ht->bin);
    pfree(ht);
}

HTStatus
ht_search(HT ht, State key, ht_value *ret_value)
{
    size_t  i     = hashfunc(key) & (ht->n_bins - 1);
    HTEntry entry = ht->bin[i];

    while (entry)
    {
        if (state_pos_equal(key, entry->key))
        {
            *ret_value = entry->value;
            return HT_SUCCESS;
        }

        entry = entry->next;
    }

    return HT_FAILED_NOT_FOUND;
}

HTStatus
ht_insert(HT ht, State key, ht_value **value)
{
    size_t  i;
    HTEntry entry, new_entry;

    if (ht_rehash_required(ht))
        ht_rehash(ht);

    i     = hashfunc(key) & (ht->n_bins - 1);
    entry = ht->bin[i];

    while (entry)
    {
        if (state_pos_equal(key, entry->key))
        {
            *value = &entry->value;
            return HT_FAILED_FOUND;
        }

        entry = entry->next;
    }

    new_entry = ht_entry_init(key);

    new_entry->next = ht->bin[i];
    ht->bin[i]      = new_entry;
    *value          = &new_entry->value;

    assert(ht->n_elems < SIZE_MAX);
    ht->n_elems++;

    return HT_SUCCESS;
}

HTStatus
ht_delete(HT ht, State key)
{
    size_t  i     = hashfunc(key) & (ht->n_bins - 1);
    HTEntry entry = ht->bin[i], prev;

    if (!entry)
        return HT_FAILED_NOT_FOUND;

    if (state_pos_equal(key, entry->key))
    {
        ht->bin[i] = entry->next;
        ht_entry_fini(entry);
        return HT_SUCCESS;
    }

    prev  = entry;
    entry = entry->next;

    while (entry)
    {
        if (state_pos_equal(key, entry->key))
        {
            prev->next = entry->next;
            ht_entry_fini(entry);

            assert(ht->n_elems > 0);
            ht->n_elems--;

            return HT_SUCCESS;
        }

        prev  = entry;
        entry = entry->next;
    }

    return HT_FAILED_NOT_FOUND;
}

void
ht_dump(HT ht)
{
    elog("%s: n_elems=%zu, n_bins=%zu\n", __func__, ht->n_elems, ht->n_bins);
}


#include <stdlib.h>
#include <unistd.h>
static long pagesize = 0;
static long n_states_par_page;

/*
 * Queue Page implementation
 */

typedef struct queue_page *QPage;
typedef struct queue_page
{
    size_t out, in;
    QPage  next;
    State  buf[];
} QPageData;

static void
set_pagesize(void)
{
    pagesize = sysconf(_SC_PAGESIZE);
    if (pagesize < 0)
    {
        elog("%s: sysconf(_SC_PAGESIZE) failed\n", __func__);
        exit(EXIT_FAILURE);
    }

    n_states_par_page = (pagesize - sizeof(QPageData)) / sizeof(State);

    elog("%s: pagesize=%ld, n_states/page=%ld\n", __func__, pagesize,
         n_states_par_page);
}

static QPage
qpage_init(void)
{
    QPage qp = (QPage)palloc(sizeof(*qp) + sizeof(State) * n_states_par_page);
    qp->in = qp->out = 0;
    qp->next         = NULL;
    return qp;
}

static void
qpage_fini(QPage qp)
{
    while (qp->out < qp->in)
        state_fini(qp->buf[qp->out++]);
    pfree(qp);
}

static inline bool
qpage_have_space(QPage qp)
{
    return (long) (qp->in + 1) < n_states_par_page;
}

static inline void
qpage_put(QPage qp, State state)
{
    assert(qpage_have_space(qp));
    qp->buf[qp->in++] = state;
}

static inline State
qpage_pop(QPage qp)
{
    return qp->out == qp->in ? NULL : qp->buf[qp->out++];
}

/*
 * Queue implementation
 */

struct queue_tag
{
    QPage head, tail;
};

Queue
queue_init(void)
{
    Queue q = (Queue)palloc(sizeof(*q));

    if (!pagesize)
        set_pagesize();

    q->tail = q->head = qpage_init();
    q->head->in = q->head->out = 0;
    q->head->next              = NULL;

    return q;
}

void
queue_fini(Queue q)
{
    QPage page = q->head;

    while (page)
    {
        QPage next = page->next;
        qpage_fini(page);
        page = next;
    }

    pfree(q);
}

void
queue_put(Queue q, State state)
{
    if (!qpage_have_space(q->tail))
    {
        q->tail->next = qpage_init();
        q->tail       = q->tail->next;
    }

    qpage_put(q->tail, state);
}

State
queue_pop(Queue q)
{
    State state = qpage_pop(q->head);

    if (!state)
    {
        QPage next = q->head->next;
        if (!next)
            return NULL;

        state = qpage_pop(next);
        assert(state);

        qpage_fini(q->head);
        q->head = next;
    }

    return state;
}

void
queue_dump(Queue q)
{
    QPage page = q->head;
    int   cnt  = 0;

    while (page)
    {
        elog("%s: page#%d in=%zu, out=%zu", __func__, cnt++, page->in,
             page->out);
        page = page->next;
    }
}

bool
distributor(State init_state, State goal_state, unsigned char *s_list_ret,
                int distr_n)
{
    int      cnt = 0;
    State    state;
    PQ    q = pq_init();
    HTStatus ht_status;
    int *    ht_value;
    HT       closed = ht_init(10000);
    bool     solved = false;

    ht_status = ht_insert(closed, init_state, &ht_place_holder);
    pq_put(q, state_copy(init_state), state_get_hvalue(init_state), 0);
    ++cnt;

    while ((state = pq_pop(q)))
    {
        --cnt;
        if (state_is_goal(state))
        {
            solved = true;
            break;
        }

        ht_status = ht_insert(closed, state, &ht_value);
        if (ht_status == HT_FAILED_FOUND && *ht_value < state_get_depth(state))
        {
            state_fini(state);
            continue;
        }
        else
            *ht_value = state_get_depth(state);

        for (int dir = 0; dir < N_DIR; ++dir)
        {
            if (state_movable(state, (Direction)dir))
            {
                State next_state = state_copy(state);
                state_move(next_state, (Direction)dir);
				state->depth++;

                ht_status = ht_insert(closed, next_state, &ht_value);
                if (ht_status == HT_FAILED_FOUND &&
                    *ht_value <= state_get_depth(next_state))
                    state_fini(next_state);
                else
                {
					if (++cnt == distr_n)
					{
						/* NOTE: put parent.
						 * FIXME: There are duplicated younger siblings */
						*ht_value = state_get_depth(state);
						pq_put(q, state,
								*ht_value +
								calc_h_value(heuristic, state, goal_state));
						state_fini(next_state);
						break;
					}

					*ht_value = state_get_depth(next_state);
					pq_put(q, next_state,
							*ht_value + calc_h_value(heuristic, next_state, goal_state));
                }
            }
        }

        state_fini(state);
    }

    if (!solved)
        for (int i = 0; i < distr_n; ++i)
            state_fill_slist(pq_pop(q), s_list_ret + STATE_N * i);

    ht_fini(closed);
    pq_fini(q);

    return solved;
}

/* main */

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
calc_hvalue(uchar s_list[])
{
    int from_x[STATE_N], from_y[STATE_N];
    int h_value = 0;

    for (int i = 0; i < STATE_N; ++i)
    {
        from_x[s_list[i]] = POS_X(i);
        from_y[s_list[i]] = POS_Y(i);
    }
    for (int i = 1; i < STATE_N; ++i)
    {
        h_value += abs(from_x[i] - POS_X(i));
        h_value += abs(from_y[i] - POS_Y(i));
    }
    return h_value;
}

#define h_d_t(op, i, dir)                                                      \
    (h_diff_table[(op) *STATE_N * DIR_N + (i) *DIR_N + (dir)])
__host__ static void
init_mdist(signed char h_diff_table[])
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
                    h_d_t(opponent, i, dir) = goal_x > from_x ? -1 : 1;
                if (dir == DIR_RIGHT)
                    h_d_t(opponent, i, dir) = goal_x < from_x ? -1 : 1;
                if (dir == DIR_UP)
                    h_d_t(opponent, i, dir) = goal_y > from_y ? -1 : 1;
                if (dir == DIR_DOWN)
                    h_d_t(opponent, i, dir) = goal_y < from_y ? -1 : 1;
            }
        }
    }
}
#undef h_d_t

#define m_t(i, d) (movable_table[(i) *DIR_N + (d)])
__host__ static void
init_movable_table(bool movable_table[])
{
    for (int i = 0; i < STATE_N; ++i)
        for (unsigned int d = 0; d < DIR_N; ++d)
        {
            if (d == DIR_RIGHT)
                m_t(i, d) = (POS_X(i) < STATE_WIDTH - 1);
            else if (d == DIR_LEFT)
                m_t(i, d) = (POS_X(i) > 0);
            else if (d == DIR_DOWN)
                m_t(i, d) = (POS_Y(i) < STATE_WIDTH - 1);
            else if (d == DIR_UP)
                m_t(i, d) = (POS_Y(i) > 0);
        }
}
#undef m_t

static char dir_char[] = {'U', 'R', 'L', 'D'};

int
main(int argc, char *argv[])
{
    uchar  s_list[STATE_N * N_CORE];
    uchar *d_s_list;
    int    s_list_size = sizeof(uchar) * STATE_N * N_CORE;

    signed char  plan[PLAN_LEN_MAX * N_CORE];
    signed char *d_plan;
    int          plan_size = sizeof(signed char) * PLAN_LEN_MAX * N_CORE;
    search_stat  stat[N_CORE];
    search_stat *d_stat;
    int          stat_size = sizeof(search_stat) * N_CORE;

    int root_h_value = 0;

    bool  movable_table[STATE_N * DIR_N];
    bool *d_movable_table;
    int   movable_table_size = sizeof(bool) * STATE_N * DIR_N;

    signed char  h_diff_table[STATE_N * STATE_N * DIR_N];
    signed char *d_h_diff_table;
    int h_diff_table_size = sizeof(signed char) * STATE_N * STATE_N * DIR_N;

    if (argc < 2)
    {
        printf("usage: bin/cumain <ifname>\n");
        exit(EXIT_FAILURE);
    }

    load_state_from_file(argv[1], s_list);
    root_h_value = calc_hvalue(s_list);

    {
	    uchar goal[STATE_N];
	    State init_state = state_init(s_list, 0),
		  goal_state;

	    for (int i = 0; i < STATE_N; ++i)
		goal[i] = i;
	    goal_state = state_init(goal, 0);

	    if (distributor_bfs(init_state, goal_state, s_list, N_CORE))
	    {
		    puts("solution is found by distributor");
		    return 0;
	    }
    }

    init_mdist(h_diff_table);
    init_movable_table(movable_table);

    CUDA_CHECK(cudaMalloc((void **) &d_s_list, s_list_size));
    CUDA_CHECK(cudaMalloc((void **) &d_plan, plan_size));
    CUDA_CHECK(cudaMalloc((void **) &d_stat, stat_size));
    CUDA_CHECK(cudaMalloc((void **) &d_movable_table, movable_table_size));
    CUDA_CHECK(cudaMalloc((void **) &d_h_diff_table, h_diff_table_size));
    CUDA_CHECK(cudaMemcpy(d_movable_table, movable_table, movable_table_size,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h_diff_table, h_diff_table, h_diff_table_size,
                          cudaMemcpyHostToDevice));

    for (uchar f_limit = root_h_value;; f_limit+=2)
    {
        printf("f=%d\n", (int) f_limit);
        CUDA_CHECK(
            cudaMemcpy(d_s_list, s_list, s_list_size, cudaMemcpyHostToDevice));

        printf("call idas_kernel(block=%d, thread=%d)\n", N_BLOCK,
               N_CORE / N_BLOCK);
        idas_kernel<<<N_BLOCK, N_CORE/N_BLOCK>>>(d_s_list, d_plan, d_stat, f_limit, d_h_diff_table, d_movable_table);

        CUDA_CHECK(cudaMemcpy(plan, d_plan, plan_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(stat, d_stat, stat_size, cudaMemcpyDeviceToHost));

        for (int i = 0; i < N_CORE; ++i)
			if (stat[i].solved) {
                printf("len=%d: ", stat[i].len);
                for (int j = 0; j < stat[i].len; ++j)
                    printf("%c ", dir_char[(int) plan[i * PLAN_LEN_MAX + j]]);
                putchar('\n');
                goto solution_found;
            }

		printf("stat nodes_expanded\n");
        for (int i = 0; i < N_CORE; ++i)
			printf("%d, ", stat[i].nodes_expanded);
		putchar('\n');
    }
solution_found:

    CUDA_CHECK(cudaFree(d_s_list));
    CUDA_CHECK(cudaFree(d_plan));
    CUDA_CHECK(cudaFree(d_stat));
    CUDA_CHECK(cudaFree(d_movable_table));
    CUDA_CHECK(cudaFree(d_h_diff_table));
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
