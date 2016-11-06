#include <stdbool.h>
#include <stdio.h>

typedef unsigned int uchar;

#define N_DIR 4
typedef uchar Direction;
#define dir_reverse(dir) ((Direction)(3 - (dir)))
#define DIR_UP 0
#define DIR_RIGHT 1
#define DIR_LEFT 2
#define DIR_DOWN 3

/* stack implementation */
#define PLAN_LEN_MAX 128

static struct dir_stack_tag
{
    int    i;
    unsigned char buf[PLAN_LEN_MAX];
} stack;

static inline void
stack_put(Direction dir)
{
    stack.buf[++stack.i] = dir;
}
static inline bool
stack_is_empty(void)
{
    return !stack.i;
}
static inline Direction
stack_pop(void)
{
    return stack.buf[stack.i--];
}
static inline Direction
stack_top(void)
{
    return stack.buf[stack.i];
}

/* state implementation */

#define STATE_EMPTY 0
#define STATE_WIDTH 4
#define STATE_N STATE_WIDTH *STATE_WIDTH

/*
 * goal: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
 */

static struct state_tag
{
    unsigned char tile[STATE_WIDTH][STATE_WIDTH];
    uchar         i, j;    /* pos of empty */
    uchar h_value; /* ub of h_value is 6*16 */
} state;

#define STATE_TILE(i, j) (state.tile[i][j])

static uchar inline distance(uchar i, uchar j)
{
    return i > j ? i - j : j - i;
}

#define H_DIFF(opponent, from_x, from_y, dir) h_diff_table[opponent][from_x][from_y][dir]
static int h_diff_table[STATE_N][STATE_WIDTH][STATE_WIDTH][N_DIR];

static void
init_mdist(void)
{
    for (int opponent = 0; opponent < STATE_N; ++opponent)
    {
        int goal_x = opponent % STATE_WIDTH, goal_y = opponent / STATE_WIDTH;

        for (int from_x = 0; from_x < STATE_WIDTH; ++from_x)
            for (int from_y = 0; from_y < STATE_WIDTH; ++from_y)
                for (uchar dir = 0; dir < N_DIR; ++dir)
                {
                    if (dir == DIR_LEFT)
                        H_DIFF(opponent, from_x, from_y, dir) =
                            goal_x > from_x ? -1 : 1;
                    if (dir == DIR_RIGHT)
                        H_DIFF(opponent, from_x, from_y, dir) =
                            goal_x < from_x ? -1 : 1;
                    if (dir == DIR_UP)
                        H_DIFF(opponent, from_x, from_y, dir) =
                            goal_y > from_y ? -1 : 1;
                    if (dir == DIR_DOWN)
                        H_DIFF(opponent, from_x, from_y, dir) =
                            goal_y < from_y ? -1 : 1;
                }
    }
}

static inline void
state_init_hvalue(void)
{
    uchar from_x[STATE_WIDTH * STATE_WIDTH],
        from_y[STATE_WIDTH * STATE_WIDTH];
    uchar x, y, i;

    state.h_value = 0;

    for (x = 0; x < STATE_WIDTH; ++x)
        for (y = 0; y < STATE_WIDTH; ++y)
        {
            from_x[STATE_TILE(x, y)] = x;
            from_y[STATE_TILE(x, y)] = y;
        }

    for (i = 1; i < STATE_WIDTH * STATE_WIDTH; ++i)
    {
        state.h_value += distance(from_x[i], i % STATE_WIDTH);
        state.h_value += distance(from_y[i], i / STATE_WIDTH);
    }
}

static void
state_tile_fill(const uchar v_list[STATE_WIDTH * STATE_WIDTH])
{
    int   cnt = 0;
    uchar i, j;

    for (j = 0; j < STATE_WIDTH; ++j)
        for (i = 0; i < STATE_WIDTH; ++i)
        {
            if (v_list[cnt] == STATE_EMPTY)
            {
                state.i = i;
                state.j = j;
            }
            STATE_TILE(i, j) = v_list[cnt++];
        }
}

static inline bool
state_is_goal(void)
{
    return !state.h_value;
}

inline static bool
state_left_movable(void)
{
    return state.i;
}
inline static bool
state_down_movable(void)
{
    return STATE_WIDTH - 1 - state.j;
}
inline static bool
state_right_movable(void)
{
    return STATE_WIDTH - 1 - state.i;
}
inline static bool
state_up_movable(void)
{
    return state.j;
}

static inline bool
state_movable(Direction dir)
{
    return (dir == DIR_LEFT && state_left_movable()) ||
           (dir == DIR_RIGHT && state_right_movable()) ||
           (dir == DIR_DOWN && state_down_movable()) ||
           (dir == DIR_UP && state_up_movable());
}
static void
state_dump(void)
{
    uchar i, j;
    printf("%s: h_value=%d, (i,j)=(%u,%u)\n", __func__, state.h_value, state.i,
           state.j);

    for (j = 0; j < STATE_WIDTH; ++j)
    {
        for (i = 0; i < STATE_WIDTH; ++i)
            printf("%u ", i == state.i && j == state.j ? 0 : STATE_TILE(i, j));
        printf("\n");
    }
    printf("-----------\n");
}

static char assert_direction
    [DIR_UP == 0 && DIR_RIGHT == 1 && DIR_LEFT == 2 && DIR_DOWN == 3 ? 1 : -1];
#define get_new_i(state, dir) (state.i + (dir & 1u) - ((dir & 2u) >> 1))
#define get_new_j(state, dir) (state.j + (dir & 1u) + ((dir & 2u) >> 1) - 1)

static inline bool
state_move_with_limit(Direction dir, unsigned int f_limit)
{
    int new_i = get_new_i(state, dir), new_j = get_new_j(state, dir);
    int opponent = STATE_TILE(new_i, new_j);
    int new_h_value =
        state.h_value + H_DIFF(opponent, new_i, new_j, dir);

    if (stack.i + 1 + new_h_value > f_limit)
        return false;

    state.h_value = new_h_value;
    STATE_TILE(state.i, state.j) = opponent;
    state.i = new_i;
    state.j = new_j;

    return true;
}

static inline void
state_move(Direction dir)
{
    int new_i = get_new_i(state, dir), new_j = get_new_j(state, dir);
    int opponent = STATE_TILE(new_i, new_j);

    state.h_value += H_DIFF(opponent, new_i, new_j, dir);
    STATE_TILE(state.i, state.j) = opponent;
    state.i = new_i;
    state.j = new_j;
}

/*
 * solver implementation
 */

static bool
idas_internal(int f_limit)
{
    uchar dir = 0;

    for (;;)
    {
        if (state_is_goal())
        {
            state_dump();
            return true;
        }

        if ((stack_is_empty() || stack_top() != dir_reverse(dir)) &&
            state_movable((Direction) dir))
        {
            if (state_move_with_limit(dir, f_limit))
            {
                stack_put(dir);
                dir = 0;
                continue;
            }
        }

        while (++dir == N_DIR)
        {
            if (stack_is_empty())
                return false;

            dir = stack_pop();
            state_move(dir_reverse(dir));
        }
    }
}

void
idas_kernel(uchar *input)
{
    int f_limit;
    state_tile_fill(input);
    state_init_hvalue();
    init_mdist();
    state_dump();

    for (f_limit = state.h_value;; ++f_limit)
        if (idas_internal(f_limit))
            break;
}

/* host implementation */

#include <errno.h>
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

    return (int) rv;
}

#define MAX_LINE_LEN 100
static void
load_state_from_file(const char *fname, uchar *s)
{
    FILE *fp;
    char  str[MAX_LINE_LEN];
    char *str_ptr = str, *end_ptr;
    int   i;

    fp = fopen(fname, "r");
    if (!fp)
        exit_failure("%s: %s cannot be opened\n", __func__, fname);

    if (!fgets(str, MAX_LINE_LEN, fp))
        exit_failure("%s: fgets failed\n", __func__);

    for (i = 0; i < STATE_N; ++i)
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
    uchar s_list[STATE_N];
    int           plan[PLAN_LEN_MAX];

    if (argc < 2)
    {
        printf("usage: bin/cumain <ifname>\n");
        exit(EXIT_FAILURE);
    }

    load_state_from_file(argv[1], s_list);
    idas_kernel(s_list);

    return 0;
}
