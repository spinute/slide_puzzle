#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>

typedef unsigned char idx_t;

#define N_DIR 4
#define dir_reverse(dir) ((Direction)(3 - (dir)))
typedef unsigned char Direction;
#define DIR_UP 0
#define DIR_RIGHT 1
#define DIR_LEFT 2
#define DIR_DOWN 3

/* stack implementation */
#define STACK_SIZE_BYTES 64
#define STACK_BUF_BYTES (STACK_SIZE_BYTES - sizeof(unsigned char))
#define STACK_DIR_BITS 2
#define STACK_DIR_MASK ((1 << STACK_DIR_BITS) - 1)
#define PLAN_LEN_MAX ((1 << STACK_DIR_BITS) * STACK_BUF_BYTES)

#define stack_byte(i) (stack.buf[(i) >> STACK_DIR_BITS])
#define stack_ofs(i) ((i & STACK_DIR_MASK) << 1)
#define stack_get(i) ((stack_byte(i) & (STACK_DIR_MASK << stack_ofs(i))) >> stack_ofs(i))

static struct dir_stack_tag
{
    unsigned char i;
	unsigned char buf[STACK_BUF_BYTES];
} stack;

static inline void
stack_put(Direction dir)
{
	stack_byte(stack.i) &= ~(STACK_DIR_MASK << stack_ofs(stack.i));
    stack_byte(stack.i) |= dir << stack_ofs(stack.i);
	++stack.i;
}
static inline bool
stack_is_empty(void)
{
    return stack.i == 0;
}
static inline Direction
stack_pop(void)
{
    --stack.i;
	return stack_get(stack.i);
}
static inline Direction
stack_peak(void)
{
	return stack_get(stack.i - 1);
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
    idx_t         i, j;    /* pos of empty */
    unsigned char h_value; /* ub of h_value is 6*16 */
} state;

#define STATE_TILE(i, j) (state.tile[i][j])

static idx_t inline distance(idx_t i, idx_t j)
{
    return i > j ? i - j : j - i;
}

static inline void
state_init_hvalue(void)
{
    unsigned char from_x[STATE_WIDTH * STATE_WIDTH],
        from_y[STATE_WIDTH * STATE_WIDTH];
    idx_t x, y, i;

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
state_tile_fill(const unsigned char v_list[STATE_WIDTH * STATE_WIDTH])
{
    int   cnt = 0;
    idx_t i, j;

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
    return state.h_value == 0;
}

inline static bool
state_left_movable(void)
{
    return state.i != 0;
}
inline static bool
state_down_movable(void)
{
    return state.j != STATE_WIDTH - 1;
}
inline static bool
state_right_movable(void)
{
    return state.i != STATE_WIDTH - 1;
}
inline static bool
state_up_movable(void)
{
    return state.j != 0;
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
    idx_t i, j;
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

#define h_diff(dir)                                                            \
    (h_diff_table[(STATE_TILE(state.i, state.j) << 6) + ((state.j) << 4) +     \
                  ((state.i) << 2) + (dir)])
const static int h_diff_table[STATE_N * STATE_N * N_DIR] = {
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

static char assert_direction[DIR_UP==0&&DIR_RIGHT==1&&DIR_LEFT==2&&DIR_DOWN==3 ? 1 : -1];
static void
state_move(Direction dir)
{
    int i_diff = (dir & 1u) - ((dir & 2u) >> 1),
        j_diff = (dir & 1u) + ((dir & 2u) >> 1) - 1;

    STATE_TILE(state.i, state.j) =
        STATE_TILE(state.i + i_diff, state.j + j_diff);

    state.i += i_diff;
    state.j += j_diff;

    state.h_value += h_diff(dir_reverse(dir));
}

/*
 * solver implementation
 */

static bool
idas_internal(int f_limit)
{
    unsigned char dir = 0;

    for (;;)
    {
        if (state_is_goal())
        {
            state_dump();
            return true;
        }

        if ((stack_is_empty() || stack_peak() != dir_reverse(dir)) &&
            state_movable((Direction) dir))
        {
            state_move((Direction) dir);

            if (stack.i + state.h_value > (size_t) f_limit)
                state_move(dir_reverse(dir));
            else
            {
                stack_put((Direction) dir);
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
idas_kernel(unsigned char *input)
{
    int f_limit;
    state_tile_fill(input);
    state_init_hvalue();
    state_dump();

    for (f_limit = 1;; ++f_limit)
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

    return (int) rv;
}

#define MAX_LINE_LEN 100
static void
load_state_from_file(const char *fname, unsigned char *s)
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
    unsigned char s_list[STATE_N];
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
