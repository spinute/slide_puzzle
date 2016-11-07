#include <stdbool.h>
#include <stdio.h>

typedef unsigned int uchar;

#define N_DIR 4
typedef uchar Direction;
#define dir_reverse(dir) (N_DIR - 1 - (dir))
#define DIR_UP 0
#define DIR_RIGHT 1
#define DIR_LEFT 2
#define DIR_DOWN 3

/* stack implementation */
#define PLAN_LEN_MAX 240

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
#define POS_X(pos) ((pos)%STATE_WIDTH)
#define POS_Y(pos) ((pos)/STATE_WIDTH)

/*
 * goal: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
 */

static struct state_tag
{
    unsigned char tile[STATE_N];
    uchar         empty;
    uchar h_value; /* ub of h_value is 6*16 */
} state;

static unsigned int inline distance(unsigned int i, unsigned int j)
{
    return i > j ? i - j : j - i;
}

#define H_DIFF(opponent, empty, empty_dir) h_diff_table[opponent][empty][empty_dir]
static int h_diff_table[STATE_N][STATE_N][N_DIR];

static void
init_mdist(void)
{
    for (int opponent = 0; opponent < STATE_N; ++opponent)
    {
        int goal_x = POS_X(opponent), goal_y = POS_Y(opponent);

		for (int i = 0; i < STATE_N; ++i)
		{
			int from_x = POS_X(i), from_y = POS_Y(i);
			for (uchar dir = 0; dir < N_DIR; ++dir)
			{
				if (dir == DIR_LEFT)
					H_DIFF(opponent, i, dir) =
						goal_x > from_x ? -1 : 1;
				if (dir == DIR_RIGHT)
					H_DIFF(opponent, i, dir) =
						goal_x < from_x ? -1 : 1;
				if (dir == DIR_UP)
					H_DIFF(opponent, i, dir) =
						goal_y > from_y ? -1 : 1;
				if (dir == DIR_DOWN)
					H_DIFF(opponent, i, dir) =
						goal_y < from_y ? -1 : 1;
			}
		}
    }
}

static inline void
state_init_hvalue(void)
{
    uchar from_x[STATE_N], from_y[STATE_N];

    for (int i = 0; i < STATE_N; ++i)
	{
		from_x[state.tile[i]] = POS_X(i);
		from_y[state.tile[i]] = POS_Y(i);
	}
    for (int i = 1; i < STATE_N; ++i)
    {
        state.h_value += distance(from_x[i], POS_X(i));
        state.h_value += distance(from_y[i], POS_Y(i));
    }
}

static void
state_tile_fill(const uchar v_list[STATE_N])
{
    for (int i = 0; i < STATE_N; ++i)
	{
		if (v_list[i] == STATE_EMPTY)
			state.empty = i;
		state.tile[i] = v_list[i];
	}
}

static char assert_direction2
    [DIR_UP == 0 && DIR_RIGHT == 1 && DIR_LEFT == 2 && DIR_DOWN == 3 ? 1 : -1];
static bool movable_table[STATE_N][N_DIR];

static void
init_movable_table(void)
{
	for(int i = 0; i < STATE_N; ++i)
		for (unsigned int d = 0; d < N_DIR; ++d)
		{
			if (d == DIR_RIGHT)
				movable_table[i][d] = (POS_X(i) < STATE_WIDTH - 1);
			else if (d == DIR_LEFT)
				movable_table[i][d] = (POS_X(i) > 0);
			else if (d == DIR_DOWN)
				movable_table[i][d] = (POS_Y(i) < STATE_WIDTH - 1);
			else if (d == DIR_UP)
				movable_table[i][d] = (POS_Y(i) > 0);
		}
}

static inline bool
state_movable(Direction dir)
{
	return movable_table[state.empty][dir];
}

static inline bool
state_is_goal(void)
{
	return !state.h_value;
}

static void
state_dump(void)
{
    printf("%s: h_value=%d, (i,j)=(%u,%u)\n",
			__func__, state.h_value, state.empty%STATE_WIDTH, state.empty/STATE_WIDTH);

	for (int i = 0; i < STATE_N; ++i)
		printf("%u%c", i == state.empty ? 0 : state.tile[i], POS_X(i)%STATE_WIDTH==STATE_WIDTH-1 ? '\n' : ' ');
    printf("-----------\n");
}

static char assert_direction
    [DIR_UP == 0 && DIR_RIGHT == 1 && DIR_LEFT == 2 && DIR_DOWN == 3 ? 1 : -1];
static int pos_diff_table[N_DIR] = {-STATE_WIDTH, 1, -1, +STATE_WIDTH};

static inline bool
state_move_with_limit(Direction dir, unsigned int f_limit)
{
	int new_empty = state.empty + pos_diff_table[dir];
    int opponent = state.tile[new_empty];
    int new_h_value = state.h_value + H_DIFF(opponent, new_empty, dir);

    if (stack.i + 1 + new_h_value > f_limit)
        return false;

    state.h_value = new_h_value;
    state.tile[state.empty] = opponent;
	state.empty = new_empty;

    return true;
}

static inline void
state_move(Direction dir)
{
	int new_empty = state.empty + pos_diff_table[dir];
    int opponent = state.tile[new_empty];

    state.h_value += H_DIFF(opponent, new_empty, dir);
    state.tile[state.empty] = opponent;
	state.empty = new_empty;
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
            state_movable(dir))
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
    init_mdist();
	init_movable_table();
    state_tile_fill(input);
    state_init_hvalue();
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
