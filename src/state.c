#include "./state.h"
#include "./utils.h"

#include <assert.h>
#include <stdbool.h>
#include <string.h>

typedef unsigned char idx_t;
/*
 *  [0,0] [1,0] [2,0]
 *  [0,1] [1,1] [2,1]
 *  [0,2] [1,2] [2,2]
 */

struct state_tag
{
    int         depth;
    state_panel pos[STATE_WIDTH][STATE_WIDTH];
    idx_t       i, j; /* pos of empty */
};

#define v(state, i, j) ((state)->pos[i][j])
#define ev(state) (v(state, state->i, state->j))
#define lv(state) (v(state, state->i - 1, state->j))
#define dv(state) (v(state, state->i, state->j + 1))
#define rv(state) (v(state, state->i + 1, state->j))
#define uv(state) (v(state, state->i, state->j - 1))

inline static State
state_alloc(void)
{
    return palloc(sizeof(struct state_tag));
}

inline static void
state_free(State state)
{
    pfree(state);
}

#ifndef NDEBUG
static void
validate_distinct_elem(state_panel v_list[STATE_WIDTH * STATE_WIDTH])
{
    for (idx_t i = 0; i < STATE_WIDTH * STATE_WIDTH; ++i)
        for (idx_t j = i + 1; j < STATE_WIDTH * STATE_WIDTH; ++j)
            assert(v_list[i] != v_list[j]);
}
#endif

State
state_init(state_panel v_list[STATE_WIDTH * STATE_WIDTH], int depth)
{
    State state = state_alloc();
    int   cnt   = 0;
#ifndef NDEBUG
    bool empty_found = false;

    validate_distinct_elem(v_list);
#endif

    assert(depth >= 0);

    state->depth = depth;
    for (idx_t j = 0; j < STATE_WIDTH; ++j)
        for (idx_t i = 0; i < STATE_WIDTH; ++i)
        {
            if (v_list[cnt] == STATE_EMPTY)
            {
                assert(!empty_found);
                state->i = i;
                state->j = j;
#ifndef NDEBUG
                empty_found = true;
#endif
            }
            v(state, i, j) = v_list[cnt++];
        }

    assert(empty_found);

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

inline static bool
state_left_movable(State state)
{
    return state->i != 0;
}
inline static bool
state_down_movable(State state)
{
    return state->j != STATE_WIDTH - 1;
}
inline static bool
state_right_movable(State state)
{
    return state->i != STATE_WIDTH - 1;
}
inline static bool
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

void
state_move(State state, Direction dir)
{
    assert(state_movable(state, dir));

    switch (dir)
    {
    case LEFT:
        ev(state) = lv(state);
        lv(state) = STATE_EMPTY;
        state->i--;
        break;
    case DOWN:
        ev(state) = dv(state);
        dv(state) = STATE_EMPTY;
        state->j++;
        break;
    case RIGHT:
        ev(state) = rv(state);
        rv(state) = STATE_EMPTY;
        state->i++;
        break;
    case UP:
        ev(state) = uv(state);
        uv(state) = STATE_EMPTY;
        state->j--;
        break;
    default:
        elog("unexpected direction");
        assert(false);
    }

    state->depth++;
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
	size_t hash_value = 0;
    for (idx_t i = 0; i < STATE_WIDTH; ++i)
        for (idx_t j = 0; j < STATE_WIDTH; ++j)
			hash_value ^= (v(state, i, j) << ((i*3+ j) << 2));
	return hash_value;
}

int
state_get_depth(State state)
{
    return state->depth;
}

void
state_dump(State state)
{
    elog("%s: depth=%d, (i,j)=(%u,%u)\n", __func__, state->depth, state->i,
         state->j);

    for (idx_t j = 0; j < STATE_WIDTH; ++j)
    {
        for (idx_t i = 0; i < STATE_WIDTH; ++i)
            elog("%u ", (unsigned int) v(state, i, j));
        elog("\n");
    }
    elog("-----------\n");
}

/*
 * Heuristic functions
 */

static state_panel from_x[STATE_WIDTH * STATE_WIDTH],
    from_y[STATE_WIDTH * STATE_WIDTH], to_x[STATE_WIDTH * STATE_WIDTH],
    to_y[STATE_WIDTH * STATE_WIDTH];

static int inline distance(int i, int j)
{
    return i > j ? i - j : j - i;
}

static void inline fill_from_to_xy(State from, State to)
{
    for (idx_t x = 0; x < STATE_WIDTH; ++x)
        for (idx_t y = 0; y < STATE_WIDTH; ++y)
        {
            from_x[v(from, x, y)] = x;
            from_y[v(from, x, y)] = y;
            to_x[v(to, x, y)]     = x;
            to_y[v(to, x, y)]     = y;
        }
}

int
heuristic_manhattan_distance(State from, State to)
{
    int h_value = 0;

    fill_from_to_xy(from, to);

    for (idx_t i = 1; i < STATE_WIDTH * STATE_WIDTH; ++i)
    {
        h_value += distance(from_x[i], to_x[i]);
        h_value += distance(from_y[i], to_y[i]);
    }

    return h_value;
}

int
heuristic_misplaced_tiles(State from, State to)
{
    int h_value = 0;

    fill_from_to_xy(from, to);

    for (idx_t i = 1; i < STATE_WIDTH * STATE_WIDTH; ++i)
        if (from_x[i] != to_x[i] || from_y[i] != to_y[i])
            h_value += 1;

    return h_value;
}

int
heuristic_tiles_out_of_row_col(State from, State to)
{
    int h_value = 0;

    fill_from_to_xy(from, to);

    for (idx_t i = 1; i < STATE_WIDTH * STATE_WIDTH; ++i)
    {
        if (from_x[i] != to_x[i])
            h_value += 1;
        if (from_y[i] != to_y[i])
            h_value += 1;
    }

    return h_value;
}
