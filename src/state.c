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
    state_panel pos[WIDTH][WIDTH];
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
validate_distinct_elem(state_panel v_list[WIDTH * WIDTH])
{
    for (idx_t i = 0; i < WIDTH * WIDTH; ++i)
        for (idx_t j = i + 1; j < WIDTH * WIDTH; ++j)
            assert(v_list[i] != v_list[j]);
}
#endif

State
state_init(state_panel v_list[WIDTH * WIDTH], int depth)
{
    State state = state_alloc();
    int   cnt   = 0;
#ifndef NDEBUG
    bool empty_found = false;

    validate_distinct_elem(v_list);
#endif

    assert(depth >= 0);

    state->depth = depth;
    for (idx_t j = 0; j < WIDTH; ++j)
        for (idx_t i = 0; i < WIDTH; ++i)
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
    return state->j != WIDTH - 1;
}
inline static bool
state_right_movable(State state)
{
    return state->i != WIDTH - 1;
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
    for (idx_t i = 0; i < WIDTH; ++i)
        for (idx_t j = 0; j < WIDTH; ++j)
            if (v(s1, i, j) != v(s2, i, j))
                return false;

    return true;
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

    for (idx_t j = 0; j < WIDTH; ++j)
    {
        for (idx_t i = 0; i < WIDTH; ++i)
            elog("%u ", (unsigned int) v(state, i, j));
        elog("\n");
    }
    elog("-----------\n");
}

/*
 * Heuristic functions
 */

int
heuristic_manhattan_distance(State from, State to)
{
	/* TODO: implement */
	(void) from;
	(void) to;
	return 0;
}
int
heuristic_linear_conflict(State from, State to)
{
	/* TODO: implement */
	(void) from;
	(void) to;
	return 0;
}
int
heuristic_pattern_database(State from, State to)
{
	/* TODO: implement */
	(void) from;
	(void) to;
	return 0;
}
int
heuristic_misplaced_tiles(State from, State to)
{
	/* TODO: implement */
	(void) from;
	(void) to;
	return 0;
}
int
heuristic_nillson(State from, State to)
{
	/* TODO: implement */
	(void) from;
	(void) to;
	return 0;
}
int
heuristic_n_max_swap(State from, State to)
{
	/* TODO: implement */
	(void) from;
	(void) to;
	return 0;
}
int
heuristic_xy(State from, State to)
{
	/* TODO: implement */
	(void) from;
	(void) to;
	return 0;
}
int
heuristic_tiles_out_of_row_col(State from, State to)
{
	/* TODO: implement */
	(void) from;
	(void) to;
	return 0;
}
