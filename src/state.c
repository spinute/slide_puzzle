#include "./state.h"
#include "./utils.h"

#include <assert.h>
#include <stdbool.h>
#include <string.h>

#define v(state, i, j) ((state)->pos[i][j])
#define ev(state) (v(state, state->i, state->j))
#define lv(state) (v(state, state->i - 1, state->j))
#define dv(state) (v(state, state->i, state->j + 1))
#define rv(state) (v(state, state->i + 1, state->j))
#define uv(state) (v(state, state->i, state->j - 1))

void
state_init(State state, value v_list[WIDTH * WIDTH], int depth)
{
    int cnt      = 0;
    state->depth = depth;
    for (idx_t j = 0; j < WIDTH; ++j)
        for (idx_t i = 0; i < WIDTH; ++i)
        {
            if (v_list[cnt] == VALUE_EMPTY)
            {
                state->i = i;
                state->j = j;
            }
            v(state, i, j) = v_list[cnt++];
        }
}

void
state_copy(State src, State dst)
{
    memcpy(src, dst, sizeof(*src));
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
    return (dir != Left || state_left_movable(state)) &&
           (dir != Down || state_down_movable(state)) &&
           (dir != Right || state_right_movable(state)) &&
           (dir != Up || state_up_movable(state));
}

void
state_move(State state, Direction dir)
{
    assert(state_movable(state, dir));

    switch (dir)
    {
    case Left:
        ev(state) = lv(state);
        lv(state) = VALUE_EMPTY;
        state->i--;
        break;
    case Down:
        ev(state) = dv(state);
        dv(state) = VALUE_EMPTY;
        state->j++;
        break;
    case Right:
        ev(state) = rv(state);
        rv(state) = VALUE_EMPTY;
        state->i++;
        break;
    case Up:
        ev(state) = uv(state);
        uv(state) = VALUE_EMPTY;
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

    elog("%s: (goal?) depth->%d\n", __func__, s1->depth);

    return true;
}

void
state_dump(State state)
{
    for (idx_t i = 0; i < WIDTH; ++i)
    {
        for (idx_t j = 0; j < WIDTH; ++j)
            elog("%u ", (unsigned int) v(state, i, j));
        elog("\n");
    }
    puts("--------");
}
