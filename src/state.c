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

void
state_copy(State src, State dst)
{
    memcpy(src, dst, sizeof(*src));
}

static void
state_validate_move(State state, Direction dir)
{
    assert(dir != Left || state_left_movable(state));
    assert(dir != Down || state_down_movable(state));
    assert(dir != Right || state_right_movable(state));
    assert(dir != Up || state_up_movable(state));
}

void
state_move(State state, Direction dir)
{
    state_validate_move(state, dir);

    switch (dir)
    {
    case Left:
        ev(state) = lv(state);
        lv(state) = VALUE_EMPTY;
        break;
    case Down:
        ev(state) = dv(state);
        dv(state) = VALUE_EMPTY;
        break;
    case Right:
        ev(state) = rv(state);
        rv(state) = VALUE_EMPTY;
        break;
    case Up:
        ev(state) = uv(state);
        uv(state) = VALUE_EMPTY;
        break;
    default:
        elog("unexpected direction");
        assert(false);
    }

    state->depth++;
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
}
