#include "./solver.h"
#include "./utils.h"

static State goal;

static bool
solver_dfs_internal(State s)
{
	State next_state;

    if (state_pos_equal(s, goal))
	{
		state_dump(s);
		state_fini(s);
        return true;
	}

    for (int dir = 0; dir < N_DIR; ++dir)
    {
        if (state_movable(s, dir))
        {
            next_state = state_copy(s);
            state_move(next_state, dir);
            if (solver_dfs_internal(next_state))
			{
				state_fini(s);
                return true;
			}
        }
    }

	state_fini(s);
    return false;
}

void
solver_dfs(State init_state, State goal_state)
{
	State s = state_copy(init_state);
    goal = goal_state;

    if (solver_dfs_internal(s))
        elog("%s: solved", __func__);
    else
        elog("%s: not solved", __func__);
}

void
solver_bfs(State init_state, State goal_state)
{
    (void) init_state;
    (void) goal_state;
}
