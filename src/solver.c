#include "./solver.h"
#include "./utils.h"

static State goal;

static bool
solver_dfs_internal(State s)
{
	struct state_tag next_state;

	if (state_pos_equal(s, goal))
		return true;

	for (int dir = 0; dir < N_DIR; ++dir)
	{
		if (state_movable(s, dir))
		{
			state_copy(s, &next_state);
			state_move(&next_state, dir);
			if (solver_dfs_internal(&next_state))
				return true;
		}
	}

	return false;
}

void
solver_dfs(State init_state, State goal_state)
{
	goal = goal_state;

	if (solver_dfs_internal(init_state))
		elog("solved");
	else
		elog("not solved");
}

void
solver_bfs(State init_state, State goal_state)
{
	(void) init_state;
	(void) goal_state;
}
