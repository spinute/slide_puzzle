#include "./solver.h"
#include "./stack.h"
#include "./queue.h"
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
    goal    = goal_state;

    if (solver_dfs_internal(s))
        elog("%s: solved\n", __func__);
    else
        elog("%s: not solved\n", __func__);
}

void
solver_stack_dfs(State init_state, State goal_state)
{
    State state;
    Stack stack  = stack_init(123);
    bool  solved = false;
    stack_put(stack, state_copy(init_state));

    while ((state = stack_pop(stack)))
    {
        if (state_pos_equal(state, goal_state))
        {
            solved = true;
            break;
        }

        for (int dir = 0; dir < N_DIR; ++dir)
        {
            if (state_movable(state, dir))
            {
                State next_state = state_copy(state);
                state_move(next_state, dir);
                stack_put(stack, next_state);
            }
        }

        state_fini(state);
    }

    if (solved)
    {
        state_dump(state);
        elog("%s: solved\n", __func__);
    }
    else
        elog("%s: not solved\n", __func__);

    stack_fini(stack);
}

void
solver_bfs(State init_state, State goal_state)
{
    State state;
    Queue q  = queue_init();
    bool  solved = false;
    queue_put(q, state_copy(init_state));

    while ((state = queue_pop(q)))
    {
        if (state_pos_equal(state, goal_state))
        {
            solved = true;
            break;
        }

        for (int dir = 0; dir < N_DIR; ++dir)
        {
            if (state_movable(state, dir))
            {
                State next_state = state_copy(state);
                state_move(next_state, dir);
                queue_put(q, next_state);
            }
        }

        state_fini(state);
    }

    if (solved)
    {
        state_dump(state);
        elog("%s: solved\n", __func__);
    }
    else
        elog("%s: not solved\n", __func__);

	queue_fini(q);
}
