#include "./solver.h"
#include "./state.h"

int
main(int argc, char *argv[])
{
    value s_list[WIDTH * WIDTH]  = {1, 2, 3, 4, 5, 6, 7, 8, VALUE_EMPTY};
    value g1_list[WIDTH * WIDTH] = {1, 2, 3, 4, 5, 6, 7, VALUE_EMPTY, 8};
    value g2_list[WIDTH * WIDTH] = {1, 2, 3, 4, 5, VALUE_EMPTY, 7, 8, 6};
    State s = state_init(s_list, 0), g1 = state_init(g1_list, 0),
          g2 = state_init(g2_list, 0);

    solver_dfs(s, g1);
    solver_stack_dfs(s, g2);
    solver_bfs(s, g1);
    solver_bfs(s, g2);

    state_fini(s);
    state_fini(g1);
    state_fini(g2);

    (void) argc;
    (void) argv;

    return 0;
}
