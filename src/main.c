#include "./solver.h"
#include "./state.h"

int
main(int argc, char *argv[])
{
    state_panel s_list[WIDTH * WIDTH]  = {1, 2, 3, 4, 5, 6, 7, 8, STATE_EMPTY};
    state_panel g1_list[WIDTH * WIDTH] = {1, 2, 3, 4, 5, 6, 7, STATE_EMPTY, 8};
    state_panel g2_list[WIDTH * WIDTH] = {1, 2, 3, 4, 5, STATE_EMPTY, 7, 8, 6};

    state_panel g3_list[WIDTH * WIDTH] = {6, 1, 3, 2, 4, STATE_EMPTY, 5, 7, 8}; /* 17 moves*/
    State       s = state_init(s_list, 0), g1 = state_init(g1_list, 0),
          g2 = state_init(g2_list, 0), g3 = state_init(g3_list, 0);

    // solver_dfs(s, g1); // can't solve this problem soon by dfs
    solver_dfs(s, g2);
    solver_bfs(s, g3);
    solver_bfs(s, g1);
    solver_bfs(s, g2);
    solver_bfs(s, g3);

    state_fini(s);
    state_fini(g1);
    state_fini(g2);

    (void) argc;
    (void) argv;

    return 0;
}
