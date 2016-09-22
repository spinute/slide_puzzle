#include "./solver.h"
#include "./state.h"

int
main(int argc, char *argv[])
{
    state_panel s_list[STATE_WIDTH * STATE_WIDTH] = {1, 2, 3, 4,          5,
                                                     6, 7, 8, STATE_EMPTY};
    state_panel g1_list[STATE_WIDTH * STATE_WIDTH] = {1, 2, 3,           4, 5,
                                                      6, 7, STATE_EMPTY, 8};
    state_panel g2_list[STATE_WIDTH * STATE_WIDTH] = {1,           2, 3, 4, 5,
                                                      STATE_EMPTY, 7, 8, 6};
    state_panel g4_list[STATE_WIDTH * STATE_WIDTH] = {1, 2, 3, 4, STATE_EMPTY,
                                                      5, 7, 8, 6};

    state_panel g3_list[STATE_WIDTH * STATE_WIDTH] = {
        6, 1, 3, 2, 4, STATE_EMPTY, 5, 7, 8}; /* 17 moves*/
    State s = state_init(s_list, 0), g1 = state_init(g1_list, 0),
          g2 = state_init(g2_list, 0), g3 = state_init(g3_list, 0),
          g4 = state_init(g4_list, 0);

    solver_dfs(s, g1);
    solver_dfs(s, g2);
    solver_dfs(s, g3);
    solver_bfs(s, g1);
    solver_bfs(s, g2);
    solver_bfs(s, g3);

    solver_dls(s, g1, 1);
    solver_dls(s, g2, 1);
    // solver_dls(s, g3, 17); //dls(3**16==40000000)
    solver_iddfs(s, g1);
    solver_iddfs(s, g2);
    // solver_iddfs(s, g3); // 2min
    solver_iddfs(s, g4);

    solver_astar(s, g1, HeuristicManhattanDistance);
    solver_astar(s, g2, HeuristicManhattanDistance);
    solver_astar(s, g3, HeuristicManhattanDistance);
    solver_astar(s, g4, HeuristicManhattanDistance);
    solver_flastar(s, g1, HeuristicManhattanDistance, 1);
    solver_flastar(s, g2, HeuristicManhattanDistance, 1);
    solver_flastar(s, g3, HeuristicManhattanDistance, 17);
    solver_flastar(s, g4, HeuristicManhattanDistance, 2);
    solver_idastar(s, g1, HeuristicManhattanDistance);
    solver_idastar(s, g2, HeuristicManhattanDistance);
    solver_idastar(s, g3, HeuristicManhattanDistance);
    solver_idastar(s, g4, HeuristicManhattanDistance);

    state_fini(s);
    state_fini(g1);
    state_fini(g2);
    state_fini(g3);
    state_fini(g4);

    (void) argc;
    (void) argv;

    return 0;
}
