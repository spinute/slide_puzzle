#include "./solver.h"
#include "./state.h"

int
main(int argc, char *argv[])
{
    value v_list[WIDTH * WIDTH] = {1, 2, 3, 4, 5, 6, 7, 8, VALUE_EMPTY};
    State s = state_init(v_list, 0), g = state_init(v_list, 0);

    solver_dfs(s, g);

    state_fini(s);
    state_fini(g);

    (void) argc;
    (void) argv;

    return 0;
}
