#include "./solver.h"
#include "./state.h"

int
main(int argc, char *argv[])
{
	value v_list[WIDTH * WIDTH] = {1,2,3,4,5,6,7,8,VALUE_EMPTY};
	struct state_tag s, g;

	state_init(&s, v_list, 0);
	state_init(&g, v_list, 0);

	solver_dfs(&s, &g);

    (void) argc;
    (void) argv;

    return 0;
}
