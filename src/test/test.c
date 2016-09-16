#include "./test.h"
#include "stdio.h"

int
main(void)
{
    RUN_TEST_GROUP(state);
    RUN_TEST_GROUP(stack);

    puts("\nTest finished.");
    return 0;
}
