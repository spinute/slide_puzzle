#include "./test.h"
#include "stdio.h"

int
main(void)
{
    RUN_TEST_GROUP(state);
    RUN_TEST_GROUP(stack);
    RUN_TEST_GROUP(queue);

    puts("\nTest finished.");
    return 0;
}
