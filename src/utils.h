#pragma once

#include <assert.h>

#include <stdarg.h>
#include <stdio.h>

#define elog(...) fprintf(stderr, __VA_ARGS__)

void *palloc(size_t size);
void *repalloc(void *ord_ptr, size_t size);
void pfree(void *ptr);
