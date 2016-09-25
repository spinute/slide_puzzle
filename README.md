# Sliding puzzle solver

# Rake tasks
* Build: `rake` (Default CC=clang)
* Test: `rake test` (Using <https://github.com/ThrowTheSwitch/Unity>)
* Benchmark: (not implemented yet)

## Strategy
* [x] DFS
* [x] BFS
* [x] DLS
* [x] IDDFS
* [x] A\*
* [x] FLA\*
* [x] IDA\*

## Heuristic
* [x] Manhattan distance
* [x] Misplaced tiles
* [x] Misplaced rows/cols
* [ ] Pattern Database

## TODO
* Implement GPU search(At first, single thread DFS/IDA\* to evaluate max speedup)
* Improve solver efficiency by following the work of Burns; 2012
* Consider more sophisticated way of state space search on GPU
