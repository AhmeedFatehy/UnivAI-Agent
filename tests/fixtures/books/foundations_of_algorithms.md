# Foundations of Algorithms

A project-authored fixture book. The text is original and deliberately short;
it exists so ingestion, citation and planning can be verified without shipping
copyrighted material.

## Asymptotic Analysis

Running time is described as a function of input size rather than measured in
seconds, because seconds depend on the machine. Big-O notation states an upper
bound on growth: an algorithm that is O(n log n) never grows faster than some
constant multiple of n log n once the input is large enough.

Constant factors are dropped deliberately. Two sorting routines that both run in
O(n log n) may differ by a factor of three in practice, and choosing between them
is an engineering question, not an asymptotic one.

## Sorting Algorithms

Comparison sorts cannot beat O(n log n) in the worst case, because a comparison
tree with n! leaves has depth at least log(n!). Merge sort reaches that bound by
splitting the input in half, sorting each half and merging the two sorted runs in
linear time.

Quicksort partitions around a pivot instead of merging. Its average case is also
O(n log n), but a badly chosen pivot degrades it to O(n squared), which is why
production implementations randomise the pivot or fall back to heapsort.

Understanding asymptotic analysis is a prerequisite for comparing these
algorithms at all: without it, "faster" is an anecdote.

## Hash Tables

A hash table stores key-value pairs in an array of buckets, choosing the bucket
by applying a hash function to the key. When the hash function distributes keys
evenly, lookup, insertion and deletion all take expected constant time.

Collisions are unavoidable once the number of keys approaches the number of
buckets. Separate chaining stores colliding keys in a list per bucket; open
addressing probes for the next free slot. Both degrade as the load factor rises,
so a table is resized once it passes a threshold, typically around seventy
percent occupancy.

## Graph Traversal

Breadth-first search visits vertices in order of distance from the start vertex
and therefore finds shortest paths in unweighted graphs. Depth-first search
follows one branch to exhaustion before backtracking, which makes it the natural
basis for cycle detection and topological ordering.

Both traversals visit each vertex and each edge once, so both run in O(V + E)
time. The difference is the data structure that holds the frontier: a queue for
breadth-first, a stack for depth-first.

## Dynamic Programming

Dynamic programming applies when a problem has optimal substructure and
overlapping subproblems. The technique stores each subproblem's answer once and
reuses it, converting exponential recursion into polynomial time.

The knapsack problem is the standard illustration. Its recursive formulation
recomputes the same subproblems repeatedly; tabulating them over item index and
remaining capacity reduces the work to the product of those two dimensions.
Sorting and graph traversal are assumed knowledge here, since most dynamic
programming problems are stated over ordered or graph-shaped data.
