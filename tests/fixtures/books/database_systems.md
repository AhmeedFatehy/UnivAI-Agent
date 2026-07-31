# Database Systems

A project-authored fixture book. The text is original and deliberately short; it
exists so ingestion, citation and planning can be verified without shipping
copyrighted material.

## The Relational Model

A relation is a set of tuples over a fixed set of attributes. Because it is a
set, a relation has no inherent order and no duplicate tuples, which is why a
query that appears to return rows "in order" is relying on an accident of
execution unless it sorts explicitly.

Keys identify tuples. A candidate key is a minimal attribute set that is unique
across the relation; the primary key is the candidate key chosen for reference by
other relations.

## Normalisation

Normalisation removes redundancy by decomposing relations along their functional
dependencies. Second normal form eliminates partial dependence on a composite
key; third normal form eliminates transitive dependence through a non-key
attribute.

Each decomposition trades write anomalies for read joins. A schema normalised to
third normal form updates cleanly but may need several joins to answer a common
query, which is why reporting schemas are often deliberately denormalised. The
relational model is assumed knowledge throughout this chapter.

## Hash Indexes

A hash index stores entries in buckets chosen by a hash function over the
indexed column, giving expected constant-time equality lookup. Collisions are
handled by chaining or by probing, and the index is rebuilt or extended once the
load factor rises past its threshold.

The trade-off against a B-tree index is range queries: a hash index answers
"equals" in constant time but cannot answer "between" at all, because hashing
destroys the ordering of the key space.

## Query Execution

A query plan is a tree of physical operators — scans, joins, aggregations —
that the executor evaluates bottom up. The planner chooses between plans using
cardinality estimates drawn from table statistics, so stale statistics produce
bad plans far more often than a bad planner does.

Join order dominates cost in multi-table queries. The search space grows
factorially with the number of relations, so planners use dynamic programming
over subsets of relations, or switch to a heuristic search once the query is
large enough that exhaustive search is impractical.

## Transactions and Isolation

A transaction groups operations so that they take effect entirely or not at all.
Atomicity and durability are provided by a write-ahead log: changes are recorded
in the log before they are applied, so an interrupted transaction can be undone
and a committed one replayed.

Isolation levels trade correctness against concurrency. Read committed prevents
dirty reads but permits non-repeatable reads; serialisable forbids all anomalies
at the cost of aborting more transactions. Choosing a level is a decision about
which anomalies an application can tolerate, not about which is "safest".
