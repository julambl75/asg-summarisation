## Next step

We need to simplify and formalize (i.e., figure out minimal set of hard-coded rules).

## Create positive/negative example to learn (stay :- ...)

We should use ILASP to add compaction as general rule (as second ILASP to get generic rules), meaning 2 sets of positive/negative examples (second time have summaries as examples), or use ASP choice rules for summaries (:- p,q,r,not a is same as a:-p,q,r). It could be useful though to do :-p,q,r,not a,not b).


## Think about metric for summary (symbolic papers, blue and red)

We can use a text relationship map with sentences/clauses (see interim report). This can be constructed by having an index assigned to each one in the ASG semantic learning predicates, using a Python script to construct the graph, then adding rules in the ASG program to restrict the summary to the most connected nodes.
  
In addition we can use weak constraints to rule out certain summaries.

---
For Mary example:
- good: was raining, stayed
- bad: entered, double-movement (out+back)

asg general_learned_actions.asg --mode=learn --depth=10 --ILASP-ss-options="-ml=5 --max-rule-length=5"

When becomes UNSAT, change learn to ss, check if rule in output and if not increase params or check mode declarations.

Bugs (send to Mark):
- #bias(":- not body(node_rule(_, _)).").
- #maxv(4).
- hypothesis space invalid without bias constraint
