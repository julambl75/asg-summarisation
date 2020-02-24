## Think about metric for summary (symbolic papers, blue and red)

We can use a text relationship map with sentences/clauses (see interim report). This can be constructed by having an index assigned to each one in the ASG semantic learning predicates, using a Python script to construct the graph, then adding rules in the ASG program to restrict the summary to the most connected nodes.
  
In addition we can use weak constraints to rule out certain summaries.

## Ideas for learning

- Recognize different actions with same subject -> remove one or somehow combine
- Recognize start and end of process -> do not talk about intermediate steps
- Preprocess sentence components to try and pick up synonyms using ConceptNet (replace synonyms with same word to reduce leaf node count)