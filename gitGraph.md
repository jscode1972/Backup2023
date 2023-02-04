```mermaid
---
title: Example Git diagram
---
gitGraph
   commit id: "on board"
   commit id: "start"
   branch trainer
   checkout trainer
   commit id: "create plan"
   checkout main
   merge trainer
   commit id: "modify plan"
   branch leader
   checkout leader
   commit id: "confirm plan"
   checkout main
   merge leader
   commit id: "quiza"
   commit id: "quizb"
   commit id: "quizc"
   branch engineer
   checkout engineer
   commit id: "judge"
   checkout main
   merge engineer
   commit id: "judge pass"
   checkout trainer
   merge main
   commit id: "trainer pass"
   checkout leader
   merge trainer
   commit id: "leader pass"
   checkout engineer
   merge leader
   commit id: "engineer pass"
   branch manager
   checkout manager
   commit id: "manager pass"
   checkout main
    merge manager
    
```

```
mindmap
Root
    A
      B
      C
```
