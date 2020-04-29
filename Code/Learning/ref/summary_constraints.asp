  summary(verb(be,T), S, object(N,D1,conjunct(A2,A3))) :- action(verb(be,T), S, object(N,D1,0)), action(verb(be,T), subject(N,D2,A1), object(0,0,conjunct(A2,A3))).

  :- summary(verb(V_N,V_T),subject(S_N,S_D,S_A),object(O_N,O_D,O_A)), not verb(V_N,V_T)@2.
  :- summary(verb(V_N,V_T),subject(S_N,S_D,S_A),object(O_N,O_D,O_A)), not subject(S_N,S_D,S_A)@1.
  :- summary(verb(V_N,V_T),subject(S_N,S_D,S_A),object(O_N,O_D,O_A)), not object(O_N,O_D,O_A)@2.
