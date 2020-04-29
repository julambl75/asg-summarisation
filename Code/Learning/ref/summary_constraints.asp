  summary(V, S, O) :- action(V, S, O).

  summary(verb(V,T), S, object(N2,D,A)) :- action(verb(V,T), S, object(N2,D,0)), action(verb(be,T), subject(it,0,0), object(0,0,A)).
  summary(verb(V,T), S, object(conjunct(N1,N2),0,0)) :- action(verb(V,T), S, object(N1,0,0)), action(verb(V,T), S, object(N2,0,0)), N1 != N2.
  summary(verb(V,T), subject(N1,0,0), object(N3,D,conjunct(A1,A2))) :- action(verb(V,T), subject(conjunct(N1,N2),0,0), object(N3,D,A1)), action(verb(V,T), subject(N1,0,0), object(N3,D,A2)).

  summary(verb(be,T), S, object(N,D1,conjunct(A2,A3))) :- action(verb(be,T), S, object(N,D1,0)), action(verb(be,T), subject(N,D2,A1), object(0,0,conjunct(A2,A3))).

  :- summary(verb(V_N,V_T),subject(S_N,S_D,S_A),object(O_N,O_D,O_A)), not verb(V_N,V_T)@2.
  :- summary(verb(V_N,V_T),subject(S_N,S_D,S_A),object(O_N,O_D,O_A)), not subject(S_N,S_D,S_A)@1.
  :- summary(verb(V_N,V_T),subject(S_N,S_D,S_A),object(O_N,O_D,O_A)), not object(O_N,O_D,O_A)@2.
