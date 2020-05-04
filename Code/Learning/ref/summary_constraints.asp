  summary(0, V, S, O) :- action(_, V, S, O).

  summary(1, verb(V,T), S, object(N2,D,A)) :- action(_, verb(V,T), S, object(N2,D,_)), action(_, verb(be,T), subject(it,_,_), object(_,_,A)).
  summary(2, verb(V,T), subject(N1,0,0), object(N3,D,conjunct(A1,A2))) :- action(_, verb(V,T), subject(conjunct(N1,N2),_,_), object(N3,D,A1)), action(_, verb(V,T), subject(N1,_,_), object(N3,D,A2)).

  summary(3, verb(V,T), S, object(conjunct(N1,N2),0,0)) :- action(I1, verb(V,T), S, object(N1,_,_)), action(I2, verb(V,T), S, object(N2,_,_)), N1 != N2, I1 < I2.
  summary(4, verb(V,T), S, object(0,0,conjunct(A1,A2))) :- action(I1, verb(V,T), S, object(_,_,A1)), action(I2, verb(V,T), S, object(_,_,A2)), A1 != A2, I1 < I2.

  summary(5, verb(be,T), S, object(N,D1,conjunct(A2,A3))) :- action(_, verb(be,T), S, object(N,D1,_)), action(_, verb(be,T), subject(N,D2,A1), object(_,_,conjunct(A2,A3))).

  complex_summary :- summary(I,V,S,O), I > 0.
  0{output(I,V,S,O)}1 :- summary(I,V,S,O).
  :- not output(_,_,_,_).

  :- output(_,verb(V_N,V_T),subject(S_N,S_D,S_A),object(O_N,O_D,O_A)), not verb(V_N,V_T)@2.
  :- output(_,verb(V_N,V_T),subject(S_N,S_D,S_A),object(O_N,O_D,O_A)), not subject(S_N,S_D,S_A)@1.
  :- output(_,verb(V_N,V_T),subject(S_N,S_D,S_A),object(O_N,O_D,O_A)), not object(O_N,O_D,O_A)@2.
