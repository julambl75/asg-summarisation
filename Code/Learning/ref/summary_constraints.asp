  summary(0, V, S, O) :- action(_, V, S, O).

  summary(1, verb(V,T), S, object(N2,D,A)) :- action(_, verb(V,T), S, object(N2,D,_)), action(_, verb(be,T), subject(it,_,_), object(_,_,A)).
  summary(2, verb(be,T), S, object(N,D1,conjunct(A2,A3))) :- action(_, verb(be,T), S, object(N,D1,_)), action(_, verb(be,T), subject(N,D2,A1), object(_,_,conjunct(A2,A3))).

  summary(3, V, subject(N1,0,0), object(N3,D,conjunct(A1,A2))) :- action(_, V, subject(conjunct(N1,N2),_,_), object(N3,D,A1)), action(_, V, subject(N1,_,_), object(N3,D,A2)).
  summary(4, V, S, object(0,0,conjunct(A1,A2))) :- action(I1, V, S, object(_,_,A1)), action(I2, V, S, object(_,_,A2)), A1 != A2, A1 != 0, A2 != 0, I1 < I2.

  summary(5, V, S, object(conjunct(N1,N2),0,0)) :- action(I1, V, S, object(N1,0,_)), action(I2, V, S, object(N2,0,_)), N1 != N2, N1 != 0, N2 != 0, I1 < I2.
  summary(6, V, S, object(conjunct(N1,N2),D,0)) :- action(I1, V, S, object(N1,0,_)), action(I2, V, S, object(N2,D,_)), N1 != N2, N1 != 0, N2 != 0, I1 < I2.
  summary(7, V, S, object(conjunct(N1,N2),D,0)) :- action(I1, V, S, object(N1,D,_)), action(I2, V, S, object(N2,0,_)), N1 != N2, N1 != 0, N2 != 0, I1 < I2.

  summary(I, V, S, object(conjunct(N1,N2),D,A)) :- summary(I, V, S, object(conjunct(conjunct(N1,N2),N3),D,A)).
  summary(I, V, S, object(conjunct(N1,N2),D,A)) :- summary(I, V, S, object(conjunct(N1,conjunct(N2,N3)),D,A)).
  summary(I, V, S, object(conjunct(N2,N3),D,A)) :- summary(I, V, S, object(conjunct(conjunct(N1,N2),N3),D,A)).
  summary(I, V, S, object(conjunct(N2,N3),D,A)) :- summary(I, V, S, object(conjunct(N1,conjunct(N2,N3)),D,A)).
  summary(I, V, S, object(conjunct(N1,N3),D,A)) :- summary(I, V, S, object(conjunct(conjunct(N1,N2),N3),D,A)).
  summary(I, V, S, object(conjunct(N1,N3),D,A)) :- summary(I, V, S, object(conjunct(N1,conjunct(N2,N3)),D,A)).

  summary(I, V, S, object(N,D,conjunct(A1,A2))) :- summary(I, V, S, object(N,D,conjunct(conjunct(A1,A2),A3))).
  summary(I, V, S, object(N,D,conjunct(A1,A2))) :- summary(I, V, S, object(N,D,conjunct(A1,conjunct(A2,A3)))).
  summary(I, V, S, object(N,D,conjunct(A2,A3))) :- summary(I, V, S, object(N,D,conjunct(conjunct(A1,A2),A3))).
  summary(I, V, S, object(N,D,conjunct(A2,A3))) :- summary(I, V, S, object(N,D,conjunct(A1,conjunct(A2,A3)))).
  summary(I, V, S, object(N,D,conjunct(A1,A3))) :- summary(I, V, S, object(N,D,conjunct(conjunct(A1,A2),A3))).
  summary(I, V, S, object(N,D,conjunct(A1,A3))) :- summary(I, V, S, object(N,D,conjunct(A1,conjunct(A2,A3)))).

  % Pick exactly one summary derivation for each sentence
  complex_summary :- summary(I,V,S,O), I > 0.
  0{output(I,V,S,O)}1 :- summary(I,V,S,O).
  :- not output(_,_,_,_).

  :- output(_,verb(V_N,V_T),subject(S_N,S_D,S_A),object(O_N,O_D,O_A)), not verb(V_N,V_T)@2.
  :- output(_,verb(V_N,V_T),subject(S_N,S_D,S_A),object(O_N,O_D,O_A)), not subject(S_N,S_D,S_A)@1.
  :- output(_,verb(V_N,V_T),subject(S_N,S_D,S_A),object(O_N,O_D,O_A)), not object(O_N,O_D,O_A)@2.
