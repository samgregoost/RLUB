W =  Integrate[(-1)^l/(n+1)*Exp[((n-l+1)*x)]*(-1)^l/(m+1)*Exp[((m-l+1)*(x+y))],{x,0,1}]


Print[N[Expand[W]]]


