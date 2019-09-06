W =  Integrate[(-1)^l/(n+1)*Sum[(((n-l+1)*x)^k)/k!,{k,0,n}]*(-1)^l/(m+1)*Sum[(((m-l+1)*(x+y))^j)/j!,{j,0,m}] ,{x,0,1}]


Print[N[Expand[W]]]


