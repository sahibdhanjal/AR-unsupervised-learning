function y=sat(x)
kk=1;
% sat is the saturation function with unit limits and unit slope.
if abs(x)>1
% elseif x<-delta 
y=sign(x);
else 
y=kk*x;
end