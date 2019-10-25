function [ coeff1 ] = fitPolynom( xs,ys ,deg)
%fits polynomial y=a*x^deg
temp=xs.^deg;
coeff1=temp*ys/(dot(temp,temp));
end

