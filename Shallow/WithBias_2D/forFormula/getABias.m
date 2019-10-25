function [ val ] = getABias( k,d )
%GETANEW Summary of this function goes here
%   Detailed explanation goes here

[ p ] = getP( k,d );
sosum=0;
for q=ceil(k/2):1:p
if mod(k,2)==0 %even
    sosum=sosum+getC2(q,d,k)*((-1/(2*(2*q-k+1)))+1/(2*(2*q-k+2))*(1-(1/(2^(2*q-k+2)))*nchoosek(2*q-k+2,0.5*(2*q-k+2))));
else
     sosum=sosum+getC2(q,d,k)*( (1/(2*(2*q-k+1)))*(1-(1/(2^(2*q-k+1)))*nchoosek(2*q-k+1,0.5*(2*q-k+1))));
end
end

val=0.5*sosum*getC1(d,k);
end

