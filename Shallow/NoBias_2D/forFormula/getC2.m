function [ val ] = getC2( q,d,k )
%GETC2 Summary of this function goes here
%   Detailed explanation goes here
p=getP( k,d );
val=(-1)^q*nchoosek(p,q)*factorial(2*q)/factorial(2*q-k);

end

