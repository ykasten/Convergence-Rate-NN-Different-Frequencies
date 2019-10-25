function [ output_args ] = getK1Bias( d )
%GETK1BIAS Summary of this function goes here
%   Detailed explanation goes here
output_args=0;
k=1;
q=1;
output_args=0.5*getC1(2,1)*getC2(1,d,k)*(1/(2*(2*q-k+2))+1/(2*(2*q-k+1))*(1-1/(2^(2*q-k+1))*nchoosek(2*q-k+1,0.5*(2*q-k+1))));

end

