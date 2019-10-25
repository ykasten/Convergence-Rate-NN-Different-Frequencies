function [ output_args ] = lossAr(  w,a,b,xs ,shouldys )
%LOSS Summary of this function goes here
%   Detailed explanation goes here

[ ys ] = forwardAr( w,a,b,xs );
output_args=0.5*sum((shouldys-ys).^2);

end

