function [ output_args ] = lossAr(  w,a,xs ,shouldys )
%LOSS Summary of this function goes here
%   Detailed explanation goes here

[ ys ] = forwardAr( w,a,xs );
output_args=0.5*sum((shouldys-ys).^2);

end

