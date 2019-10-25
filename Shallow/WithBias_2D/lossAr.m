function [ output_args ] = lossAr(  w,a,b,xs ,shouldys )

[ ys ] = forwardAr( w,a,b,xs );
output_args=0.5*sum((shouldys-ys).^2);

end

