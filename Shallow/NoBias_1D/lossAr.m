function [ output_args ] = lossAr(  w,a,xs ,shouldys )


[ ys ] = forwardAr( w,a,xs );
output_args=0.5*sum((shouldys-ys).^2);

end

