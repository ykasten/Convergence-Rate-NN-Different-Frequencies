function [ output_args ] = getPnm( z,n,m )
output_args=(1-z.^2).^(m/2)/((2^n)*factorial(n));

%Derrivative of (x^2-1)^n, n+m times
syms x
fun=@(x) (x^2-1)^n;
for i=1:(n+m)
fun = matlabFunction( diff(fun(x) ));
end
output_args=output_args.*fun(z);


end

