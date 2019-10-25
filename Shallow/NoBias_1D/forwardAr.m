function [ ys ,passedinds] = forwardAr( w,a,xs )
%n samples , m channels
%x \in R^{2xn}
%w \in R^{2xm}
%a \in R^{mx1}

temp=xs'*w;
%temp is nxm

passedinds=temp>=0;
temp(temp<0)=0;

m=size(w,2);
%now temp is nxm
ys=(temp*a)/m^0.5;
%ys is nx1



end

