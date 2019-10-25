function [ ys ,passedinds] = forwardAr( w,a,xs )

temp=xs'*w;

passedinds=temp>=0;
temp(temp<0)=0;

m=size(w,2);

ys=(temp*a)/m^0.5;

end

