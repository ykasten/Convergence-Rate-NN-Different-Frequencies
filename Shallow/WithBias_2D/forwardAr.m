function [ ys ,passedinds] = forwardAr( w,a,b,xs )

n=size(xs,2);
temp=xs'*w+repmat(b',n,1);

passedinds=temp>=0;
temp(temp<0)=0;

m=size(w,2);

ys=(temp*a)/m^0.5;
end

