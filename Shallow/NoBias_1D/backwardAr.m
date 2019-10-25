function [ dl_dw ] = backwardAr(  w,a,xs ,shouldys )
%BACKWARDAR Summary of this function goes here
%   Detailed explanation goes here


%n samples , m channels
%x \in R^{2xn}
%w \in R^{2xm}
%a \in R^{mx1}
%passedinds is nxm


m=size(w,2);
n=size(xs,2);
[ ys ,passedinds] = forwardAr( w,a,xs );

%ys is nx1
tempx=sum(repmat((ys-shouldys),1,m).*passedinds.*xs(1,:)'/sqrt(m));
tempy=sum(repmat((ys-shouldys),1,m).*passedinds.*xs(2,:)'/sqrt(m));
dl_dw= [tempx.*a.';tempy.*a.'];

end

