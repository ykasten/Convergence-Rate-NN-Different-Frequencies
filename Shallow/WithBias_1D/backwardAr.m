function [ dl_dw,dl_db ] = backwardAr(  w,a,b,xs ,shouldys )
%n samples , m channels

%x \in R^{2xn}
%w \in R^{2xm}
%a \in R^{mx1}
%passedinds is nxm
m=size(w,2);
n=size(xs,2);
[ ys ,passedinds] = forwardAr( w,a,b,xs );
%ys is mx1

 
tempx=sum(repmat((ys-shouldys),1,m).*passedinds.*xs(1,:)'/sqrt(m));
tempy=sum(repmat((ys-shouldys),1,m).*passedinds.*xs(2,:)'/sqrt(m));
dl_dw= [tempx.*a.';tempy.*a.'];

tempb=sum(repmat((ys-shouldys),1,m).*passedinds.*ones(1,n)'/sqrt(m));
dl_db=[tempb.*a.'];


end

