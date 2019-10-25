function [ dl_dw ,dl_db] = backwardAr(  w,a,b,xs ,shouldys )
%n samples , m channels

%x \in R^{2xn}
%w \in R^{2xm}
%a \in R^{mx1}

m=size(w,2);
n=size(xs,2);
[ ys ,passedinds] = forwardAr( w,a,b,xs );
%passedinds is nxm
%ys is mx1
tempx=sum(repmat((ys-shouldys),1,m).*passedinds.*xs(1,:)'/sqrt(m));
tempy=sum(repmat((ys-shouldys),1,m).*passedinds.*xs(2,:)'/sqrt(m));
tempz=sum(repmat((ys-shouldys),1,m).*passedinds.*xs(3,:)'/sqrt(m));
dl_dw= [tempx.*a.';tempy.*a.';tempz.*a.'];
tempb=sum(repmat((ys-shouldys),1,m).*passedinds.*ones(n,1)/sqrt(m));
dl_db=tempb.*a.';
end

