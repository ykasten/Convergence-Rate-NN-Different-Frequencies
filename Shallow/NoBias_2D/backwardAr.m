function [ dl_dw ] = backwardAr(  w,a,xs ,shouldys )
%n samples , m channels

%passedinds is nxm
m=size(w,2);
n=size(xs,2);
[ ys ,passedinds] = forwardAr( w,a,xs );
%ys is mx1
tempx=sum(repmat((ys-shouldys),1,m).*passedinds.*xs(1,:)'/sqrt(m));
tempy=sum(repmat((ys-shouldys),1,m).*passedinds.*xs(2,:)'/sqrt(m));
tempz=sum(repmat((ys-shouldys),1,m).*passedinds.*xs(3,:)'/sqrt(m));
dl_dw= [tempx.*a.';tempy.*a.';tempz.*a.'];

end

