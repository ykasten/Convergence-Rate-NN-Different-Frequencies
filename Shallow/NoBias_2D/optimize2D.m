function [iters,m,n,stdChosen,LR]=optimize3D(k)
m=16000;
n=1001;
stdChosen=1;
w=randn(3,m)*stdChosen;

xs=randn(3,n);
xs=xs./repmat(sum(xs.^2,1).^0.5,3,1);
theta=acos(xs(3,:));

ys=sqrt((2*k+1)/(4*pi)*factorial(k-abs(0))/factorial(k+abs(0)))*getPnm( cos(theta),k,0 );
ys=ys';

ys=ys/norm(ys);
a=randn(m,1);
a(a>0)=1;
a(a<=0)=-1;

iters=40000;
LR=0.01;

[ losses ] = trainAr( iters,w,a ,xs,ys,LR);
iters=length(losses);
iters
end


