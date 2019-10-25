function [iters,m,n,stdval,LR]=optimize(k)

m=4000;
n=1001;
stdval=1;
w=randn(2,m)*stdval;
theta=rand(1,n)*2*pi-pi;
xs=[cos(theta);sin(theta)];

ys=[sin(k*theta)].';
ys=ys/norm(ys);
a=randn(m,1);
a(a>0)=1;
a(a<=0)=-1;

iters=1800;
LR=0.01;

[ losses ] = trainAr( iters,w,a ,xs,ys,LR);
iters=length(losses);

end

