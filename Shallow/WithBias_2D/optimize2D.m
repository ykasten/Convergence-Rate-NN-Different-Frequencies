function [iters,m,n,stdval]=optimize3D(k)

m=16000;
n=1001;
stdval=1.0;
w=randn(3,m)*stdval;
b=zeros(m,1);

xs=randn(3,n);
xs=xs./repmat(sum(xs.^2,1).^0.5,3,1);

theta=acos(xs(3,:));
ys=sqrt((2*k+1)/(4*pi)*factorial(k-abs(0))/factorial(k+abs(0)))*getPnm( cos(theta),k,0 );   
ys=ys';

ys=ys/norm(ys);
a=randn(m,1);
a(a>0)=1;
a(a<=0)=-1;


iters=100000;
LR=0.01;

[ losses ] = trainAr( iters,w,a,b ,xs,ys,LR);
iters=length(losses);

end


% figure, plot(losses)