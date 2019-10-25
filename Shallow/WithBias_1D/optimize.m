function [iters,m,n,std_chosen,LR]=optimize(k)

m=4000;
n=1001;
std_chosen=2.5;
w=randn(2,m)*std_chosen;
theta=rand(1,n)*2*pi-pi;
xs=[cos(theta);sin(theta)];

ys=[sin(k*theta)].';
 ys=ys/norm(ys);
b=zeros(m,1);
 a=randn(m,1);
a(a>0)=1;
a(a<=0)=-1;

%x \in R^{2xn}
%w \in R^{2xm}
%a \in R^{mx1}

iters=50000;
LR=0.01;

[ losses ] = trainAr( iters,w,a,b,xs,ys,LR);
iters=length(losses);
iters
end

