function [yfit,coef] = fitEvenFormula3D(k,means,mink)

addpath('forFormula');
ev=zeros(length(k),1)';
for i=1:length(k)
    ev(i)=abs(getANew( k(i),2 ));
end

temp=1./(0.01*ev);

tofit=temp(k>=mink);
coef=tofit*means(k>=mink)/(dot(tofit,tofit));
coef
yfit=coef*temp;
end

