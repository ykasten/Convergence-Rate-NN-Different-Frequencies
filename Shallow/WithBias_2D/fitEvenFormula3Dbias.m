function [yfit,coef] = fitEvenFormula3Dbias(k,means,mink)


addpath('forFormula');
ev=zeros(length(k),1)';
for i=1:length(k)
    ev(i)=abs(getABias( k(i),2 ));
end


temp=1./(0.01*ev);
tofit=temp(k>=mink);
coef=tofit*means(k>=mink)/(dot(tofit,tofit));

yfit=coef*temp;
coef
end

