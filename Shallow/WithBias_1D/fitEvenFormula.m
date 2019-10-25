function [yfit] = fitEvenFormula(k,meds,mink)


%FITEVENFORMULA Summary of this function goes here
%   Detailed explanation goes here
ev=((k.^2+1).*(1))./(pi*(k.^2-1).^2);
temp=1./(0.01*ev);

coef=temp(k>=mink)*meds(k>=mink)/(dot(temp(k>=mink),temp(k>=mink)));
yfit=coef*temp;
coef
end

