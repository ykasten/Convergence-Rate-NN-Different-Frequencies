function  runOneSession( idd )
%RUNONESESSION Summary of this function goes here
%   Detailed explanation goes here
num=str2num(idd);
rng(num);


numIters=30;
k = floor((num - 1)/numIters + 1);
i_iter = mod(num - 1, numIters);
k
i_iter
[timeRun,m,n,stdval,LR]=optimize(k);
save(sprintf('results/%s',idd),'timeRun','i_iter','k','m','n','stdval','LR');


end

