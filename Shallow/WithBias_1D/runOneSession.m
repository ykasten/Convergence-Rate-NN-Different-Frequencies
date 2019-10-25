function  runOneSession( idd )
num=str2num(idd);
rng(num);

numIters=30;
k = floor((num - 1)/numIters + 1);
i_iter = mod(num - 1, numIters);
k
i_iter
[timeRun,m,n,std_chosen,LR]=optimize(k);
save(sprintf('results/%s',idd),'timeRun','i_iter','k','m','n','std_chosen','LR');
end

