function  runOneSession( idd )
num=str2num(idd);
rng(num);


numIters=20;
k = floor((num - 1)/numIters + 1);
i_iter = mod(num - 1, numIters);

[timeRun,m,n,stdval]=optimize2D(k);
save(sprintf('results/%s',idd),'timeRun','i_iter','k','m','n','stdval');

end

