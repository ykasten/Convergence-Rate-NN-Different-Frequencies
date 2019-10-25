function [ losses ] = trainAr( iters,w,a ,xs,ys,LR)
losses=zeros(iters,1);

show=true;
if show
    figure;
end
for i=1:iters
    [ dl_dw ] = backwardAr(  w,a,xs ,ys );
    w=w-LR*dl_dw;
    losses(i)=lossAr(  w,a,xs ,ys );
    [ yscur ,passedinds] = forwardAr( w,a,xs );
    if mod(i,5)==3
        curIt=i
        curProj=(dot(yscur,ys)/(norm(yscur)*norm(ys)))
        curLoss=losses(i)
        if show
            hold off
            plot(atan2(xs(2,:),xs(1,:)),ys,'*');
            hold on;
            plot(atan2(xs(2,:),xs(1,:)),yscur,'*');
            drawnow
        end
    end
    converged=losses(i)<0.05;
    if(converged)
        break;
    end
end

losses=losses(1:i);
end

