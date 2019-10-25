function [ losses ] = trainAr( iters,w,a,b ,xs,ys,LR)
losses=zeros(iters,1);

show=true;
if show
    figure;
end


for i=1:iters
    
    [ dl_dw,dl_db ] = backwardAr(  w,a,b,xs ,ys );
    w=w-LR*dl_dw;
    b=b-LR*dl_db';
    
    losses(i)=lossAr(  w,a,b,xs ,ys );
    [ yscur ,passedinds] = forwardAr( w,a,b,xs );
    if mod(i,5)==3
        i
        projectioneig=(dot(yscur,ys)/(norm(yscur)*norm(ys)))
        curLoss=losses(i)
        if show
            hold off
             plot(acos(xs(3,:)),ys,'*');
            hold on;
            plot(acos(xs(3,:)),yscur,'*');
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

