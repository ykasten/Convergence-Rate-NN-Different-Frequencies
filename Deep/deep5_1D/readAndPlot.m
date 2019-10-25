clear
close all
clear
maxk=30;
repeatt=20;
iters=zeros(maxk,repeatt)./zeros(maxk,repeatt);

for i=1:(repeatt*maxk)
    try
        fileID = fopen(sprintf('myResults/test_deep5_euc_%d.json',i),'r');
        A = fscanf(fileID,'%s');
        
        fclose(fileID);
        value = jsondecode(A);
        k=value.ks.x0;
        i_iter=value.iter.x0;
        timeRun=length(value.train_loss.x0);
        iters(k,i_iter+1)=timeRun;
    catch
    end
end
%%
figure('DefaultAxesFontSize',13);
plot([2:1:30],iters([2:1:30],1),'b*');hold on;
for i=1:repeatt
    plot([2:1:30],iters([2:1:30],i),'b*');
end
xlabel('k');
ylabel('iters');
set(findall(gca, 'Type', 'Line'),'LineWidth',1.5);
%%
averaging=@(x) nanmean(x,2);
maxk=30;
mink=8;

meds=averaging(iters(1:maxk,:));
meds=meds(mink:1:end);
figure('DefaultAxesFontSize',13);
plot(log([mink:1:maxk]),log(meds),'*');hold on;
p = polyfit(log([mink:1:maxk])',log(meds),1);
y1 = polyval(p,log([mink:1:maxk]));
plot(log([mink:1:maxk]),y1)
xlabel('log(k)');
ylabel('log(iters)');
legend('measurements','line fit');
p(1)
set(findall(gca, 'Type', 'Line'),'LineWidth',2);


%%
maxk=30;
mink=1;

fig=figure('DefaultAxesFontSize',35);
ks=mink:maxk;

messure=averaging(iters(1:maxk,:));
stds=nanstd(iters(1:maxk,:),[],2);
messures=messure(ks);


%QUADRATIC
quadFitxs=[mink:0.1:maxk];
p = polyfit(ks',messures,2);
quadFit=polyval(p,quadFitxs);


plot(quadFitxs,quadFit,'Color',[0.8500, 0.3250, 0.0980],'LineWidth',2); hold on;
plot(ks,messure(ks),'.','MarkerSize',25,'Color',[0, 0.1070, 0.6410]);
errorbar(ks,messure(ks),stds(ks),'.','MarkerSize',10,'LineWidth',2,'CapSize',6,'Color',[0, 0.1070, 0.6410]);hold on;

plot(ks,messure(ks),'.','MarkerSize',25,'Color',[0, 0.1070, 0.6410]);
ylim([0,40000])
xlabel('$k$','Interpreter','latex');
ylabel('Iterations');
legend({'Quadratic Fit','Measurements'},'Location','northwest');
set(gcf, 'Position', [100, 100, 650, 550])
temp=get(gca, 'YTick');
set(gca, 'YTickLabel', arrayfun(@(x) sprintf('%dk',x/1000),temp,'UniformOutput',false))

