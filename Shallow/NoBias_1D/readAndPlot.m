close all
clear
maxk=30;
repeatt=30;
iters=zeros(maxk,repeatt);

for i=1:900
    load(sprintf('myResults/%d.mat',i));
    iters(k,i_iter+1)=timeRun;
end
%%
figure('DefaultAxesFontSize',13);
plot([2:1:30],iters([2:1:30],1),'b*');hold on;
for i=1:30
    plot([2:1:30],iters([2:1:30],i),'b*');
end
xlabel('k');
ylabel('iters');
set(findall(gca, 'Type', 'Line'),'LineWidth',1.5);
%%
averaging=@(x) mean(x,2);
maxk=30;
mink=6;
%%
maxk=30;
mink=8;
deg=4;

meds=averaging(iters(1:maxk,:));
meds=meds(mink:2:end);
figure('DefaultAxesFontSize',13);
plot(log([mink:2:maxk]),log(meds),'*');hold on;
p = polyfit(log([mink:2:maxk])',log(meds),1);
y1 = polyval(p,log([mink:2:maxk]));
plot(log([mink:2:maxk]),y1)
xlabel('log(k)');
ylabel('log(iters)');
legend('measurements','line fit');
p(1)
set(findall(gca, 'Type', 'Line'),'LineWidth',2);

%%
maxk=30;
mink=6;
fig=figure('DefaultAxesFontSize',35);


ks=1:maxk;
keven=2:2:maxk;
kodds=2+1:2:maxk;


messure=averaging(iters(1:maxk,:));
stds=std(iters(1:maxk,:),[],2);
messureEven=messure(keven);


%THEORETIC
[fitEven,coefit] = fitEvenFormula(keven,messureEven,mink);
theoreticalFit=zeros(maxk,1);
theoreticalFit(keven)=fitEven;
theoreticalFit(kodds)=1800;
theoreticalFit(1)=coefit/(0.01*pi/4);

%QUADRATIC
quadFitxs=[mink:0.1:maxk];
p = polyfit(keven',messureEven,2);
quadFit=polyval(p,quadFitxs);



plot(ks,theoreticalFit(ks),'o','Color',[0.9290, 0.6940, 0.1250],	'MarkerSize',8,'LineWidth',3	);hold on;
plot(quadFitxs,quadFit,'-','LineWidth',2);
plot(ks([1 2:2:maxk]),messure(ks([1 2:2:maxk])),'.','MarkerSize',25,'Color',[0, 0.1070, 0.6410]);
errorbar(ks,messure(ks),stds(ks),'.','MarkerSize',10,'LineWidth',2,'CapSize',6,'Color',[0, 0.1070, 0.6410]);hold on;
plot(ks([3:2:maxk]),messure(ks([3:2:maxk])),'o','MarkerSize',6,'LineWidth',2,'Color',[0, 0.1070, 0.6410]);

ylim([0 1800])
xlabel('$k$','Interpreter','latex');
ylabel('Iterations');
legend({'Theoretical fit','Quadratic fit','Measurements'},'Location','northwest');
set(gcf, 'Position', [100, 100, 650, 550])
temp=get(gca, 'YTick');
set(gca, 'YTickLabel', arrayfun(@(x) sprintf('%0.1fk',x/1000),temp,'UniformOutput',false))
