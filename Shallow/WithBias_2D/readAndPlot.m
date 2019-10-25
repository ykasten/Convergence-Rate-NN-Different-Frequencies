close all
clear
maxk=30;
repeatt=20
addpath('forFormula');
iters=zeros(maxk,repeatt)./zeros(maxk,repeatt);

for i=1:maxk*repeatt
    
    try
        load(sprintf('myResults/%d.mat',i));
        iters(k,i_iter+1)=timeRun;
    catch
        i
    end
 end
%%
figure('DefaultAxesFontSize',13);
plot([2:1:maxk],iters([2:1:maxk],1),'b*');hold on;
for i=1:repeatt
    plot([2:1:maxk],iters([2:1:maxk],i),'b*');
end
xlabel('k');
ylabel('iters');
set(findall(gca, 'Type', 'Line'),'LineWidth',1.5);
%%
averaging=@(x) nanmean(x,2);
maxk=30;
mink=8;
%%

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
fig=figure('DefaultAxesFontSize',35);

ks=1:maxk;

keven=2:2:maxk;
kodds=3:2:maxk;


messure=averaging(iters(1:maxk,:));
stds=nanstd(iters(1:maxk,:),[],2);
messureEven=messure(keven);
meassureOdds=messure(kodds);

[fitEven,coefit] = fitEvenFormula3Dbias(keven,messureEven,mink);
[fitOdds,coefit] = fitEvenFormula3Dbias(kodds,meassureOdds,mink);

theoreticalFit=zeros(maxk,1);
theoreticalFit(keven)=fitEven;
theoreticalFit(kodds)=fitOdds;
theoreticalFit(1)=coefit/(0.01*getK1Bias(2));

%Cubic
cubicFitxs=[mink:0.1:maxk];
p = polyfit(keven',messureEven,3);
cubicFit=polyval(p,cubicFitxs);



plot(ks,theoreticalFit(ks),'o','Color',[0.9290, 0.6940, 0.1250],'MarkerSize',8 	,'LineWidth',3);hold on;
plot(cubicFitxs,cubicFit,'-','LineWidth',2);
plot(ks,messure(ks),'.','MarkerSize',25,'Color',[0, 0.1070, 0.6410]);
errorbar(ks,messure(ks),stds(ks),'.','MarkerSize',10,'LineWidth',2,'CapSize',6,'Color',[0, 0.1070, 0.6410]);hold on;


xlabel('$k$','Interpreter','latex');
ylabel('Iterations');
legend({'Theoretical fit','Cubic fit','Measurements'},'Location','northwest');
set(gcf, 'Position', [100, 100, 600, 550])
