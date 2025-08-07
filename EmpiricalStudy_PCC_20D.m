%% Script to generate estimates, plots and tables for empirical copula study 
%
% This script was used in the following article:
% Gubbels, K.B., Ypma, J.Y. & Oosterlee, C.W. (2025),
% Principal Component Copulas for Capital Modelling and Systemic Risk, Computational Economics 
% https://doi.org/10.1007/s10614-025-11051-7   
%
% Data based on devolatized weekly logreturns of 20 major stock indices from Yahoo finance
%
% This script generates the parameter estimates in the tables and the figures 
% for 20-dimensional case study of global stock indices
%
% Script takes less than a minute to run

% Initialize
close all;
clear;
rng(1);
addpath('./functions');

%% Get data

% Load devolatized weekly logreturn data (order of indices is same as in article)
load('dataLogReturns');
[nData, nDim] = size(xsHist);

% Determine pseudo-copula observations
usHist = tiedrank(xsHist)/(nData + 1);

% Determine gaussian rank correlations and perform initial PCA 
nsHist = icdf('norm', usHist, 0, 1);
[evN, ewN] = pcacov(corr(nsHist));

% Initial PC observations
psHistIni = nsHist * evN;

%% Fit PCC HB-N copula with GMM algorithm 

% Initialize
pars0  = [5, 5];
parsLb = [0.25, 0.25];
parsUb = [100, 100];

% Perform initial ML for shape parameters
optMl    = optimoptions('fmincon', 'MaxIterations', 50, 'StepTolerance',0.1); 
fOpt = @(x) -fLogLhPccHbN(x, evN, ewN, usHist);
parsIni  = fmincon(fOpt,pars0,[],[],[],[],parsLb,parsUb,[],optMl);

% Record results from initializing estimation step
disp(parsIni)
parsRec = parsIni;
evRec   = evN;
ewRec   = ewN;

% Perform n recursions
for iRec = 1:5
   % Update densities and correlations
   [~,ysRec,~] = fLogLhPccHbN(parsRec, evRec, ewRec, usHist);
   [evRec, ewRec] = pcacov(corr(ysRec));

   % Perform maximum likelihood (with previous step as start)
   fOpt  = @(x) -fLogLhPccHbN(x, evRec, ewRec, usHist);
   optMl     = optimoptions('fmincon', 'MaxIterations', 20, 'StepTolerance',0.1);
   parsRec   = fmincon(fOpt, parsRec,[],[],[],[],parsLb,parsUb,[],optMl);
end

% Final risk factor and PC observations 
ysHist = ysRec;
psHist = ysRec * evRec;

% PCC
loglPccHbN = fLogLhPccHbN(parsRec, evRec, ewRec, usHist);

% PCC
evPccHbN   = evRec;
ewPccHbN   = ewRec;

%% Simulate PCC HB-N copula

% Hyperbolic distribution for P1
alpha   = (parsRec(1) + parsRec(2))/2;
beta    = (parsRec(1) - parsRec(2))/2;
gammaHB = sqrt(alpha^2 - beta^2);

% Parameter mu leading to mean zero when delta equals one
mu    = -beta/gammaHB*besselk(2,gammaHB)/besselk(1,gammaHB);

% Variance of HB distribution when delta equals one
varHB = 1/gammaHB*besselk(2,gammaHB)/besselk(1,gammaHB)+(beta^2/gammaHB^2)*(besselk(3,gammaHB)/besselk(1,gammaHB)-besselk(2,gammaHB)^2/besselk(1,gammaHB)^2);

% Scale parameters (scaling step explained in supplementary note)
alphaP1 = alpha*sqrt(varHB)/sqrt(ewRec(1));
betaP1  = beta*sqrt(varHB)/sqrt(ewRec(1));
gammaP1 = sqrt(alphaP1^2 - betaP1^2);
deltaP1 = sqrt(ewRec(1))/sqrt(varHB);
muP1    = -deltaP1*betaP1/gammaP1*besselk(2,deltaP1*gammaP1)/besselk(1,deltaP1*gammaP1);

% Characteristic functions
cfGH = @(t) exp(1i*mu*t).*gammaHB.*besselk(1,sqrt(alpha^2 - (beta+1i*t).^2)) ...
   ./(sqrt(alpha^2 - (beta+1i*t).^2).* besselk(1,gammaHB));
cfN  = @(t) exp(-t.^2/2);       

% Simulate PCs
nSims = 10^6;
cfP1  = @(t) cfGH(t/sqrt(varHB));
p1    = cf2QdCos(cfP1, [], rand(nSims,1));
psSim(:,1) = sqrt(ewRec(1))*p1.qYs;
   
% Simulate other PCs
for iDim = 2:nDim
   psSim(:,iDim) = icdf('normal', rand(nSims,1), 0, sqrt(ewRec(iDim)));
end
   
% Determine simulated ys
ysSim = psSim * evRec';

% Determine simulated us
usSim = zeros(size(ysSim));
for iVar = 1:nDim
   sigmaP1 = sqrt(ewRec(1))*evRec(iVar,1); 
   sigmaN  = sqrt(evRec(iVar,2:end)*diag(ewRec(2:end))*evRec(iVar,2:end)'); 
   cfY = @(t) cfGH(sigmaP1*t/sqrt(varHB)) .* cfN(sigmaN*t);
   pdY = cf2QdCos(cfY, ysSim(:,iVar), []);
   usSim(:,iVar) = pdY.cYs;
end

% Gather simulated us 
usPccHbN = usSim;

%% Visualize HB-N PCC

% Scatter plots
nPlot = 2500;
iVar1 = 2;  % Dax
iVar2 = 15; % S&P500
xsSim1 = ksdensity(xsHist(:,iVar1),usSim(1:nPlot,iVar1),'function','icdf','Bandwidth',0.1); 
xsSim2 = ksdensity(xsHist(:,iVar2),usSim(1:nPlot,iVar2),'function','icdf','Bandwidth',0.1); 

pos4 = {[-0.05 0.21 0.4 0.7], [0.2 0.21 0.4 0.7], [0.45 0.21 0.4 0.7], [0.69 0.21 0.4 0.7]};
figure;
set(gcf, 'Position',  [200, 150, 1150, 300]);  

pos = pos4{1};
axes('Position', pos);
hold on;
box on;
scatter(xsSim1, xsSim2,3,"o","r");
box on;
hold on;
scatter(xsHist(:,iVar1), xsHist(:,iVar2),8,"o","b");
set(gca,'fontsize',13)
xlabel('{\it x}_{2,t} (FTSE)', 'FontName', 'Times', 'FontSize', 16)
ylabel('{\it x}_{15,t} (S&P)', 'FontName', 'Times', 'FontSize', 16)
pbaspect([1 1 1])

pos = pos4{2};
axes('Position', pos);
hold on;
box on;
scatter(usSim(1:nPlot,iVar1), usSim(1:nPlot,iVar2),3,"o","r");
box on;
hold on;
scatter(usHist(:,iVar1), usHist(:,iVar2),8,"o","b");
set(gca,'fontsize',13)
xlabel('{\it u}_{2,t}  (FTSE)', 'FontName', 'Times', 'FontSize', 16)
ylabel('{\it u}_{15,t}  (S&P)', 'FontName', 'Times', 'FontSize', 16)
pbaspect([1 1 1])

pos = pos4{3};
axes('Position', pos);
hold on;
box on;
scatter(ysSim(1:nPlot,iVar1), ysSim(1:nPlot,iVar2),3,"o","r");
box on;
hold on;
scatter(ysHist(:,iVar1), ysHist(:,iVar2),8,"o","b");
set(gca,'fontsize',13)
xlabel('{\it y}_{2,t}  (FTSE)', 'FontName', 'Times', 'FontSize', 16)
ylabel('{\it y}_{15,t}  (S&P)', 'FontName', 'Times', 'FontSize', 16)
pbaspect([1 1 1])

pos = pos4{4};
iPc1 = 1;
iPc2 = 2;
axes('Position', pos);
hold on;
box on;
scatter(psSim(1:nPlot,iPc1), psSim(1:nPlot,iPc2),3,"o","r");
box on;
hold on;
scatter(psHist(:,iPc1), psHist(:,iPc2),8,"o","b");
set(gca,'fontsize',13)
xlabel('{\it p}_{1,t}', 'FontName', 'Times', 'FontSize', 16)
ylabel('{\it p}_{2,t}', 'FontName', 'Times', 'FontSize', 16)
pbaspect([1 1 1])

%% Fit Skew t_1 - t_{d-1} copula

% Initialize
pars0  = [10, -1];
parsLb = [5, -3];
parsUb = [50, -0.1];

% Perform initial ML for shape parameters
optMl   = optimoptions('fmincon', 'MaxIterations', 25); 
fOpt    = @(x) -fLogLhPccSkewtMvt(x, evN, ewN, usHist);
parsIni = fmincon(fOpt,pars0,[],[],[],[],parsLb,parsUb,[],optMl);

% Record results from initializing estimation step
parsRec = parsIni;
evRec   = evN;
ewRec   = ewN;

% Perform n recursions
for iRec = 1:5
   % Update densities and correlations
   [~,ysRec,~]    = fLogLhPccSkewtMvt(parsRec, evRec, ewRec, usHist);
   [evRec, ewRec] = pcacov(corr(ysRec));

   % Perform maximum likelihood (with previous step as start)
   fOpt      = @(x) -fLogLhPccSkewtMvt(x, evRec, ewRec, usHist);
   optMl     = optimoptions('fmincon', 'MaxIterations', 20);
   parsRec   = fmincon(fOpt, parsRec,[],[],[],[],parsLb,parsUb,[],optMl);
end

% Gather implicit copula risk factor and PC observations 
ysHist = ysRec;
psHist = ysRec * evRec;

% Gather iterative estimation results
parsPccSkewtMvt = parsRec;
evPccSkewtMvt   = evRec;
ewPccSkewtMvt   = ewRec;

% Gather loglikelihood
loglPccSkewtMvt = fLogLhPccSkewtMvt(parsPccSkewtMvt, evRec, ewRec, usHist);

%% Simulate Skew t1 - t_{d-1} copula

% Get parameters
nuT    = parsPccSkewtMvt(1);
gammaT = parsPccSkewtMvt(2);

% Determine additional parameters
beta   = gammaT;
delta  = sqrt(nuT);
varSkewT = 2*beta^2*delta^4/((nuT-2)^2*(nuT-4)) + delta^2/(nuT-2);          
nuTp   = nuT/2;
varTp  = nuTp/(nuTp-2);

% Simulate PCs (with unit variance)
mu1  = -gammaT*nuT /(nuT-2);
mus  = zeros(1,nDim);
rhos = eye(nDim);
ws1  = 1./gamrnd(nuT/2,2/nuT, nSims,1);
ws2  = 1./gamrnd(nuTp/2,2/nuTp, nSims,1);
zSims   = mvnrnd(mus, rhos, nSims); 
ps(:,1) = (mu1 + gammaT*ws1 + sqrt(ws1).*zSims(:,1))/sqrt(varSkewT); % First PC: skew t
ps(:,2:nDim) = sqrt(ws2).*zSims(:,2:end)/sqrt(varTp);                % Higher PCs: multivariate t

% Scale PCs to required variance
ps = ps .* sqrt(ewRec)';

% Determine ys
ys = ps * evRec';

% Determine us
us = tiedrank(ys)/(nSims+1);

% Gather simulated us 
usPccSkewtMvt = us;

%% Figure: eigenvector weights

% Show eigenvector weights
figure
hold on;
plot1 = plot(evPccHbN(:,1), '-x','LineWidth',0.1,'MarkerSize', 5);
plot2 = plot(evPccHbN(:,2), '-x','LineWidth',0.1,'MarkerSize', 5);
plot3 = plot(evPccHbN(:,3), '-x','LineWidth',0.1,'MarkerSize', 5);
yline(0)
pbaspect([3 1 1])
axis([1, 20, -0.5, 0.6])
xlabel('Variable index')
ylabel('Eigen vector weight')
legend([plot1,plot2,plot3],'PC1','PC2','PC3', 'Location','southeast','NumColumns',3);

%% Fit Gaussian and t copula

% Gauss copula
rhoEstG = copulafit('Gaussian',usHist);
cPdfG   = copulapdf('Gaussian',usHist,rhoEstG);
loglG   = sum(log(cPdfG));

% t copula
[rhoEstT, nuEst] = copulafit('t',usHist, 'Method', 'ApproximateML');
cPdfT = copulapdf('t',usHist,rhoEstT,nuEst);
loglT = sum(log(cPdfT));

% Simulate t and Gauss copula
usT = copularnd('t', rhoEstT, nuEst,nSims);
usG = copularnd('Gaussian', rhoEstG, nSims); 

%% Fit PCC Skew t1 - t1 copula

% Initialize
pars0  = [10, -1];
parsLb = [5, -3];
parsUb = [50, -0.1];

% Perform initial ML for shape parameters
optMl   = optimoptions('fmincon', 'MaxIterations', 50); 
fOpt    = @(x) -fLogLhPccSkewtT(x, evN, ewN, usHist);
parsIni = fmincon(fOpt,pars0,[],[],[],[],parsLb,parsUb,[],optMl);

% Record results from initializing estimation step
parsRec = parsIni;
evRec   = evN;
ewRec   = ewN;

% Perform n recursions
for iRec = 1:5
   % Update densities and correlations
   [~,ysRec,~]    = fLogLhPccSkewtT(parsRec, evRec, ewRec, usHist);
   [evRec, ewRec] = pcacov(corr(ysRec));

   % Perform maximum likelihood (with previous step as start)
   fOpt      = @(x) -fLogLhPccSkewtT(x, evRec, ewRec, usHist);
   optMl     = optimoptions('fmincon', 'MaxIterations', 20);
   parsRec   = fmincon(fOpt, parsRec,[],[],[],[],parsLb,parsUb,[],optMl);
end

% Gather iterative estimation results
parsPccSkewtT = parsRec;

% Gather loglikelihood
loglPccSkewtT = fLogLhPccSkewtT(parsPccSkewtT, evRec, ewRec, usHist);

%% Simulate PCC Skew t1-t1 copula

% Get fit paramaters
nuT    = parsRec(1);
gammaT = parsRec(2);
 
% Define additional parameters
alpha  = abs(gammaT);
beta   = gammaT;
lambda = -nuT/2;
delta  = sqrt(nuT);
mu     = -delta^2*beta/2*(gamma(-lambda-1)/gamma(-lambda));
varSkewT = 2*beta^2*delta^4/((nuT-2)^2*(nuT-4)) + delta^2/(nuT-2);        
nuTp  = nuT/2;
varTp = nuTp/(nuTp-2);

% Characteristic functions
cfSkewT = @(t) 2^(lambda+1)/gamma(nuT/2)*exp(1i*mu*t).*besselk(lambda,delta*sqrt(alpha^2 - (beta+1i*t).^2))./(delta*sqrt(alpha^2 - (beta+1i*t).^2)).^lambda;    
cfT     = @(t) besselk(nuTp/2,sqrt(nuTp)*abs(t)).*(sqrt(nuTp)*abs(t)).^(nuTp/2)./(gamma(nuTp/2)*2^(nuTp/2-1));
cfP1    = @(t) cfSkewT(t/sqrt(varSkewT));   
cfPj    = @(t) cfT(t/sqrt(varTp));

% Simulate pcs
p1    = cf2QdCos(cfP1, [], rand(nSims,1));
psSim(:,1) = sqrt(ewRec(1))*p1.qYs;
   
% Simulate other PCs
for iPc = 2:nDim   
   pj    = cf2QdCos(cfPj, [], rand(nSims,1));
   psSim(:,iPc) = sqrt(ewRec(iPc))*pj.qYs;
end
   
% Determine simulated ys
ysSim = psSim * evRec';

% Determine simulated us
usPccSkewtT = tiedrank(ysSim)/(nSims+1);

%% Fit Skew t_d copula

% Initialize
pars0  = [10, -1, ewN(1), ewN(2)];
parsLb = [5, -3, 5, 0.5];
parsUb = [50, -0.1, 15, 5];

% Estimate Skew T with ML
optMl     = optimoptions('fmincon', 'MaxIterations', 100); 
fOpt      = @(x) -fLogLhSkewT(x, evN, ewN, usHist);
parsSkewT = fmincon(fOpt,pars0,[],[],[],[],parsLb,parsUb,[],optMl);

% Gather loglikelihood
loglSkewT = fLogLhSkewT(parsSkewT, evN, ewN, usHist);

%% Simulate Skew t copula

% Get parameters
ev     = evN;
ew     = ewN;
ew(1)  = parsSkewT(3);
ew(2)  = parsSkewT(4);
ew     = ew*nDim/sum(ew); % Trace equals number of dimensions
rhos   = ev*diag(ew)*ev';
nuT    = parsSkewT(1);
gammaT = parsSkewT(2);

% Simulate skew t
mus   = zeros(1,nDim);
wSims = 1./gamrnd(nuT/2,2/nuT, nSims,1);
zSims = mvnrnd(mus, rhos, nSims);
ySims = gammaT*ev(:,1)'.*wSims + sqrt(wSims) .* zSims;
uSims = tiedrank(ySims)/(nSims+1);

% Simulate t and Gauss copula
usSkewT = uSims;

%% Show estimation results

% Correlations for t and Gaussian
disp('Degree of freedom for t:')
disp(nuEst)
% Dof for t copula
disp('Degree of freedom and gamma for skew t:')
disp([parsSkewT(1),parsSkewT(2)])
disp('Shape parameters for first PCC HB-N:')
disp([alphaP1,betaP1])
disp('Shape parameters for first PCC Skew t-t:')
disp([parsPccSkewtT(1)/2, parsPccSkewtT(2)]);
disp('Shape parameters for first PCC Skew t-mvt:')
disp([parsPccSkewtMvt(1)/2, parsPccSkewtMvt(2)]);
disp('Loglikelihoods:')
disp([loglG, loglT, loglSkewT, loglPccHbN, loglPccSkewtT, loglPccSkewtMvt]);
disp('Delta AIC:')
disp([loglG, 2+2*(loglG-loglT), 4+2*(loglG-loglSkewT), 4+2*(loglG-loglPccHbN), 4+2*(loglG-loglPccSkewtT), 4+2*(loglG-loglPccSkewtMvt)]);
disp('Delta BIC:')
disp([loglG, log(nData)+2*(loglG-loglT), 2*log(nData)+2*(loglG-loglSkewT), 2*log(nData)+2*(loglG-loglPccHbN), 2*log(nData)+2*(loglG-loglPccSkewtT), 2*log(nData)+2*(loglG-loglPccSkewtMvt)]);

%% Pairwize quantile exceedances

% Bootstrap historic observations for confidence intervals
nBoot = 1000;
usBoot = zeros(nBoot,nData,nDim);
for iBoot = 1:nBoot
   idsBoot = randsample(nData, nData,1);
   usBoot(iBoot,:,:)  = usHist(idsBoot,:);
end

% Initialize
qsP = [0.01:0.01:0.15, 0.175:0.025:0.5];
jqeHist = zeros(numel(qsP),1);
jqePccHbN   = zeros(numel(qsP),1);
jqePccSkewtT = zeros(numel(qsP),1);
jqePccSkewtMvt = zeros(numel(qsP),1); 
jqeSkewT    = zeros(numel(qsP),1); 
jqeT    = zeros(numel(qsP),1);
jqeG    = zeros(numel(qsP),1); 
jqeBoot = zeros(nBoot,numel(qsP)); 

% Determine pairwize CPJQE
for iQ = 1:numel(qsP)
   q  = qsP(iQ);
   jqeHist(iQ)= sum(sum(usHist(:,[iVar1,iVar2]) <= q,2)==2)/(q*nData);
   jqePccHbN(iQ) = sum(sum(usPccHbN(:,[iVar1,iVar2]) <= q,2)==2)/(q*nSims);
   jqePccSkewtT(iQ) = sum(sum(usPccSkewtT(:,[iVar1,iVar2]) <= q,2)==2)/(q*nSims);  
   jqePccSkewtMvt(iQ) = sum(sum(usPccSkewtMvt(:,[iVar1,iVar2]) <= q,2)==2)/(q*nSims);  
   jqeSkewT(iQ) = sum(sum(usSkewT(:,[iVar1,iVar2]) <= q,2)==2)/(q*nSims);  
   jqeT(iQ)   = sum(sum(usT(:,[iVar1,iVar2]) <= q,2)==2)/(q*nSims);
   jqeG(iQ)   = sum(sum(usG(:,[iVar1,iVar2]) <= q,2)==2)/(q*nSims);  
   
   for iBoot = 1:nBoot  
      jqeBoot(iBoot,iQ)= sum(sum(usBoot(iBoot,:,[iVar1,iVar2]) <= q,3)==2)/(q*nData);
   end   
end

%% Market Distress Factor as a function of quantile, fraction and dimension

% Initialize MDF as a function of quantile q
qs = 0.1:0.025:0.5;
mdfqHist     = zeros(numel(qs),1);
mdfqPccHbN   = zeros(numel(qs),1);
mdfqPccSkewtT = zeros(numel(qs),1); 
mdfqPccSkewtMvt = zeros(numel(qs),1); 
mdfqSkewT    = zeros(numel(qs),1); 
mdfqCombT    = zeros(numel(qs),1);
mdfqG    = zeros(numel(qs),1); 
mdfqBoot = zeros(nBoot,numel(qs)); 

% Loop over quantiles
for iQ = 1:numel(qs)
   q  = qs(iQ);
   mdfqHist(iQ)= sum(sum(usHist <= q,2)==nDim)/nData;
   mdfqPccHbN(iQ) = sum(sum(usPccHbN <= q,2)==nDim)/nSims;
   mdfqPccSkewtT(iQ) = sum(sum(usPccSkewtT <= q,2)==nDim)/nSims; 
   mdfqPccSkewtMvt(iQ) = sum(sum(usPccSkewtMvt <= q,2)==nDim)/nSims; 
   mdfqSkewT(iQ) = sum(sum(usSkewT <= q,2)==nDim)/nSims;  
   mdfqCombT(iQ) = sum(sum(usT <= q,2)==nDim)/nSims;
   mdfqG(iQ)   = sum(sum(usG <= q,2)==nDim)/nSims;  
   
   for iBoot = 1:nBoot  
      mdfqBoot(iBoot,iQ)= sum(sum(usBoot(iBoot,:,:) <= q,3)==nDim)/nData;
   end   
end

% Initialize MDF as a function of fraction k/d
q = 0.1;
ks = 0.5:0.05:1;
mdfkHist = zeros(numel(ks),1);
mdfkPccHbN   = zeros(numel(ks),1);
mdfkPccSkewtT = zeros(numel(ks),1); 
mdfkPccSkewtMvt = zeros(numel(ks),1); 
mdfkSkewT    = zeros(numel(ks),1); 
mdfkT    = zeros(numel(ks),1);
mdfkG    = zeros(numel(ks),1); 
mdfkBoot = zeros(nBoot,numel(ks)); 

% Loop over fraction k
for iK = 1:numel(ks)
   k  = ks(iK);
   mdfkHist(iK)= sum(sum(usHist <= q,2)>=k*nDim)/nData;
   mdfkPccHbN(iK) = sum(sum(usPccHbN <= q,2)>=k*nDim)/nSims;
   mdfkPccSkewtT(iK) = sum(sum(usPccSkewtT <= q,2)>=k*nDim)/nSims; 
   mdfkPccSkewtMvt(iK) = sum(sum(usPccSkewtMvt <= q,2)>=k*nDim)/nSims; 
   mdfkSkewT(iK) = sum(sum(usSkewT <= q,2)>=k*nDim)/nSims;  
   mdfkT(iK)   = sum(sum(usT <= q,2)>=k*nDim)/nSims;
   mdfkG(iK)   = sum(sum(usG <= q,2)>=k*nDim)/nSims;  
   
   for iBoot = 1:nBoot  
      mdfkBoot(iBoot,iK)= sum(sum(usBoot(iBoot,:,:) <= q,3)>=k*nDim)/nData;
   end   
end

% Initialize MDF as a function of dimension
q = 0.10;
ds = 2:nDim;
ids = randsample(nDim,nDim);
mdfdHist = zeros(numel(ds),1);
mdfdPccHbN   = zeros(numel(ds),1);
mdfdPccSkewtT = zeros(numel(ds),1); 
mdfdPccSkewtMvt = zeros(numel(ds),1); 
mdfdSkewT    = zeros(numel(ds),1); 
mdfdT    = zeros(numel(ds),1);
mdfdG    = zeros(numel(ds),1); 
mdfdBoot = zeros(nBoot,numel(ds)); 

% Loop over dimension
for iD = 1:numel(ds)
   d  = ds(iD);
   mdfdHist(iD)= sum(sum(usHist(:,ids(1:d)) <= q,2)==d)/nData;
   mdfdPccHbN(iD) = sum(sum(usPccHbN(:,ids(1:d)) <= q,2)==d)/nSims;
   mdfdPccSkewtT(iD) = sum(sum(usPccSkewtT(:,ids(1:d)) <= q,2)==d)/nSims; 
   mdfdPccSkewtMvt(iD) = sum(sum(usPccSkewtMvt(:,ids(1:d)) <= q,2)==d)/nSims; 
   mdfdSkewT(iD) = sum(sum(usSkewT(:,ids(1:d)) <= q,2)==d)/nSims;  
   mdfdT(iD)   = sum(sum(usT(:,ids(1:d)) <= q,2)==d)/nSims;
   mdfdG(iD)   = sum(sum(usG(:,ids(1:d)) <= q,2)==d)/nSims;  
   
   for iBoot = 1:nBoot  
      mdfdBoot(iBoot,iD)= sum(sum(usBoot(iBoot,:,ids(1:d)) <= q,3)==d)/nData;
   end   
end

%% Figure of Market Distress Cube

% Pairwize CPJQE figure
fontSize = 11;
pos4 = {[0.08 0.4 0.4 0.7], [0.57 0.4 0.4 0.7],[0.08 0.08 0.4 0.4], [0.57 0.08 0.4 0.4]};
figure;
set(gcf, 'Position',  [200, 175, 800, 575]);  
pos = pos4{1};
axes('Position', pos);
hold on;
box on;
plot(qsP, jqeHist, 'color', [0,0,0], 'linewidth', 1);
plot(qsP, jqePccHbN, 'color', [1,0,0], 'linewidth', 1);
plot(qsP, jqePccSkewtMvt, 'color', [0,0,1], 'linewidth', 1);
plot(qsP, jqePccSkewtT, 'color', [1,0,1], 'linewidth', 1);
plot(qsP, jqeSkewT, 'color', [0,1,0], 'linewidth', 1);
plot(qsP, jqeT, 'color', [1,1,0], 'linewidth', 1);
plot(qsP, jqeG, 'color', [0 1 1], 'linewidth', 1);
xCI =[qsP, flip(qsP)];
yCI = [prctile(jqeBoot,5), flip(prctile(jqeBoot,95))];
fill(xCI, yCI, 1,'facecolor', 'red', 'edgecolor', 'none', 'facealpha', 0.2);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',fontSize);
pbaspect([1.5 1 1]);
xlabel('quantile','fontsize',fontSize);
ylabel('CPJQE','fontsize',fontSize);
legend('History', 'PCC HB-N','PCC Skew t_1-t_d_-_1','PCC Skew t_1-t_1','Skew t','t', 'Gauss','90% CI','location','southeast');
%title('FTSE 100 versus S&P 500', 'fontsize', 11);
axis([0, 0.5, 0, 0.75])
xticks([0 0.25 0.5])
xticklabels({'0', '0.25', '0.5'})
yticks([0 0.25  0.5 0.75])
yticklabels({'0', '0.25', '0.5', '0.75'})
grid

% Market Distress Frequency figure as function of quantile
pos = pos4{2};
axes('Position', pos);
hold on;
box on;
plot(qs, mdfqHist, 'color', [0,0,0], 'linewidth', 1);
plot(qs, mdfqPccHbN, 'color', [1,0,0], 'linewidth', 1);
plot(qs, mdfqPccSkewtMvt, 'color', [0,0,1], 'linewidth', 1);
plot(qs, mdfqPccSkewtT, 'color', [1,0,1], 'linewidth', 1);
plot(qs, mdfqSkewT, 'color', [0,1,0], 'linewidth', 1);
plot(qs, mdfqCombT, 'color', [1,1,0], 'linewidth', 1);
plot(qs, mdfqG, 'color', [0 1 1], 'linewidth', 1);
xCI =[qs, flip(qs)];
yCI = [prctile(mdfqBoot,5), flip(prctile(mdfqBoot,95))];
fill(xCI, yCI, 1,'facecolor', 'red', 'edgecolor', 'none', 'facealpha', 0.2);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',fontSize);
pbaspect([1.5 1 1]);
set(gca, 'YScale', 'log')
xlabel('Quantile','fontsize',fontSize);
ylabel('MDF','fontsize',fontSize);
xticks([0.1 0.3 0.5])
xticklabels({'0.1', '0.3', '0.5'})
grid
set(gca, 'YMinorGrid', 'off')

% Market Distress Frequency figure as function of fraction
pos = pos4{3};
axes('Position', pos);
hold on;
box on;
plot(ks, mdfkHist, 'color', [0,0,0], 'linewidth', 1);
plot(ks, mdfkPccHbN, 'color', [1,0,0], 'linewidth', 1);
plot(ks, mdfkPccSkewtMvt, 'color', [0,0,1], 'linewidth', 1);
plot(ks, mdfkPccSkewtT, 'color', [1,0,1], 'linewidth', 1);
plot(ks, mdfkSkewT, 'color', [0,1,0], 'linewidth', 1);
plot(ks, mdfkT, 'color', [1,1,0], 'linewidth', 1);
plot(ks, mdfkG, 'color', [0 1 1], 'linewidth', 1);
xCI =[ks, flip(ks)];
yCI = [prctile(mdfkBoot,5), flip(prctile(mdfkBoot,95))];
fill(xCI, yCI, 1,'facecolor', 'red', 'edgecolor', 'none', 'facealpha', 0.2);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',fontSize);
pbaspect([1.5 1 1]);
xlabel('Fraction','fontsize',fontSize);
ylabel('MDF','fontsize',fontSize);
set(gca, 'YScale', 'log')
xticks([0.5 0.75 1])
xticklabels({'0.5', '0.75', '1'})
grid
set(gca, 'YMinorGrid', 'off')

% Market Distress Frequency figure as function of dimension
pos = pos4{4};
axes('Position', pos);
hold on;
box on;
plot(ds, mdfdHist, 'color', [0,0,0], 'linewidth', 1);
plot(ds, mdfdPccHbN, 'color', [1,0,0], 'linewidth', 1);
plot(ds, mdfdPccSkewtMvt, 'color', [0,0,1], 'linewidth', 1);
plot(ds, mdfdPccSkewtT, 'color', [1,0,1], 'linewidth', 1);
plot(ds, mdfdSkewT, 'color', [0,1,0], 'linewidth', 1);
plot(ds, mdfdT, 'color', [1,1,0], 'linewidth', 1);
plot(ds, mdfdG, 'color', [0 1 1], 'linewidth', 1);
xCI =[ds, flip(ds)];
yCI = [prctile(mdfdBoot,5), flip(prctile(mdfdBoot,95))];
fill(xCI, yCI, 1,'facecolor', 'red', 'edgecolor', 'none', 'facealpha', 0.2);
set(gca, 'YScale', 'log')
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',fontSize);
pbaspect([1.5 1 1]);
xlabel('Dimension','fontsize',fontSize);
ylabel('MDF','fontsize',fontSize);
axis([2, 20, 0.0001, 0.1])
xticks([5 10 15 20])
xticklabels({'5', '10', '15', '20'})
grid 
set(gca, 'YMinorGrid', 'off')

%% Perform binomial test

% Initialize
nTest = 4;
nsExcHist  = zeros(nTest,1);
pValPccHbN  = zeros(nTest,1);
pValPccSkewT = zeros(nTest,1);
pValPccSkewMvT = zeros(nTest,1);
pValSkewT  = zeros(nTest,1);
pValT      = zeros(nTest,1);
pValG       = zeros(nTest,1);
nValPccHbN  = zeros(nTest,1);
nValPccSkewT = zeros(nTest,1);
nValPccSkewMvT = zeros(nTest,1);
nValSkewT  = zeros(nTest,1);
nValT      = zeros(nTest,1);
nValG      = zeros(nTest,1);

% Determine test specifications (quantiles and market fractions)
qs = [0.15, 0.2, 0.15, 0.2];
ks = [nDim, nDim, 0.9*nDim, 0.9*nDim];

% Determine p-values for test specifications
for iT = 1:nTest
   q = qs(iT);
   k = ks(iT);
   nExcHist        = sum(sum(usHist <= q,2)>= k);
   nsExcHist(iT,1) = nExcHist;
   probPccHbN = sum(sum(usPccHbN <= q,2)>= k)/nSims;
   probPccSkewT = sum(sum(usPccSkewtT <= q,2)>= k)/nSims;
   probPccSkewMvT = sum(sum(usPccSkewtMvt <= q,2)>= k)/nSims;
   probSkewT = sum(sum(usSkewT <= q,2)>= k)/nSims;
   probT   = sum(sum(usT <= q,2)>= k)/nSims;
   probG   = sum(sum(usG <= q,2)>= k)/nSims;

   % Expected values of market stresses in binomial test 
   nValPccHbN(iT,1)     = nData*probPccHbN;
   nValPccSkewT(iT,1)   = nData*probPccSkewT;
   nValPccSkewMvT(iT,1) = nData*probPccSkewMvT;
   nValSkewT(iT,1)      = nData*probSkewT;
   nValT(iT,1)          = nData*probT;
   nValG(iT,1)          = nData*probG;
   
   % p-values binomial test (sum from nExcHist to nData)
   pValPccHbN(iT,1)    = 1-binocdf(nExcHist-1,nData,probPccHbN);
   pValPccSkewT(iT,1)  = 1-binocdf(nExcHist-1,nData,probPccSkewT);
   pValPccSkewMvT(iT,1)= 1-binocdf(nExcHist-1,nData,probPccSkewMvT);
   pValSkewT(iT,1)     = 1-binocdf(nExcHist-1,nData,probSkewT);
   pValT(iT,1)         = 1-binocdf(nExcHist-1,nData,probT);
   pValG(iT,1)         = 1-binocdf(nExcHist-1,nData,probG);
end

% Show realized and expected number of market stresses (n-values)
disp('n-values:')
disp([nsExcHist,nValG,nValT,nValSkewT, nValPccHbN, nValPccSkewT, nValPccSkewMvT])
% Show p-values for binomial test
disp('p-values:')
disp([nsExcHist,pValG,pValT,pValSkewT, pValPccHbN, pValPccSkewT, pValPccSkewMvT])
