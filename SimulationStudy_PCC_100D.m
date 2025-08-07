%% Script to perform analysis of estimator performance
%
% Script analyzes estimation performance of the hyperbolic-normal Principal Component Copula
%
% First, we specify correlation structure and shape parameters for first PC and higher PCs
% Then, we simulate 100-dimensional copula data using the Data Generating Process of the PCC
% Next, we estimate the copula parameters from the simulated copula data 
% We perform simulation and estimation multiple times to assess estimator performance
%
% This script is an used in the following article:
% Gubbels, K.B., Ypma, J.Y. & Oosterlee, C.W. (2025),
% Principal Component Copulas for Capital Modelling and Systemic Risk, Computational Economics 
% https://doi.org/10.1007/s10614-025-11051-7  
%
% In case of 10 Monte Carlo iterations, the script takes less than five minutes to run
% In the article 100 simulations are used

% Initialize
close all;
clear;
rng(1);
addpath('./functions');

%% Settings

% Random number seed
rng(1);

% Settings estimation
nRec  = 5;     % Number of recursions
nSims = 1500;  % Number of observations
nMc   = 10;    % Number of Monte Carlo runs

%% Correlation structure

% Parametrize correlation matrix in high dimensions using 2 factors
nDim   = 100;
betas  = 2*(exp(-(1:nDim)/nDim)+1)/5;
gammas = 3*tanh((4*(1:nDim)/nDim-2))/5;

% Specify correlation matrix
corrMat = ones(nDim);
for iVar = 1:nDim
   for jVar = iVar+1:nDim
      corrMat(iVar, jVar) = betas(iVar)*betas(jVar)+ gammas(iVar)*gammas(jVar);
      corrMat(jVar, iVar) = corrMat(iVar, jVar);
   end
end     

% Perform PCA
[ev, ew] = pcacov(corrMat);

%% Specify parameters and distributions for Principal Components

% We start by specifying parameters for PC upper and lower tails
alphaP1 = 0.5;
betaP1  = -0.25;
alphaP2 = 1;
betaP2  = 0.25;

% Determine alternative parametrization (see additional note)
parsOut = [alphaP1,betaP1,alphaP2,betaP2];
funPars = @(x) ScalePars(x, ew) - parsOut;
parsSol = fsolve(funPars, [2,10,15,10]);
parsCheck = ScalePars(parsSol, ew);

% Parameters for alternative parametrization
alpha1 = (parsSol(1) + parsSol(2))/2;
beta1  = (parsSol(1) - parsSol(2))/2;
alpha2 = (parsSol(3) + parsSol(4))/2;
beta2  = (parsSol(3) - parsSol(4))/2;

% Mean and variance of hyperbolic distribution for PC1
gamma1 = sqrt(alpha1^2 - beta1^2);
mu1    = -beta1/gamma1*besselk(2,gamma1)/besselk(1,gamma1);
varHb1 = 1/gamma1*besselk(2,gamma1)/besselk(1,gamma1)+(beta1^2/gamma1^2)*(besselk(3,gamma1)/besselk(1,gamma1)-besselk(2,gamma1)^2/besselk(1,gamma1)^2); 

% Mean and variance of hyperbolic disitrbution for PC1
gamma2 = sqrt(alpha2^2 - beta2^2);
mu2    = -beta2/gamma2*besselk(2,gamma2)/besselk(1,gamma2);
varHb2 = 1/gamma2*besselk(2,gamma2)/besselk(1,gamma2)+(beta2^2/gamma2^2)*(besselk(3,gamma2)/besselk(1,gamma2)-besselk(2,gamma2)^2/besselk(1,gamma2)^2);

%% Perform Monte Carlo analysis of estimator

% Characteristic function of PC1 and PC2 (before scaling variance)
cfHb1 = @(t) exp(1i*mu1*t).*gamma1.*besselk(1,sqrt(alpha1^2 - (beta1+1i*t).^2))./(sqrt(alpha1^2 - (beta1+1i*t).^2).*besselk(1,gamma1));
cfHb2 = @(t) exp(1i*mu2*t).*gamma2.*besselk(1,sqrt(alpha2^2 - (beta2+1i*t).^2))./(sqrt(alpha2^2 - (beta2+1i*t).^2).*besselk(1,gamma2));

% Characteristic functions for higher PCs (normal)
cfN  = @(t) exp(-t.^2/2);

% Perform initial ML for shape parameters
parsMlAll  = zeros(nMc, 6);
parsGmmAll = zeros(nMc, 6);

for iMc = 1:nMc

   % Perform simulation: simulate principal components (first with unit variance)
   cfP1 = @(t) cfHb1(t/sqrt(varHb1));
   p1   = cf2QdCos(cfP1, [], rand(nSims,1));
   cfP2 = @(t) cfHb2(t/sqrt(varHb2));
   p2   = cf2QdCos(cfP2, [], rand(nSims,1));
   % Scale PCs to required variance
   ps(:,1) = sqrt(ew(1))*p1.qYs;
   ps(:,2) = sqrt(ew(2))*p2.qYs;
   
   % Simulate other PCs
   for iDim = 3:nDim
      ps(:,iDim) = icdf('normal', rand(nSims,1), 0, sqrt(ew(iDim)));
   end

   % Determine simulated y's
   ys = ps * ev';

   % Determine simulated u's with marginal distribution
   us = zeros(nSims,nDim);
   for iVar = 1:nDim
      sigmaP1 = sqrt(ew(1))*ev(iVar,1); 
      sigmaP2 = sqrt(ew(2))*ev(iVar,2); 
      sigmaN  = sqrt(ev(iVar,3:end)*diag(ew(3:end))*ev(iVar,3:end)'); 
      cfY = @(t) cfHb1(sigmaP1*t/sqrt(varHb1)) .* cfHb2(sigmaP2*t/sqrt(varHb2)) .* cfN(sigmaN*t);
      pdY = cf2QdCos(cfY, ys(:,iVar),[]);
      us(:,iVar) = pdY.cYs;
   end
   
   % Perform estimation: initialize with gaussian rank correlations
   ns    = icdf('norm', us, 0, 1);
   corrN = corr(ns);
   [evN, ewN] = pcacov(corr(ns));
   evRec = evN;
   ewRec = ewN;

   % Perform ML with exact ev and ew as performance check
   pars0  = [1, 5, 20, 5];
   parsLb = [0.5, 0.5, 0.5, 0.5];
   parsUb = [25, 25, 25, 25];
   funLogLh = @(x) -DetermineLogLh(x, ev, ew, us);
   optMl      = optimoptions(@fmincon, 'MaxIterations', 50, 'StepTolerance', 0.05); 
   parsEstMl  = fmincon(funLogLh,pars0,[],[],[],[],parsLb,parsUb,[],optMl);
   parsMl     = ScalePars(parsEstMl, ew);
   parsMlAll(iMc,:)  = [ew(1), ew(2), parsMl];
   
   % Perform initialization step for GMM
   parsUb = [50, 50, 50, 50];
   funLogLh = @(x) -DetermineLogLh(x, evN, ewN, us);
   optMl    = optimoptions('fmincon', 'MaxIterations', 20, 'StepTolerance',0.05);
   parsEst  = fmincon(funLogLh,pars0,[],[],[],[],parsLb,parsUb,[],optMl);

   % Perform n recursions for GMM estimation
   for iRec = 1:nRec
      % Update densities and correlations
      [~,ysRec,~] = DetermineLogLh(parsEst, evRec, ewRec, us);
      [evRec, ewRec] = pcacov(corr(ysRec));

      % Perform maximum likelihood for shape parameters 
      funLogLh  = @(x) -DetermineLogLh(x, evRec, ewRec, us);
      optMl     = optimoptions('fmincon', 'MaxIterations', 15, 'StepTolerance',0.05);
      parsEst   = fmincon(funLogLh, parsEst,[],[],[],[],parsLb,parsUb,[],optMl);
      disp(['Recursion ' num2str(iRec) ' completed']);
   end

   % Gather result with recursive GMM estimation
   parsGmm  = ScalePars(parsEst, ewRec);
   parsGmmAll(iMc,:) = [ewRec(1), ewRec(2), parsGmm];
   disp(iMc);
end

% Show results
parsTrue = [ew(1),ew(2),alphaP1,betaP1,alphaP2,betaP2];
tabResultsMc = [parsTrue', mean(parsMlAll)', std(parsMlAll)', mean(parsGmmAll)', std(parsGmmAll)'];
disp(tabResultsMc);

%% Show eigenvectors in high dimensional case

% Figure
figure
hold on;
plot(ev(:,1),'LineWidth',1);
plot(evRec(:,1), '--x','LineWidth',0.1,'MarkerSize', 5);
plot(ev(:,2),'LineWidth',1);
plot(evRec(:,2), '--x','LineWidth',0.1,'MarkerSize', 5);
pbaspect([3 1 1])
xlabel('Variable index')
ylabel('Eigen vector weight')
legend('PC1: $w_{i,1}$','PC1: $\hat{w}_{i,1}$','PC2: $w_{i,2}$','PC2: $\hat{w}_{i,2}$', 'Location','southeast','Interpreter','latex')

%% Functions

function [logLh, ys, fYs] = DetermineLogLh(x, ev, ew, us)

   % Specify parameters
   alpha1 = (x(1) + x(2))/2;
   beta1  = (x(1) - x(2))/2;
   alpha2 = (x(3) + x(4))/2;
   beta2  = (x(3) - x(4))/2;

   % Additional parameters
   gamma1 = sqrt(alpha1^2 - beta1^2);
   gamma2 = sqrt(alpha2^2 - beta2^2);
   mu1    = -beta1/gamma1*besselk(2,gamma1)/besselk(1,gamma1);
   mu2    = -beta2/gamma2*besselk(2,gamma2)/besselk(1,gamma2);
   vars.Hb1 = 1/gamma1*besselk(2,gamma1)/besselk(1,gamma1)+(beta1^2/gamma1^2)*(besselk(3,gamma1)/besselk(1,gamma1)-besselk(2,gamma1)^2/besselk(1,gamma1)^2);
   vars.Hb2 = 1/gamma2*besselk(2,gamma2)/besselk(1,gamma2)+(beta2^2/gamma2^2)*(besselk(3,gamma2)/besselk(1,gamma2)-besselk(2,gamma2)^2/besselk(1,gamma2)^2);
        
   % Characteristic functions
   cfs.Hb1 = @(t) exp(1i*mu1*t).*sqrt(alpha1^2 - beta1^2).*besselk(1,sqrt(alpha1^2 ...
     - (beta1+1i*t).^2))./(sqrt(alpha1^2 - (beta1+1i*t).^2).*besselk(1,sqrt(alpha1^2 - beta1^2)));
   cfs.Hb2 = @(t) exp(1i*mu2*t).*sqrt(alpha2^2 - beta2^2).*besselk(1,sqrt(alpha2^2 ...
     - (beta2+1i*t).^2))./(sqrt(alpha2^2 - (beta2+1i*t).^2).*besselk(1,sqrt(alpha2^2 - beta2^2)));
   cfs.N  = @(t) exp(-t.^2/2);       
   
   % Determine ys and density of ys through convolution
   [ys, fYs] = DetermineYs(cfs, vars, ev, ew, us);

   % Determine pc-observations
   ps = ys * ev;
   
   % Determine densities for first pcs
   pdGH1    = @(x) gamma1*exp(-alpha1*sqrt(1 + (x - mu1).^2)+beta1*(x-mu1))./(2*alpha1*besselk(1,gamma1));
   scale1   = 1/sqrt(ew(1))*sqrt(vars.Hb1);
   fPs(:,1) = pdGH1(ps(:,1)*scale1)*scale1;  
   pdGH2    = @(x) gamma2*exp(-alpha2*sqrt(1 + (x - mu2).^2)+beta2*(x-mu2))./(2*alpha2*besselk(1,gamma2));
   scale2   = 1/sqrt(ew(2))*sqrt(vars.Hb2);
   fPs(:,2) = pdGH2(ps(:,2)*scale2)*scale2;
   
   % Determine densities for other pcs
   for iVar = 3:numel(ew)
      fPs(:, iVar) = normpdf(ps(:,iVar),0,sqrt(ew(iVar)));
   end

   % Determine loglikelihood
   logLh = sum(sum(log(fPs)-log(fYs)));  
end

function [ys, fYs] = DetermineYs(cfs, vars, ev, ew, us)        
    
   % Initialize
   ys  = zeros(size(us));
   fYs = zeros(size(us));
    
   % Determine ys and density of ys through convolution
   for iVar = 1:numel(ew)
      
      % Determine standard deviations
      sigmaP1 = sqrt(ew(1))*ev(iVar,1); 
      sigmaP2 = sqrt(ew(2))*ev(iVar,2); 
      sigmaN  = sqrt(ev(iVar,3:end)*diag(ew(3:end))*ev(iVar,3:end)'); 
      
      % Perform convolution 
      cfY = @(t) cfs.Hb1(sigmaP1*t/sqrt(vars.Hb1)) .* cfs.Hb2(sigmaP2*t/sqrt(vars.Hb2)) .* cfs.N(sigmaN*t);
      pdY = cf2QdCos(cfY, [], us(:,iVar));
      
      % Determine ys and density of ys
      ys(:,iVar)  = pdY.qYs;
      fYs(:,iVar) = pdY.fYs; 
   end   
end

function parsOut = ScalePars(x, ew)

   % Specify parameters
   alpha1 = (x(1) + x(2))/2;
   beta1  = (x(1) - x(2))/2;
   alpha2 = (x(3) + x(4))/2;
   beta2  = (x(3) - x(4))/2;

   % Additional parameters for scaling 
   gamma1 = sqrt(alpha1^2 - beta1^2);
   gamma2 = sqrt(alpha2^2 - beta2^2);
   varGH1  = 1/gamma1*besselk(2,gamma1)/besselk(1,gamma1)+(beta1^2/gamma1^2)*(besselk(3,gamma1)/besselk(1,gamma1)-besselk(2,gamma1)^2/besselk(1,gamma1)^2);
   varGH2  = 1/gamma2*besselk(2,gamma2)/besselk(1,gamma2)+(beta2^2/gamma2^2)*(besselk(3,gamma2)/besselk(1,gamma2)-besselk(2,gamma2)^2/besselk(1,gamma2)^2);

   % Show new rescaled parameters
   parsOut(1) = alpha1* sqrt(varGH1)/sqrt(ew(1)); 
   parsOut(2) = beta1 * sqrt(varGH1)/sqrt(ew(1)); 
   parsOut(3) = alpha2 * sqrt(varGH2)/sqrt(ew(2)); 
   parsOut(4) = beta2 *sqrt(varGH2)/sqrt(ew(2)); 
end 
