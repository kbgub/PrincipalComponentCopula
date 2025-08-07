%% Script to perform analysis of estimator performance
%
% Script analyzes estimation performance of the skew t_1 and multivariate
% t_{d_1} Principal Component Copula
%
% First, we specify correlation structure and shape parameters for first PC and higher PCs
% Then, we simulate 20-dimensional copula data using the Data Generating Process of the PCC
% Next, we estimate the copula parameters from the simulated copula data 
% We perform simulation and estimation multiple times to assess estimator performance
%
% This script is an additional analysis based on the following article:
% Gubbels, K.B., Ypma, J.Y. & Oosterlee, C.W. (2025),
% Principal Component Copulas for Capital Modelling and Systemic Risk, Computational Economics 
% https://doi.org/10.1007/s10614-025-11051-7  
%
% In case of 10 Monte Carlo iterations, the script takes less than one minute to run

% Initialize
close all;
clear;
addpath('./functions');

%% Settings

% Random number seed
rng(1);

% Settings estimation
nRec  = 3;     % Number of recursions
nSims = 1500;  % Number of observations
nMc   = 10;    % Number of monte-carlo replicaitons

%% Specify correlation structure

% Parametrize correlation matrix in highD using 2 factors
nDim   = 20;
betas  = (2*exp(-(1:nDim)/nDim)+2)/5;
gammas = 3*tanh(4*(-nDim/2:nDim/2)/nDim)/5;

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

% Specify parameters for the Skew t and multivariate t
nuT    = 20;
gammaT = -2;
alpha  = abs(gammaT);
beta   = gammaT;
lambda = -nuT/2;
delta  = sqrt(nuT);
varSkewT = 2*beta^2*delta^4/((nuT-2)^2*(nuT-4)) + delta^2/(nuT-2);
nuTp  = nuT/2;
varTp = nuTp/(nuTp-2);
mus  = zeros(1,nDim);
rhos = eye(nDim);
mu1  = -gammaT*nuT /(nuT-2);

%% Simulate copula and estimate parameters

% Loop over monte carlo samples
parsMcRec = zeros(nMc, 4);
for iMc = 1:nMc
  
   %%  Simulate skew t-mvt PCC

   % Simulate PCs
   ws1  = 1./gamrnd(nuT/2,2/nuT, nSims,1);
   ws2  = 1./gamrnd(nuTp/2,2/nuTp, nSims,1);
   zSims   = mvnrnd(mus, eye(nDim), nSims);
   ps(:,1) = (mu1 + gammaT*ws1 + sqrt(ws1).*zSims(:,1))/sqrt(varSkewT);
   ps(:,2:nDim) = sqrt(ws2).*zSims(:,2:end)/sqrt(varTp);

   % Determine simulated ys
   ys = (ps .* sqrt(ew)') * ev';

   % Determine simulated us
   us = tiedrank(ys)/(nSims+1);

   %% Estimate skew t-mvt PCC

   % Determine gaussian rank correlations (initialize recursion)
   ns    = icdf('norm', us, 0, 1);
   corrN = corr(ns);
   [evN, ewN] = pcacov(corr(ns));

   % Initialize estimation
   pars0  = [15, -1];
   parsLb = [5, -3];
   parsUb = [50, -0.1];

   % Perform initial ML for shape parameters
   optMl   = optimoptions('fmincon','Display', 'off', 'MaxIterations', 25);
   fOpt    = @(x) -fLogLhPccSkewtMvt(x, evN, ewN, us);
   parsIni = fmincon(fOpt,pars0,[],[],[],[],parsLb,parsUb,[],optMl);

   % Record results from initializing estimation step
   parsRec = parsIni;
   evRec   = evN;
   ewRec   = ewN;

   % Perform n recursions
   for iRec = 1:nRec
      % Update densities and correlations
      [~,ysRec,~]    = fLogLhPccSkewtMvt(parsRec, evRec, ewRec, us);
      [evRec, ewRec] = pcacov(corr(ysRec));

      % Perform maximum likelihood  for shape parameters (with previous step as start)
      fOpt      = @(x) -fLogLhPccSkewtMvt(x, evRec, ewRec, us);
      optMl     = optimoptions('fmincon', 'MaxIterations', 20);
      parsRec   = fmincon(fOpt, parsRec,[],[],[],[],parsLb,parsUb,[],optMl);      
   end
   parsMcRec(iMc,:) = [parsRec, ewRec(1:2)'];
   disp(iMc);
end

% Show GMM estimation results
parsTrue = [nuT,gammaT,ew(1:2)'];
disp('Show true parameters, mean estimates and standard deviation:');
disp([parsTrue', mean(parsMcRec)', std(parsMcRec)']);
