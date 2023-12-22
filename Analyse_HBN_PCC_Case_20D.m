%% Script to estimate PCC on historical return 
%
% This script performs (simplified version of) estimation in case
% study of: 'Principal Component Copulas for Capital Modelling' [1]
% https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4669797
%
% The results may differ from the paper due to different implementation
% choices to avoid Matlab toolboxes, etc. Conclusions are not affected.
%
% This is not production code, but rather an illustrative example.
%
% If you find any issues with this script, please let me know:
% koosgubbels@gmail.com

close all;
clear;
rng(0);

%% Prepare data

% Load historical weekly index data, obtained from Yahoo Finance
load('dataIndices');
% For the meaning of the indices, see Ref. [1]

% Determine historical total returns and logreturns
trHist = dataAll{2:end,2:end} ./ dataAll{1:end-1,2:end};
xsHist = log(dataAll{2:end,2:end} ./ dataAll{1:end-1,2:end});

% Initialize variables
nDim   = size(xsHist,2);
nData  = size(xsHist,1);
zsHist = zeros(size(xsHist));
sigmasHist = zeros(size(xsHist));
parsGarch  = zeros(nDim,3);

% Apply GARCH filter
for iVar = 1:nDim
   pars0  = [0.5, 0.5, 0.5];
   parsLb = [0, 0, 0];
   parsUb = [1, 1, 1];

   % Perform Quasi ML (see likelihood function below)
   fOpt = @(x) -fLogLhGarch(x, xsHist(:,iVar));
   parsGarch(iVar,:) = fminunc(fOpt,pars0);
   [logLh, zsHist(:,iVar), sigmasHist(:,iVar)]= fLogLhGarch(parsGarch(iVar,:), xsHist(:,iVar));
end

% Use filtered results
xsHist = zsHist;

% Perform PCA for filtered variables
corrX = corr(xsHist, 'rows','complete');
[evX, ewX] = pcacov(corrX);

% Determine pseudo-copula observations
usHist = tiedrank(xsHist)/(nData + 1);

% Determine gaussian rank correlations and perform initial PCA 
nsHist = icdf('norm', usHist, 0, 1);
[evN, ewN] = pcacov(corr(nsHist));

% Historical Principal Component (PC) observations (initiation step)
psHistIni = nsHist * evN;

%% Visualize vector weights PCs

% Show eigenvector weights of first 3 PC vectors
figure
hold on;
plot1 = plot(evN(:,1), '-x','LineWidth',0.1,'MarkerSize', 5);
plot2 = plot(evN(:,2), '-x','LineWidth',0.1,'MarkerSize', 5);
plot3 = plot(evN(:,3), '-x','LineWidth',0.1,'MarkerSize', 5);
yline(0)
pbaspect([3 1 1])
axis([1, 20, -0.4, 0.6])
xlabel('Variable index')
ylabel('Eigen vector weight')
legend([plot1, plot2, plot3],'PC1','PC2','PC3', 'Location','southeast','NumColumns',3);

%% Fit PCC HB-N copula

% Initialize shape parameters of hyperbolic distribution
pars0  = [5, 5];

% Perform initial ML for shape parameters
fOpt    = @(x) -fLogLhPccHbN(x, evN, ewN, usHist);
parsIni = fminunc(fOpt,pars0);

% Initialize shape and correlation parameters (eigen vectors and weights)
disp(parsIni)
parsRec = parsIni;
evRec   = evN;
ewRec   = ewN;

% Perform n recursions for convergence (GMM estimation approach from [1])
nRec = 3;
for iRec = 1:nRec
   % Update densities and correlations
   [~,ysRec,~] = fLogLhPccHbN(parsRec, evRec, ewRec, usHist);
   [evRec, ewRec] = pcacov(corr(ysRec));

   % Perform maximum likelihood (with previous step as start)
   fOpt    = @(x) -fLogLhPccHbN(x, evRec, ewRec, usHist);
   parsRec = fminunc(fOpt, parsRec);
   
   % Show in between iterative estimation results
   disp(parsRec)
end

% Final historical transformed risk factor and PC observations 
ysHist = ysRec;
psHist = ysRec * evRec;

%% Visualize densities of PCs in PCC HB-N copula

% Parameters for estimated Hyperbolic distribution for P1
alpha   = (parsRec(1) + parsRec(2))/2;
beta    = (parsRec(1) - parsRec(2))/2;
gammaHB = sqrt(alpha^2 - beta^2);
mu      = -beta/gammaHB*besselk(2,gammaHB)/besselk(1,gammaHB);
varHB   = 1/gammaHB*besselk(2,gammaHB)/besselk(1,gammaHB)+...
   (beta^2/gammaHB^2)*(besselk(3,gammaHB)/besselk(1,gammaHB)-besselk(2,gammaHB)^2/besselk(1,gammaHB)^2);

% First version of pdf for first PC (with delta=1 as in estimation)
sigma  = sqrt(ewRec(1))/sqrt(varHB);
pdP1v1 = @(x) gammaHB*exp(-alpha*sqrt(1 + (x/sigma - mu).^2)+ beta*(x/sigma-mu))...
   ./(2*sigma*alpha*besselk(1,gammaHB));

% Rescale parameters towards their most common form    
alphaP1 = alpha*sqrt(varHB)/sqrt(ewRec(1));
betaP1  = beta*sqrt(varHB)/sqrt(ewRec(1));
gammaP1 = sqrt(alphaP1^2 - betaP1^2);
deltaP1 = sqrt(ewRec(1))/sqrt(varHB);
muP1    = -deltaP1*betaP1/gammaP1*besselk(2,deltaP1*gammaP1)/besselk(1,deltaP1*gammaP1);

% Second, most common version of pdf for first PC
pdP1v2  = @(x) gammaP1*exp(-alphaP1*sqrt(deltaP1^2 + (x - muP1).^2)+ betaP1*(x-muP1))...
   ./(2*alphaP1*deltaP1*besselk(1,deltaP1*gammaP1));

% Plot Principal Component densities and compare with data
for iPc = 1:3
   
   % Plot PC densities and compare with normal
   sigmaPc   = std(psHist(:,iPc));
   x         = -4*sigmaPc :sigmaPc/100 :4*sigmaPc;
   if iPc == 1
      pdfPc = pdP1v2(x);
   else
      ndPc  = fitdist(psHist(:,iPc), 'Normal');  
      pdfPc = pdf(ndPc,x);
   end
   figure;
   histogram(psHist(:,iPc),50,'Normalization','pdf');
   line(x,pdfPc,'LineStyle','-','Color','r', 'linewidth', 1.5);
end

%% Simulate estimated HB-N PCC 

% Characteristic functions of hyperbolic and normal distribution
cfHB = @(t) exp(1i*mu*t).*gammaHB.*besselk(1,sqrt(alpha^2 - (beta+1i*t).^2)) ...
   ./(sqrt(alpha^2 - (beta+1i*t).^2).*besselk(1,gammaHB));
cfN  = @(t) exp(-t.^2/2);       

% Simulate PCs
nSims = 10^6;
cfP1  = @(t) cfHB(t/sqrt(varHB));
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
   cfY = @(t) cfHB(sigmaP1*t/sqrt(varHB)) .* cfN(sigmaN*t);
   pdY  = cf2QdCos(cfY, ysSim(:,iVar), []);
   usSim(:,iVar) = pdY.cYs;
end

%% CPJQE analysis

% Initialize. Choose the 5% quantile. 
q     = 0.05;
nComb = nDim*(nDim-1)/2;
cpjqeHist    = zeros(nComb,1);
cpjqePccHbN  = zeros(nComb,1); 

% Determine average CPJQE over all risk pairs (historically and simulated)
iCount = 1;
for iVar = 1:nDim-1
   for jVar = iVar:nDim
      cpjqeHist(iCount)  = sum(sum(usHist(:,[iVar,jVar]) <= q,2)==2)/(q*nData);
      cpjqePccHbN(iCount)= sum(sum(usSim(:,[iVar,jVar]) <= q,2)==2)/(q*nSims);
      iCount=iCount+1;
   end
end

% Pairwize dependence
disp('Aggregate pairwise exceedances (HB-N PCC and historical):')
disp([mean(cpjqePccHbN), mean(cpjqeHist)]);

%% Visualize simulations and historic data of HB-N PCC model

% Scatter plots
nPlot = 5000;
iVar1 = 2;  % FTSE
iVar2 = 15; % S&P500

figure;
scatter(usSim(1:nPlot,iVar1), usSim(1:nPlot,iVar2),2,"o","r");
box on;
hold on;
scatter(usHist(:,iVar1), usHist(:,iVar2),7,"o","b");
xlabel('{\it u}_{2,t}', 'FontName', 'Times')
ylabel('{\it u}_{15,t}', 'FontName', 'Times')
legend('Simulated','Historic', 'Location','southeast')

figure;
scatter(ysSim(1:nPlot,iVar1), ysSim(1:nPlot,iVar2),2,"o","r");
box on;
hold on;
scatter(ysHist(:,iVar1), ysHist(:,iVar2),7,"o","b");
xlabel('{\it y}_{2,t}', 'FontName', 'Times')
ylabel('{\it y}_{15,t}', 'FontName', 'Times')
legend('Simulated','Historic', 'Location','southeast')

figure;
iPc1 = 1;
iPc2 = 2;
scatter(psSim(1:nPlot,iPc1), psSim(1:nPlot,iPc2),2,"o","r");
box on;
hold on;
scatter(psHist(:,iPc1), psHist(:,iPc2),7,"o","b");
xlabel('{\it p}_{1,t}', 'FontName', 'Times')
ylabel('{\it p}_{2,t}', 'FontName', 'Times')
legend('Simulated','Historic', 'Location','southeast')

%% Required likelihood functions

function [logLh, zHist, sigmas]= fLogLhGarch(x, xHist)
     
   % Specify GARCH parameters
   alpha0 = x(1);
   alpha1 = x(2);
   beta1  = x(3);

   % Initialize
   nData  = size(xHist,1);
   sigmas = zeros(nData,1);
   fXs    = zeros(nData,1);
   sigmas(1) = sqrt(alpha0);
   fXs(1) = normpdf(xHist(1)/sigmas(1))/sigmas(1);

   % Loop over time
   for iT = 1:nData-1
      % Determine volatility and density
      sigmas(iT+1) = sqrt(alpha0 + alpha1*xHist(iT)^2 + beta1*sigmas(iT)^2);
      fXs(iT+1) = normpdf(xHist(iT+1)/sigmas(iT+1))/sigmas(iT+1);
   end

   % Determine loglikelihood
   logLh = sum(log(fXs));  
   zHist = xHist./sigmas;
end

function [logLh, ys, fYs] = fLogLhPccHbN(x, ev, ew, us)
   % Loglikelihood of copula density for Hyberbolic-Normal PCC

   % Specify parameters of HB distribution
   alpha = (x(1) + x(2))/2;
   beta  = (x(1) - x(2))/2;
   gamma = sqrt(alpha^2 - beta^2);
   
   % The location parameter mu is fixed such that mean is zero
   mu    = -beta/gamma*besselk(2,gamma)/besselk(1,gamma);
   % We use a parametrization of the HB distribution where delta is one. 
   
   % Variance of hyperbolic distribution
   varHB  = 1/gamma*besselk(2,gamma)/besselk(1,gamma)+(beta^2/gamma^2)...
      *(besselk(3,gamma)/besselk(1,gamma)-besselk(2,gamma)^2/besselk(1,gamma)^2);
        
   % Characteristic functions of hyperbolic and normal
   cfHB = @(t) exp(1i*mu*t).*sqrt(alpha^2 - beta^2).*besselk(1,sqrt(alpha^2 ...
     - (beta+1i*t).^2))./(sqrt(alpha^2 - (beta+1i*t).^2).*besselk(1,sqrt(alpha^2 - beta^2)));   
   
   % Determine ys and density of ys through convolution
   [ys, fYs] = DetermineYs(cfHB, varHB, ev, ew, us);

   % Determine pc-observations
   ps = ys * ev;
   
   % Determine densities for first pcs
   pdHB = @(x) gamma*exp(-alpha*sqrt(1 + (x - mu).^2)+beta*(x-mu))...
      ./(2*alpha*besselk(1,gamma));
   
   % Scale towards variance of first PC (given by first eigenvalue ew(1))
   scaleP1  = 1/sqrt(ew(1))*sqrt(varHB);
   fPs(:,1) = pdHB(ps(:,1)*scaleP1)*scaleP1;  
   
   % Determine densities for other PCs
   for iVar = 2:numel(ew)
      fPs(:, iVar) = normpdf(ps(:,iVar),0,sqrt(ew(iVar)));
   end

   % Determine loglikelihood of copula
   logLh = sum(sum(log(fPs)-log(fYs)));  
end

function [ys, fYs] = DetermineYs(cfHB, varHB, ev, ew, us)        
    
   % Initialize
   ys  = zeros(size(us));
   fYs = zeros(size(us));
    
   % Determine ys and density of ys through convolution
   for iVar = 1:numel(ew)
      
      % Perform convolution: w_i1*P_1 + sum_j>1 w_ij*P_j. 
      % Standard deviation of the two terms:
      sigmaP1 = sqrt(ew(1))*ev(iVar,1); 
      sigmaN  = sqrt(ev(iVar,2:end)*diag(ew(2:end))*ev(iVar,2:end)'); 
      
      % Perform convolution through characteristic functions
      cfN = @(t) exp(-t.^2/2);    
      cfY = @(t) cfHB(sigmaP1*t/sqrt(varHB)) .* cfN(sigmaN*t);
      pdY = cf2QdCos(cfY, [], us(:,iVar));
      
      % Determine ys and density of ys
      ys(:,iVar)  = pdY.qYs;
      fYs(:,iVar) = pdY.fYs; 
   end   
end

function pdY = cf2QdCos(cfY, ys, us)

% Parameters below typically appropriate when variance of Y equals 1.
a    = -10;
b    = 10;
nCos = 100;
deltaGrid = 1/100;

% Initialize grids
yGrid = a : deltaGrid :b;
kGrid = 1:nCos;
sinGrid = sin(kGrid'.*(yGrid - a)*pi/(b-a));
cosGrid = cos(kGrid'.*(yGrid - a)*pi/(b-a));

% Calculate expansion coefficients of COS method
cfY0  = 1;
coefs = 2/(b-a)*real(cfY(kGrid*pi/(b-a)).*exp(-1i*kGrid*a*pi/(b-a)));
coef0 = 2/(b-a)*cfY0;

% Calculate pdf and cdf
pdfY = coef0/2 + coefs * cosGrid;
cdfY = coef0/2*(yGrid-a) + (b-a)/pi*(coefs./kGrid) * sinGrid;
pdfY = max(eps, pdfY);
cdfY = min(max(eps, cdfY),1-eps);

% Interpolate to final results
if ~isempty(us)
   [cdfU,idU] = unique(cdfY);
   pdY.qYs    = interp1(cdfU, yGrid(idU), us, 'pchip');
end
if isempty(ys)
   pdY.fYs  = interp1(yGrid, pdfY, pdY.qYs, 'pchip');
else
   pdY.fYs  = interp1(yGrid, pdfY, ys, 'pchip');
   pdY.cYs  = interp1(yGrid, cdfY, ys, 'pchip');
end

end
