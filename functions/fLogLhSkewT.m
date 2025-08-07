function [logLh, ys, fYsUv] = fLogLhSkewT(x, ev, ew, us)
   % Function to obtain the loglikelihood of the multivariate GH skew t copula 
   %
   % This function was used in the following article:
   % Gubbels, K.B., Ypma, J.Y. & Oosterlee, C.W. (2025),
   % Principal Component Copulas for Capital Modelling and Systemic Risk, Computational Economics 
   % https://doi.org/10.1007/s10614-025-11051-7   
   %
   % Inputs:
   %    x:    parameter vector that can be optimized
   %    ev:   eigenvectors of correlation matrix (direction of PCs)
   %    ew:   eigenvalues of correlation matrix  (variance of PCs)
   %    us:   copula observations for which copula likelihood is determined

   % Specify parameters
   nDim   = size(us,2);
   nuT    = x(1);
   gammaT = x(2);
   ew(1)  = x(3);
   ew(2)  = x(4);
   ew     = max(ew,0);
   ew     = ew*nDim/sum(ew); % Trace equals number of dimensions
   
   % Get valid correlation matrices
   rhos    = ev*diag(ew)*ev';
   
   % Get other relevant parameters
   mu     = 0;
   mus    = mu*ones(1,nDim);
   gammas = gammaT*ev(:,1)';
   alpha  = sqrt(gammas*(rhos\gammas'));
   pow    = (nuT + nDim)/2;
   lambda = -nuT/2;
   delta  = sqrt(nuT);
   
   % Initialize
   ys    = zeros(size(us));
   fYsUv = zeros(size(us));
   
   % Univariate skew T
   for iVar = 1:nDim
      % Univariate alpha and beta
      alphaI = abs(gammaT*ev(iVar,1));
      betaI  = gammaT*ev(iVar,1);

      % Characteristic functions
      cfSkewT  = @(t) 2^(lambda+1)/gamma(nuT/2)*exp(1i*mu*t).*besselk(lambda,delta*sqrt(alphaI^2 - (betaI+1i*t).^2))...
         ./(delta*sqrt(alphaI^2 - (betaI+1i*t).^2)).^lambda;
      pdY = cf2QdCos(cfSkewT, [], us(:,iVar));
      
      % Determine univariate density of ys
      ys(:,iVar)  = pdY.qYs;
      fYsUv(:,iVar) = pdY.fYs; 
   end
 
   % Perform Choleski decomposition
   rhosChol   = cholcov(rhos,0);
   ysSq       = sum(((ys-mus)/rhosChol).^2,2);
   sqrtDetRho = prod(diag(rhosChol));
   
   % Multivariate Skew T
   cNorm = 2^(lambda+1)/gamma(nuT/2)*delta^(-2*lambda)/(2*pi)^(nDim/2)*alpha.^(pow)/sqrtDetRho;
   fYsMv = cNorm * exp((ys-mus)*(rhos\gammas')).*besselk(pow,alpha*sqrt(nuT + ysSq))...
      ./(nuT + ysSq).^(pow/2);

   % Determine loglikelihood
   logLh = sum(log(fYsMv)) - sum(sum(log(fYsUv)));
end