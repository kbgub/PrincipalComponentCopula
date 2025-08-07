function [logLh, ys, fYs] = fLogLhPccSkewtMvt(x, ev, ew, us)
   % Function to obtain the loglikelihood of the skew t_1 - multivariate t_{d-1} PCC 
   %
   % This function was used in the following article:
   % Gubbels, K.B., Ypma, J.Y. & Oosterlee, C.W. (2025),
   % Principal Component Copulas for Capital Modelling and Systemic Risk, Computational Economics 
   % https://doi.org/10.1007/s10614-025-11051-7   
   %
   % Inputs:
   %    x:    parameter vector 
   %    ev:   eigenvectors of correlation matrix (direction of PCs)
   %    ew:   eigenvalues of correlation matrix  (variance of PCs)
   %    us:   copula observations for which copula likelihood is determined

   % Specify parameters
   nuT    = x(1);
   gammaT = x(2);
 
   % Additional parameters
   alpha  = abs(gammaT);
   beta   = gammaT;
   lambda = -nuT/2;
   delta  = sqrt(nuT);
   mu     = -delta^2*beta/2*(gamma(-lambda-1)/gamma(-lambda));

   % Characteristic functions (scaled to unit variance for PCs)
   cfSkewT  = @(t) 2^(lambda+1)/gamma(nuT/2)*exp(1i*mu*t).*besselk(lambda,delta*sqrt(alpha^2 - (beta+1i*t).^2))./(delta*sqrt(alpha^2 - (beta+1i*t).^2)).^lambda;
   varSkewT = 2*beta^2*delta^4/((nuT-2)^2*(nuT-4)) + delta^2/(nuT-2);              
   cfP1 = @(t) cfSkewT(t/sqrt(varSkewT));
   nuTp  = nuT/2;
   varTp = nuTp/(nuTp-2);
   cfT   = @(t) besselk(nuTp/2,sqrt(nuTp)*abs(t)).*(sqrt(nuTp)*abs(t)).^(nuTp/2)./(gamma(nuTp/2)*2^(nuTp/2-1));
   cfPj  = @(t) cfT(t/sqrt(varTp));
      
   % Determine ys and density of ys through convolution
   [ys, fYs] = DetermineYs2(cfP1, cfPj, ev, ew, us);

   % Determine PCs
   ps = ys * ev;
   
   % Determine densities for first PCs
   pdSkewT = @(y) 2^(lambda+1)/gamma(nuT/2)*delta^(-2*lambda)/sqrt(2*pi)*exp(beta*(y-mu)).*besselk(lambda-1/2,alpha*sqrt(delta^2 + (y - mu).^2))./((sqrt(delta^2+(y - mu).^2)/alpha).^(1/2-lambda));
   scaleP1  = 1/sqrt(ew(1))*sqrt(varSkewT);
   fPs(:,1) = pdSkewT(ps(:,1)*scaleP1)*scaleP1;  

   % Determine densities for other PCs
   scalesPj = 1./sqrt(ew(2:end))'*sqrt(varTp);
   fPs(:,2) = mvtpdf(ps(:,2:end).*scalesPj,eye(numel(ew)-1),nuTp)*prod(scalesPj);

   % Determine loglikelihood
   logLh = sum(sum(log(fPs)))-sum(sum(log(fYs)));  
end