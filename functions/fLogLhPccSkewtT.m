function [logLh, ys, fYs] = fLogLhPccSkewtT(x, ev, ew, us)
   % Function to obtain the loglikelihood of the skew t_1 - t_1 PCC 
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

   % Characteristic functions (scaled to unit variance)
   cfSkewT  = @(t) 2^(lambda+1)/gamma(nuT/2)*exp(1i*mu*t).*besselk(lambda,delta*sqrt(alpha^2 - (beta+1i*t).^2))./(delta*sqrt(alpha^2 - (beta+1i*t).^2)).^lambda;
   varSkewT = 2*beta^2*delta^4/((nuT-2)^2*(nuT-4)) + delta^2/(nuT-2); 
   nuTp  = nuT/2;
   varTp = nuTp/(nuTp-2);
   cfT  = @(t) besselk(nuTp/2,sqrt(nuTp)*abs(t)).*(sqrt(nuTp)*abs(t)).^(nuTp/2)./(gamma(nuTp/2)*2^(nuTp/2-1));
   cfP1 = @(t) cfSkewT(t/sqrt(varSkewT));
   cfPj = @(t) cfT(t/sqrt(varTp));
      
   % Determine ys and density of ys through convolution
   [ys, fYs] = DetermineYs1(cfP1, cfPj, ev, ew, us);

   % Determine PCs
   ps = ys * ev;
   
   % Determine densities for first PCs
   pdSkewT = @(y) 2^(lambda+1)/gamma(nuT/2)*delta^(-2*lambda)/sqrt(2*pi)*exp(beta*(y-mu)).*besselk(lambda-1/2,alpha*sqrt(delta^2 + (y - mu).^2))./((sqrt(delta^2+(y - mu).^2)/alpha).^(1/2-lambda));
   scaleP1  = 1/sqrt(ew(1))*sqrt(varSkewT);
   fPs(:,1) = pdSkewT(ps(:,1)*scaleP1)*scaleP1;  

   % Determine densities for other PCs
   for iPc = 2:numel(ew)
      scalePj = 1/sqrt(ew(iPc))*sqrt(varTp);
      fPs(:, iPc) = tpdf(ps(:,iPc)*scalePj,nuTp)*scalePj;
   end

   % Determine loglikelihood
   logLh = sum(sum(log(fPs)-log(fYs)));  
end
