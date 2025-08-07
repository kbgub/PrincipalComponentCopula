function [logLh, ys, fYs] = fLogLhPccHbN(x, ev, ew, us)
   % Function to obtain the loglikelihood of the hyperbolic-normal PCC 
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
   alpha = (x(1) + x(2))/2;
   beta  = (x(1) - x(2))/2;
   
   % Additional parameters
   gamma = sqrt(alpha^2 - beta^2);
   mu    = -beta/gamma*besselk(2,gamma)/besselk(1,gamma);
   varHB = 1/gamma*besselk(2,gamma)/besselk(1,gamma)+(beta^2/gamma^2)*(besselk(3,gamma)/besselk(1,gamma)-besselk(2,gamma)^2/besselk(1,gamma)^2);
        
   % Characteristic functions (scaled to unit variance for PCs)
   cfHB = @(t) exp(1i*mu*t).*sqrt(alpha^2 - beta^2).*besselk(1,sqrt(alpha^2 ...
     - (beta+1i*t).^2))./(sqrt(alpha^2 - (beta+1i*t).^2).*besselk(1,sqrt(alpha^2 - beta^2)));
   cfP1 = @(t) cfHB(t/sqrt(varHB));
   cfPj = @(t) exp(-t.^2/2);       
   
   % Determine ys and density of ys through convolution
   [ys, fYs] = DetermineYs2(cfP1, cfPj, ev, ew, us);

   % Determine PCs
   ps = ys * ev;
   
   % Determine densities for first PCs
   pdGH     = @(x) gamma*exp(-alpha*sqrt(1 + (x - mu).^2)+beta*(x-mu))./(2*alpha*besselk(1,gamma));
   scaleP1  = 1/sqrt(ew(1))*sqrt(varHB);
   fPs(:,1) = pdGH(ps(:,1)*scaleP1)*scaleP1;  
   
   % Determine densities for other PCs
   for iVar = 2:numel(ew)
      fPs(:, iVar) = normpdf(ps(:,iVar),0,sqrt(ew(iVar)));
   end

   % Determine loglikelihood
   logLh = sum(sum(log(fPs)-log(fYs)));  
end
