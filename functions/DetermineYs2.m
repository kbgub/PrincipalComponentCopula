function [ys, fYs] = DetermineYs2(cfP1, cfPj, ev, ew, us)        
   % Function to perform probability integral transform from u to y, when:
   %    - First PC is independent, while higher PCs have joint distribution
   %
   % This function was used in the following article:
   % Gubbels, K.B., Ypma, J.Y. & Oosterlee, C.W. (2025),
   % Principal Component Copulas for Capital Modelling and Systemic Risk, Computational Economics 
   % https://doi.org/10.1007/s10614-025-11051-7
   %
   % The function corresponds to the second case from lemma 1 in the article.
   %
   % Inputs:
   %    cfP1: univariate characteristic function of first PC with unit variance
   %    cfPj: multivariate characteristic function of higher PCs with unit variance
   %    ev:   eigenvectors of correlation matrix (direction of PCs)
   %    ew:   eigenvalues of correlation matrix  (variance of PCs)
   %    us:   uniform inputs to be transformed
   %
   % Note that input characteristic functions cfP1 and cfPj should have unit variance

   % Initialize
   ys  = zeros(size(us));
   fYs = zeros(size(us));
    
   % Determine ys and density of ys through convolution
   for iVar = 1:numel(ew)
      
      % Determine standard deviations
      sigmaP1 = sqrt(ew(1))*ev(iVar,1); 
      sigmaPj = sqrt(ev(iVar,2:end)*diag(ew(2:end))*ev(iVar,2:end)'); 
      
      % Perform convolution by multiplying characteristic functions
      cfY = @(t) cfP1(sigmaP1*t) .* cfPj(sigmaPj*t);

      % Use COS-method to determine density
      pdY = cf2QdCos(cfY, [], us(:,iVar));
      
      % Determine ys and density of ys
      ys(:,iVar)  = pdY.qYs;
      fYs(:,iVar) = pdY.fYs; 
   end   
end