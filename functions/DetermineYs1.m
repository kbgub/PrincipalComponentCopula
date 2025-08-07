function [ys, fYs] = DetermineYs1(cfP1,cfPj, ev, ew, us)        
   % Function to perform probability integral transform from u to y, when:
   %    - All PC's are independent and first PC has different distribution from higher PCs
   %
   % This function was used in the following article:
   % Gubbels, K.B., Ypma, J.Y. & Oosterlee, C.W. (2025),
   % Principal Component Copulas for Capital Modelling and Systemic Risk, Computational Economics 
   % https://doi.org/10.1007/s10614-025-11051-7
   %
   % The function corresponds to the first case from lemma 1 in the article.
   %
   % Inputs:
   %    cfP1: univariate characteristic function of first PC with unit variance
   %    cfPj: univariate characteristic function of higher PCs with unit variance
   %    ev:   eigenvectors of correlation matrix (direction of PCs)
   %    ew:   eigenvalues of correlation matrix  (variance of PCs)
   %    us:   uniform inputs to be transformed
   %
   % Note that input characteristic functions cfP1 and cfPj should have unit variance

   % Initialize
   ys  = zeros(size(us));
   fYs = zeros(size(us));
  
   % Parameters for COS-method (suitable only when density beyond -10 and 10 is neglible)
   a = -10;
   b = 10;
   nCos = 100;

   % Initialize grids for COS-method
   yGrid = a : (1/25) :b;
   kGrid = 1:nCos;
   sinGrid = sin(kGrid'.*(yGrid - a)*pi/(b-a));
   cosGrid = cos(kGrid'.*(yGrid - a)*pi/(b-a));

   % Determine ys and density of ys through convolution
   for iVar = 1:numel(ew)
      
      % Initialize characteristic function
      sigmaP1  = sqrt(ew(1))*ev(iVar,1);          
      cfY  = cfP1(kGrid*pi/(b-a)*sigmaP1);
      
      % Perform convolution by multiplying characteristic functions
      for iPc = 2:numel(ew)  
         sigmaPj = sqrt(ew(iPc))*ev(iVar,iPc); 
         cfY     = cfPj(kGrid*pi/(b-a)*sigmaPj).*cfY;
      end
   
      % Calculate expansion coefficients for COS-method
      cfY0  = 1;
      coefs = 2/(b-a)*real(cfY.*exp(-1i*kGrid*a*pi/(b-a)));
      coef0 = 2/(b-a)*cfY0; 
          
      % Calculate pdf and cdf
      pdfY = coef0/2 + coefs * cosGrid;
      cdfY = coef0/2*(yGrid-a) + (b-a)/pi*(coefs./kGrid) * sinGrid;
      pdfY = max(eps, pdfY);
      cdfY = min(max(eps, cdfY),1-eps);
      
      % Determine ys and density of ys
      [cdfU,idU] = unique(cdfY);
      pdY.qYs    = interp1(cdfU, yGrid(idU), us(:,iVar), 'pchip');
      ys(:,iVar) = pdY.qYs;
      fYs(:,iVar) = interp1(yGrid, pdfY, pdY.qYs, 'pchip'); 
   end   
end