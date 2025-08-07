function pdY = cf2QdCos(cfY, ys, us)
   % Function to obtain probability densities, distribution function and
   % quantiles from the characteristic function using the COS method 
   %
   % This function was used in the following article:
   % Gubbels, K.B., Ypma, J.Y. & Oosterlee, C.W. (2025),
   % Principal Component Copulas for Capital Modelling and Systemic Risk, Computational Economics 
   % https://doi.org/10.1007/s10614-025-11051-7
   %
   %    cfY:  characteristic function of distribution
   %    ys:   inputs for which pdf and cdf are determined
   %    us:   uniform inputs to be transformed to (inverse) quantile

   % Parameters (suitable only when densities beyond -10 and 10 are neglible)
   a = -10;
   b = 10;
   nCos = 100;

   % Initialize grids
   yGrid = a : (1/25) :b;
   kGrid = 1:nCos;
   sinGrid = sin(kGrid'.*(yGrid - a)*pi/(b-a));
   cosGrid = cos(kGrid'.*(yGrid - a)*pi/(b-a));

   % Calculate expansion coefficients
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