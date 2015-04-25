function [ out ] = FWSumLogNormal( mu, sigma )
% mu: N-by-1 mean vector of the lognormal RVs
% sigma: N-by-N covariance matrix of the lognormal RVs
% ret: 2-by-1 vector, [mean; variance], of the merged lognormal 

% Based on Fenton-Wilkinson Approximation
% Fenton, L.F. (1960). The sum of log-normal probability distibutions in 
%    scattered transmission systems.
% IEEE Trans. Commun. Systems 8: 57-67.
% Correlated version from
% Pekka Pirinen. STATISTICAL POWER SUM ANALYSIS FOR NONIDENTICALLY 
%    DISTRIBUTED CORRELATED LOGNORMAL SIGNALS

% Copyright (c) 2011 Takaki Makino.

% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License http://www.gnu.org/licenses/gpl.html 
% for more details.

sz = size(mu, 1);

musigmad = mu + diag(sigma) * 0.5;

%firstmoment = exp( musigmad );
%sumfm = sum(firstmoment);
% logsumfm = log(sumfm);
logsumfm = log(sum(exp( musigmad )));

%muxsig = repmat( musigmad, 1, size(sigma, 2) );
%secondmoment = exp( muxsig + muxsig' + 0.5*(sigma + sigma') );
%sumsm = sum( sum( secondmoment ) );
% logsumsm = log(sumsm);
logsumsm = log( sum( sum( exp( repmat( musigmad, 1, sz ) + repmat( musigmad', sz, 1 ) + 0.5*(sigma + sigma') ) ) ) );

omu     = 2*logsumfm - 0.5*logsumsm;
osigma2 = logsumsm - 2*logsumfm;

out = [omu; osigma2];
end
