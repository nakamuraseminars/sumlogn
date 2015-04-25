function [ ret ] = SYSumLogNormal( mu, sigma )
% mu: N-by-1 mean vector of the lognormal RVs
% sigma: N-by-N covariance matrix of the lognormal RVs
% ret: 2-by-1 vector, [mean; variance], of the merged lognormal 

% Based on Ho's Approximation for Schwartz-Yeh Approximation
% IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY, VOL. 44. NO. 4, NOVEMBER 1995 
% Calculating the Mean and Variance of Power Sums with Two Log-Normal Component
% Chia-Lu Ho

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


% prepare constants used in trape_integral
delta = 0.3;
xval = -3.6:delta:3.6;
uval = exp( -2 * sinh(xval) );
vval = 1 ./ (1+uval);
zval = (2 * delta) * vval .* uval .* cosh(xval);

% sort inputs 
[sortmu, idx] = sort(mu, 1, 'descend')%;
sortsigma = sigma(idx, idx)%;

% merge two-by-two
while size(sortmu,1) >= 2 
    
mw = sortmu(2) - sortmu(1);
sigy12 = sortsigma(1,1) ;
sigy1 = sqrt( sortsigma(1,1) );
sigy22 = sortsigma(2,2) ;
sigy2 = sqrt( sortsigma(2,2) );
rhoy1y2sigy1sigy2 = sortsigma(1,2) ;
sigw2 = sigy12 + sigy22 - 2*rhoy1y2sigy1sigy2;
sigw = sqrt( sigw2 );
% rhoy1y2 = rhoy1y2sigy1sigy2 / (sigy1 * sigy2)

eta = - mw / sigw;

fw = @(w)( exp(-(w - mw).^2/(2*sigw2) ) / sqrt(2*pi*sigw2) );

h0 = @(v)( exp( -0.5*(-log(v)+eta).^2 ) * sqrt(0.5/pi) );
h1 = @(v)( (fw(log(v))+fw(-log(v))).* log(1+v) );
h2 = @(v)( (fw(log(v))-fw(-log(v)))./ (1+1./v) );
h3 = @(v)( (fw(log(v))+fw(-log(v))).* log(1+v).^2 );
h4 = @(v)( -fw(-log(v)) .* log(v) .* log(1+v) );
h5 = @(v)( fw(-log(v)) ./ (1+1./v) );
h6 = @(v)( fw(-log(v)) .* log(1+v) );

h1234_old = @(v)( [h1(v); h2(v); h3(v); h4(v)] );
%hall = @(v)( [h1(v); h2(v); h3(v); h5(v); h6(v) ] );

I0 = erfc( sqrt(eta) ) %;

I = h1234(vval) * zval'%;
% I = trape_integral( @h1234 );

%    I1 = I(1);
%    I2 = I(2);
%    I3 = I(3);
%    I4 = I(4);
%    I5 = Iall(4);
%    I6 = Iall(5);
%    I4 = sigw2 * (fw(0) * log(2) - I5) + mw * I6;

A0 = sigw / sqrt(2 * pi) * exp( - eta^2 / 2 ) + mw * I0;

G1 = A0 + I(1);
G2 = I(3) + 2*I(4) + sigw2 * I0 + mw * A0;
G3 = sigw2 * (I(2) + I0);

mz = sortmu(1) + G1;
sigz2 =  sigy12 - G1^2 + G2 + 2*(rhoy1y2sigy1sigy2 - sigy12)*(G3 / sigw2);
sigz = sqrt( sigz2 );
diag3 = diag(sortsigma)%; 
diag3 = sqrt(diag3(3:size(diag3,1)))%;
rhos = sortsigma(3:size(sortsigma,1),1) ./ diag3%;
rhosy = sortsigma(3:size(sortsigma,1),2) ./ diag3%;
newrhos = rhos + (rhosy - rhos) * (G3 / sigw2)%;
newsig = newrhos .* diag3%;

newmu = vertcat( mz, sortmu(3:size(sortmu)))%;
newsig0 = sortsigma( 3:size(sortsigma,1), 3:size(sortsigma,2))%;
newsig1 = horzcat( sigz2, newsig')%;
newsig2 = horzcat( newsig, newsig0)%;
newsigma = vertcat( newsig1, newsig2 )%;

sortmu = newmu%;
sortsigma = newsigma%;

end

ret = [ sortmu ; sortsigma ];

%function [out] = trape_integral( func )
%       hval = func( vval );
%       out = hval * zval';
%end

    function [out] = h1234(v)
        logv = log(v);
        fwlogv = fw(logv);
        fwmlogv = fw(-logv);
        log1v = log(1+v);
        h1_ = (fwlogv+fwmlogv).* log1v;
        out = [ h1_ ;
                (fwlogv-fwmlogv)./ (1+1./v) ;
                h1_ .* log1v ;
                -fwmlogv .* log1v .* logv ];
    end

end
