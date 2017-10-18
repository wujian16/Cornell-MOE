function varargout = covMatern(mode, par, d, varargin)

% Matern covariance function with nu = d/2 and isotropic distance measure. For
% d=1 the function is also known as the exponential covariance function or the 
% Ornstein-Uhlenbeck covariance in 1d. The covariance function is:
%
%   k(x,z) = f( sqrt(d)*r ) * exp(-sqrt(d)*r)
%
% with f(t)=1 for d=1, f(t)=1+t for d=3 and f(t)=1+t+t^2/3 for d=5.
% Here r is the Mahalanobis distance sqrt(maha(x,z)). The function takes a
% "mode" parameter, which specifies precisely the Mahalanobis distance used, see
% covMaha. The function returns either the number of hyperparameters (with less
% than 3 input argments) or it returns a covariance matrix and (optionally) a
% derivative function.
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2016-05-23.
%
% See also covMaha.m.

if nargin < 1, error('Mode cannot be empty.'); end                  % no default
if nargin < 2, par = []; end                                           % default
varargout = cell(max(1, nargout), 1);                  % allocate mem for output
if nargin < 5, varargout{1} = covMaha(mode,par); return, end

if all(d~=[1,3,5]), error('only 1, 3 and 5 allowed for d'), end         % degree
switch d
  case 1, f = @(t) 1;               df = @(t) 1./t;     % df(t) = (f(t)-f'(t))/t
  case 3, f = @(t) 1 + t;           df = @(t) 1;
  case 5, f = @(t) 1 + t.*(1+t/3);  df = @(t) (1+t)/3;
end
          m = @(t,f) f(t).*exp(-t); dm = @(t,f) df(t).*exp(-t);

   k = @(d2)              m(sqrt(d*d2),f);
if d==1
  dk = @(d2,k) set_zero( -dm(sqrt(  d2),f)/2, d2==0 );    % fix limit case d2->0
else
  dk = @(d2,k)           -dm(sqrt(d*d2),f)*d/2;
end

[varargout{:}] = covMaha(mode, par, k, dk, varargin{:});
function A = set_zero(A,I), A(I) = 0;