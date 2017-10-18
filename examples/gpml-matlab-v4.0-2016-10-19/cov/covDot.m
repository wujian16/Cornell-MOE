function [K,dK,S] = covDot(mode, k, dk, hyp, x, z)

% Mahalanobis distance-based covariance function. The covariance function is
% parameterized as:
%
% k(x,z) = k(s), s = dot(x,z) = x'*inv(P)*z 
%
% where the P matrix is the metric. The hyperparameters are:
%
% hyp = [ log(ell) ]
%
% We offer three different modes:
%   'eye':   inv(P) = eye(D);        hyp = [];
%   'iso':   inv(P) = ell^2*eye(D);  hyp = [log(ell)];
%   'ard':   inv(P) = diag(ell.^2);  hyp = [log(ell_1); ..; log(ell_D)];
%
% For more help on design of covariance functions, try "help covFunctions".
%
% Note: dk(s) = d k / d s
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2016-04-27.
%
% See also COVFUNCTIONS.M.

if nargin<1, mode = 'eye'; end                                   % default value
if     isequal(mode,'ard'), ne = 'D';
elseif isequal(mode,'iso'), ne = '1';
elseif isequal(mode,'eye'), ne = '0';
else error('Parameter mode is either ''eye'', ''iso'' or ''ard''.'), end

if nargin<5, K = ne; return; end                   % report number of parameters
if nargin<6, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');             % sort out different types
[n,D] = size(x); ne = eval(ne);                                 % determine mode
hyp = hyp(:);                                 % make sure hyp is a column vector

ell = exp(hyp(1:ne));                              % characteristic length scale
if numel(ell)==0, ell = 1; end                                       % catch eye
A = @(x) bsxfun(@times,x,1./ell(:)'.^2);              % mvm with metric A=inv(P)

% compute dot product
if dg                                                               % vector sxx
  Az = A(x); S = sum(x.*Az,2);
else
  if xeqz                                                 % symmetric matrix Sxx
    Az = A(x);
  else                                                         % cross terms Sxz
    Az = A(z);
  end
  S = x*Az';
end
K = k(S);                                                           % covariance
if nargout > 1
  dK = @(Q) dirder(Q,S,dk,x,Az,dg,xeqz,mode);             % dir hyper derivative
end

function [dhyp,dx] = dirder(Q,S,dk,x,Az,dg,xeqz,mode)
  R = dk(S).*Q;
  if isequal(mode,'ard')
    if dg
      dhyp = -2*sum(x.*bsxfun(@times,R,Az),1)';
    else
      dhyp = -2*sum(x.*(R*Az),1)';
    end
  elseif isequal(mode,'iso')
    dhyp = -2*R(:)'*S(:);
  else
    dhyp = zeros(0,1);
  end
  if nargout > 1
    if xeqz, dx = R*Az+R'*Az; else dx = R*Az; end
  end