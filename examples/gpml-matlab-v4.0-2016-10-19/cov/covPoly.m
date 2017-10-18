function [K,dK] = covPoly(mode, d, hyp, x, z)

% Polynomial covariance function. The covariance function is parameterized as:
%
% k(x,z) = sf^2 * ( c + s )^d , where s = x*inv(P)*z is the dot product
%
% The hyperparameters are:
%
% hyp = [ hyp_dot
%         log(c)
%         log(sf)  ]
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2016-04-23.
%
% See also covDot.m.

if nargin<1, mode = 'eye'; end                                    % default mode
if ~ischar(mode)                                     % compatible to old version
  if nargin>3, z = x; end
  if nargin>2, x = hyp; end
  if nargin>1, hyp = d; end
  if nargin>0, d = mode; end
  mode = 'eye'; narg = nargin+1;
else
  narg = nargin;
end

if narg<4, K = [covDot(mode),'+2']; return, end    % report number of parameters
if narg<5, z = []; end                                     % make sure, z exists
[n,D] = size(x);                                                    % dimensions
ne = eval(covDot(mode));
c = exp(hyp(ne+1));                                       % inhomogeneous offset
sf2 = exp(2*hyp(ne+2));                                        % signal variance
if d~=max(1,fix(d)), error('only nonzero integers allowed for d'), end  % degree

k = @(s) (c+s).^d; dk = @(s) d*(c+s).^(d-1);

if nargout > 1
  [K,dK0] = covScale({'covDot',mode,k,dk},hyp([1:ne,ne+2]),x,z);
  S = covDot(mode,@(s)s,[],hyp(1:ne),x,z);
  dK = @(Q) dirder(Q,S,dK0,c,d,sf2,ne);
else
  K = covScale({'covDot',mode,k,dk},hyp([1:ne,ne+2]),x,z);
end

function [dhyp,dx] = dirder(Q,S,dK0,c,d,sf2,ne)
  if nargout > 1, [dhyp,dx] = dK0(Q); else dhyp = dK0(Q); end
  dhyp = [dhyp(1:ne); c*d*sf2*(Q(:)'*(c+S(:)).^(d-1)); dhyp(ne+1)];   % insert c