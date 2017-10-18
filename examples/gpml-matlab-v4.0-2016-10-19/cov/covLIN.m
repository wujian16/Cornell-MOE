function [K,dK] = covLIN(mode,hyp,x,z)

% Linear covariance function.
% The covariance function is parameterized as:
%
% k(x,z) = dot(x,z)
%
% where dot(x,z) is a dot product. The hyperparameters are:
%
% hyp = [ hyp_maha ]
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2016-04-26.
%
% See also covDot.m.

if nargin<1, mode = 'eye'; end, narg = nargin;                    % default mode
if ~ischar(mode)                                     % compatible to old version
  if nargin>2, z = x; end
  if nargin>1, x = hyp; end
  if nargin>0, hyp = mode; end
  mode = 'eye'; narg = narg+1;
end
if narg<3, K = covDot(mode); return, end
if narg<4, z = []; end                                     % make sure, z exists

k = @(s) s; dk = @(s) ones(size(s));

if nargout > 1
  [K,dK] = covDot(mode,k,dk,hyp,x,z);
else
  K = covDot(mode,k,dk,hyp,x,z);
end