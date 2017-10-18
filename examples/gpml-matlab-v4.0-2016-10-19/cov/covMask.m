function [K,dK] = covMask(cov, hyp, x, z)

% Apply a covariance function to a subset of the dimensions only. The subset can
% either be specified by a 0/1 mask by a boolean mask or by an index set.
%
% This function doesn't actually compute very much on its own, it merely does
% some bookkeeping, and calls another covariance function to do the actual work.
%
% The function was suggested by Iain Murray, 2010-02-18 and is based on an
% earlier implementation of his dating back to 2009-06-16.
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2016-04-17.
%
% See also COVFUNCTIONS.M.

mask = fix(cov{1}(:));                    % either a binary mask or an index set
cov = cov(2);                                 % covariance function to be masked
if iscell(cov{:}), cov = cov{:}; end        % properly unwrap nested cell arrays
nh_string = feval(cov{:});    % number of hyperparameters of the full covariance

if max(mask)<2 && length(mask)>1, mask = find(mask); end    % convert 1/0->index
D = length(mask);                                             % masked dimension
if nargin<3, K = num2str(eval(nh_string)); return, end    % number of parameters
if nargin<4, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                       % determine mode

if eval(nh_string)~=length(hyp)                          % check hyperparameters
  error('number of hyperparameters does not match size of masked data')
end

xm = x(:,mask); if ~dg && ~xeqz, zm = z(:,mask); else zm = z; end
if nargout>1
  [K,dK] = feval(cov{:}, hyp, xm, zm);
  dK = @(Q) dirder(Q,dK,x,mask);
else
  K = feval(cov{:}, hyp, xm, zm);
end

function [dhyp,dx] = dirder(Q,dK,x,mask)
  if nargout>1
    [dhyp,dxm] = dK(Q); n = size(x,1);
    subs = [repmat((1:n)',length(mask),1), reshape(repmat(mask(:)',n,1),[],1)];
    dx = accumarray(subs,dxm(:),size(x));
  else
    dhyp = dK(Q);
  end