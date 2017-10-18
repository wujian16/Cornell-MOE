function K = apx(hyp,cov,x,opt)

% (Approximate) linear algebra operations involving the covariance matrix K.
%
% A) Exact covariance computations.
%    There are no parameters in this mode.
%    Depending on the sign of W, we switch between
%    - the symmetric Cholesky mode [1], where B = I + sqrt(W)*K*sqrt(W), and
%    - the asymmetric LU mode [2],      where B = I + K*W.
%    Note that (1) is faster but it is only applicable if all W are positive.
%
% B) Sparse covariance approximations aka FITC [4], VFE [5] and SPEP [3].
%    We have a parameter opt.s from [0,1], the power in sparse power EP [3] 
%    interpolating between the Fully Independent Training Conditionals (FITC)
%    approach [4] and a Variational Free Energy (VFE) approximation [5].
%    In particular:
%  opt.s, default is 1.0 for FITC, opt.s = 0.0 corresponds to VFE.
%    Please see cov/apxSparse.m for details.
%
% C) Grid-based covariance approximations aka KISS-GP [6].
%    Please see cov/apxGrid.m for further details and more parameters.
%  opt.cg_tol,   default is 1e-6      as in Matlab's pcg function
%  opt.cg_maxit, default is min(n,20) as in Matlab's pcg function
%    The conjugate gradient-based linear system solver has two adjustable
%    parameters, the relative residual threshold for convergence opt.cg_tol and
%    the maximum number of MVMs opt.cg_maxit until the process stops.
%  opt.deg,      default is 3         degree of interpolation polynomial
%    For equispaced axes, opt.deg sets the degree of the interpolation
%    polynomial along each of the p axes. Here 0 means nearest neighbor,
%    1 means linear interpolation, and 3 uses a cubic.
%    For non-equispaced axes, only linear interpolation with inverse distance
%    weighting is offered and opt.deg is ignored.
%  opt.ldB2_cheby = true employs Monte-Carlo trace estimation aka the Hutchinson
%    method and Chebyshev polynomials to approximate the term log(det(B))/2
%    stochastically, see [7]. The following four parameters configure different
%    aspects of the estimator and are only valid if opt.ldB2_cheby equals true.
%  opt.ldB2_cheby_hutch,  default is 10, number of samples for the trace estim
%  opt.ldB2_cheby_degree, default is 15, degree of Chebyshev approx polynomial
%  opt.ldB2_cheby_maxit,  default is 50, max # of MVMs to estimate max(eig(B))
%  opt.ldB2_cheby_seed,   default is [], random seed for the stoch trace estim
%  opt.stat = true returns a little bit of output summarising the exploited
%    structure of the covariance of the grid.
%    The log-determinant approximation employs Fiedler's 1971 inequality and a
%    rescaled version of the eigenspectrum of the covariance evaluated on the
%    complete grid.
%
% The call K = apx(hyp,cov,x,opt) yields a structure K with a variety of
% fields.
% 1) Matrix-vector multiplication with covariance matrix
%    K.mvm(x) = K*x
% 2) Projection and its transpose (unity except for mode B) Sparse approx.)
%    post.alpha = K.P(solveKiW(f-m))
%    post.L = L = @(r) -K.P(solveKiW(K.Pt(r)))
% 3) Linear algebra functions depending on W
%    [ldB2,solveKiW,dW,dldB2,L] = K.fun(W)
%   a) Log-determinant (approximation), called f in the sequel
%      ldB2 = log(det(B))/2
%   b) Solution of linear systems
%      solveKiW(r) = (K+inv(W)) \ r
%   c) Log-determinant (approximation) derivative w.r.t. W
%      dW = d f / d W, where f = ldB2(W), exact value dW = diag(inv(B)*K)/2
%   d) Log-determinant (approximation) derivative w.r.t. hyperparameters
%      dhyp = dldB2(alpha,a,b)
%      Q = d f / d K, exact value would be Q = inv(K+inv(W))
%      R = alpha*alpha' + 2*a*b'
%      Here dhyp(i) = tr( (Q-R)'*dKi )/2, where dKi = d K / d hyp(i).
%   e) Matrix (operator) to compute the predictive variance
%      L = -K.P(solveKiW(K.Pt(r))) either as a dense matrix or function.
%      See gp.m for details on post.L.
% [1] Seeger, GPs for Machine Learning, sect. 4, TR, 2004.
% [2] Jylanki, Vanhatalo & Vehtari, Robust GPR with a Student's-t
%     Likelihood, JMLR, 2011.
% [3] Bui, Yan & Turner, A Unifying Framework for Sparse GP Approximation
%     using Power EP, 2016, https://arxiv.org/abs/1605.07066.
% [4] Snelson & Ghahramani, Sparse GPs using pseudo-inputs, NIPS, 2006.
% [5] Titsias, Var. Learning of Inducing Variables in Sparse GPs, AISTATS, 2009
% [6] Wilson & Nickisch, Kernel Interp. for Scalable Structured GPs, ICML, 2015
% [7] Han, Malioutov & Shin,  Large-scale Log-det Computation through Stochastic
%     Chebyshev Expansions, ICML, 2015.
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch 2016-10-28.
%
% See also apxSparse.m, apxGrid.m, gp.m.

if nargin<4, opt = []; end                           % make sure variable exists
if isnumeric(cov), c1 = 'numeric'; else c1 = cov{1}; end         % detect matrix
if isa(c1, 'function_handle'), c1 = func2str(c1); end         % turn into string
sparse = strcmp(c1,'apxSparse') || strcmp(c1,'covFITC');
grid   = strcmp(c1,'apxGrid')   || strcmp(c1,'covGrid');
exact = ~grid && ~sparse;

if exact                   % A) Exact computations using dense matrix operations
  if strcmp(c1,'numeric'), K = cov; dK = [];           % catch dense matrix case
  else
    [K,dK] = feval(cov{:},hyp.cov,x);     % covariance matrix and dir derivative
  end
  K = struct('mvm',@(x)mvmK_exact(K,x), 'P',@(x)x, 'Pt',@(x)x,... % mvm and proj 
             'fun',@(W)ldB2_exact(W,K,dK));

elseif sparse                                         % B) Sparse approximations
  if isfield(opt,'s'), s = opt.s; else s = 1.0; end            % default is FITC
  if isfield(hyp,'xu'), cov{3} = hyp.xu; end   % hyp.xu provided, replace cov{3}
  xu = cov{3}; nu = size(xu,1);                        % extract inducing points
  [Kuu,   dKuu]   = feval(cov{2}{:}, hyp.cov, xu);     % get the building blocks
  [Ku,    dKu]    = feval(cov{2}{:}, hyp.cov, xu, x);
  [diagK, ddiagK] = feval(cov{2}{:}, hyp.cov, x, 'diag');
  snu2 = 1e-6*(trace(Kuu)/nu);                 % stabilise by 0.1% of signal std
  Luu  = chol(Kuu+snu2*eye(nu));                       % Kuu + snu2*I = Luu'*Luu
  V  = Luu'\Ku;                                   % V = inv(Luu')*Ku => V'*V = Q
  g = max(diagK-sum(V.*V,1)',0);                         % g = diag(K) - diag(Q)
  K.mvm = @(x) V'*(V*x) + bsxfun(@times,s*g,x);   % efficient matrix-vector mult
  K.P = @(x) Luu\(V*x); K.Pt = @(x) V'*(Luu'\x);         % projection operations
  xud = isfield(hyp,'xu');      % flag deciding whether to compute hyp.xu derivs
  K.fun = @(W) ldB2_sparse(W,V,g,Luu,dKuu,dKu,ddiagK,s,xud);

elseif grid                                            % C)  Grid approximations
  n = size(x,1);
  if isfield(opt,'cg_tol'), cgtol = opt.cg_tol;       % stop conjugate gradients
  else cgtol = 1e-6; end                                        % same as in pcg
  if isfield(opt,'cg_maxit'), cgmit = opt.cg_maxit;    % number of cg iterations
  else cgmit = min(n,20); end                                   % same as in pcg
  if isfield(opt,'deg'), deg = opt.deg; else deg = 3; end     % interpol. degree
  if isfield(opt,'stat'), stat = opt.stat; else stat = false; end    % show stat
  cgpar = {cgtol,cgmit}; xg = cov{3}; p = numel(xg);  % conjugate gradient, grid
  if isfield(opt,'ldB2_cheby'),cheby=opt.ldB2_cheby; else cheby=false; end
  if isfield(opt,'ldB2_cheby_hutch'),m=opt.ldB2_cheby_hutch; else m=10; end
  if isfield(opt,'ldB2_cheby_degree'),d=opt.ldB2_cheby_degree; else d=15; end
  if isfield(opt,'ldB2_cheby_maxit'),mit=opt.ldB2_cheby_maxit; else mit=50; end
  if isfield(opt,'ldB2_cheby_seed'),sd=opt.ldB2_cheby_seed; else sd=[]; end
  ldpar = {cheby,m,d,mit,sd};                                % logdet parameters
  [Kg,Mx] = feval(cov{:},hyp.cov,x,[],deg);  % grid cov structure, interp matrix
  if stat    % show some information about the nature of the p Kronecker factors
    fprintf(apxGrid('info',Kg,Mx,xg,deg));
  end
  K.mvm = @(x) Mx*Kg.mvm(Mx'*x);                    % mvm with covariance matrix
  K.P = @(x)x; K.Pt = @(x)x;                             % projection operations
  K.fun = @(W) ldB2_grid(W,K,Kg,xg,Mx,cgpar,ldpar);
end

%% A) Exact computations using dense matrix operations =========================
function [ldB2,solveKiW,dW,dldB2,L] = ldB2_exact(W,K,dK)
  isWneg = any(W<0); n = numel(W);
  if isWneg                  % switch between Cholesky and LU decomposition mode
    A = eye(n) + bsxfun(@times,K,W');                     % asymmetric A = I+K*W
    [L,U,P] = lu(A); u = diag(U);         % compute LU decomposition, A = P'*L*U
    signU = prod(sign(u));                                           % sign of U
    detP = 1;               % compute sign (and det) of the permutation matrix P
    p = P*(1:n)';
    for i=1:n                                                     % swap entries
      if i~=p(i), detP = -detP; j = find(p==i); p([i,j]) = p([j,i]); end
    end
    if signU~=detP     % log becomes complex for negative values, encoded by inf
      ldB2 = Inf;
    else          % det(L) = 1 and U triangular => det(A) = det(P)*prod(diag(U))
      ldB2 = sum(log(abs(u)))/2;
    end                                            % compute inverse if required
    if nargout>1, Q = U\(L\P); solveKiW = @(r) bsxfun(@times,W,Q*r); end
    if nargout>4, L = -diag(W)*Q; end                              % overwrite L
  else                                                 % symmetric B = I+sW*K*sW
    sW = sqrt(W); L = chol(eye(n)+sW*sW'.*K);             % Cholesky factor of B
    ldB2 = sum(log(diag(L)));                                    % log(det(B))/2
    solveKiW = @(r) bsxfun(@times,solve_chol(L,bsxfun(@times,r,sW)),sW);
    if nargout>2, Q = bsxfun(@times,1./sW,solve_chol(L,diag(sW))); end
  end
  if nargout>2
    dW = sum(Q.*K,2)/2;            % d log(det(B))/2 / d W = diag(inv(inv(K)+W))
    dldB2 = @(varargin) ldB2_deriv_exact(W,dK,Q, varargin{:});     % derivatives
  end

function dhyp = ldB2_deriv_exact(W,dK,Q, alpha,a,b)
  if nargin>3, R = alpha*alpha'; else R = 0; end
  if nargin>5, R = R + 2*a*b'; end
  dhyp.cov = dK( bsxfun(@times,Q,W) - R )/2;

function z = mvmK_exact(K,x)
  if size(x,2)==size(x,1) && max(max( abs(x-eye(size(x))) ))<eps      % x=eye(n)
    z = K;                             % avoid O(n^3) operation as it is trivial
  else
    z = K*x;
  end

%% B) Sparse approximations ====================================================
function [ldB2,solveKiW,dW,dldB2,L]=ldB2_sparse(W,V,g,Luu,dKuu,dKu,ddiagK,s,xud)
  z = s*g.*W; t = 1/s*log(z+1); i = z<1e-4;  % s=0: t = g*W, s=1: t = log(g*W+1)
  t(i) = g(i).*W(i).*(1-z(i)/2+z(i).^2/3);         % 2nd order Taylor for tiny z
  dt = 1./(z+1); d = W.*dt;                               % evaluate derivatives
  nu = size(Luu,1); Vd = bsxfun(@times,V,d');
  Lu = chol(eye(nu) + V*Vd'); LuV = Lu'\V;               % Lu'*Lu=I+V*diag(d)*V'
  ldB2 = sum(log(diag(Lu))) + sum(t)/2;    % s=1 => t=log(g.*W+1), s=0 => t=g.*W
  md = @(r) bsxfun(@times,d,r); solveKiW = @(r) md(r) - md(LuV'*(LuV*md(r)));
  if nargout>2                % dW = d log(det(B))/2 / d W = diag(inv(inv(K)+W))
    dW = sum(LuV.*((LuV*Vd')*V),1)' + s*g.*d.*sum(LuV.*LuV,1)';
    dW = dt.*(g+sum(V.*V,1)'-dW)/2;                % add trace "correction" term
    dldB2 = @(varargin) ldB2_deriv_sparse(V,Luu,d,LuV,dKuu,dKu,ddiagK,s,xud,...
                                                                   varargin{:});
    if nargout>4
      L = solve_chol(Lu*Luu,eye(nu))-solve_chol(Luu,eye(nu));   % Sigma-inv(Kuu)
    end
  end

function dhyp = ldB2_deriv_sparse(V,Luu,d,LuV,dKuu,dKu,ddiagK,s,xud,alpha,a,b)
  % K + 1./W = V'*V + inv(D), D = diag(d)
  % Q = inv(K+inv(W)) = inv(V'*V + diag(1./d)) = diag(d) - LuVd'*LuVd;
  LuVd = bsxfun(@times,LuV,d'); diagQ = d - sum(LuVd.*LuVd,1)';
  F = Luu\V; Qu = bsxfun(@times,F,d') - (F*LuVd')*LuVd;
  if nargin>9, diagQ = diagQ-alpha.*alpha; Qu = Qu-(F*alpha)*alpha'; end
  Quu = Qu*F';
  if nargin>11
    diagQ = diagQ-2*a.*b; Qu = Qu-(F*a)*b'-(F*b)*a'; Quu = Quu-2*(F*a)*(F*b)';
  end
  diagQ = s*diagQ + (1-s)*d;                          % take care of s parameter
  Qu = Qu - bsxfun(@times,F,diagQ'); Quu = Quu - bsxfun(@times,F,diagQ')*F';
  nu = size(Quu,1); Quu = Quu + 1e-6*trace(Quu)/nu*eye(nu);
  if xud
    dhyp.cov = ddiagK(diagQ)/2; dhyp.xu = 0;
    [dc,dx] = dKu(Qu);   dhyp.cov = dhyp.cov + dc;   dhyp.xu = dhyp.xu + dx;
    [dc,dx] = dKuu(Quu); dhyp.cov = dhyp.cov - dc/2; dhyp.xu = dhyp.xu - dx/2;
  else
    dhyp.cov = ddiagK(diagQ)/2 + dKu(Qu) - dKuu(Quu)/2;
  end


%% C)  Grid approximations =====================================================
function [ldB2,solveKiW,dW,dldB2,L] = ldB2_grid(W,K,Kg,xg,Mx,cgpar,ldpar)
  if all(W>=0)                                 % well-conditioned symmetric case
    sW = sqrt(W); msW = @(x) bsxfun(@times,sW,x);
    mvmB = @(x) msW(K.mvm(msW(x)))+x;
    solveKiW = @(r) msW(linsolve(msW(r),mvmB,cgpar{:}));
  else                 % less well-conditioned symmetric case if some negative W
    mvmKiW = @(x) K.mvm(x)+bsxfun(@times,1./W,x);
    solveKiW = @(r) linsolve(r,mvmKiW,cgpar{:});
  end                                                   % K*v = Mx*Kg.mvm(Mx'*v)
  dhyp.cov = [];                                                          % init
  if ldpar{1}                 % stochastic estimation of logdet cheby/hutchinson
    dK = @(a,b) apxGrid('dirder',Kg,xg,Mx,a,b);
    if nargout<3            % save some computation depending on required output
      ldB2 = logdet_sample(W,K.mvm,dK, ldpar{2:end});
    else
      [ldB2,emax,dhyp.cov,dW] = logdet_sample(W,K.mvm,dK, ldpar{2:end});
    end
  else
    s = 3;                                    % Whittle embedding overlap factor
    [V,ee,e] = apxGrid('eigkron',Kg,xg,s);         % perform eigen-decomposition
    [ldB2,de,dW] = logdet_fiedler(e,W);     % Fiedler's upper bound, derivatives
    de = de.*double(e>0); % chain rule of max(e,0) in eigkron, Q = V*diag(de)*V'
    if nargout>3, dhyp.cov = ldB2_deriv_grid_fiedler(Kg,xg,V,ee,de,s); end
  end
  dldB2 = @(varargin) ldB2_deriv_grid(dhyp, Kg,xg,Mx, varargin{:});
  if ~isreal(ldB2), error('Too many negative W detected.'), end
  L = @(r) -K.P(solveKiW(K.Pt(r)));

function dhyp = ldB2_deriv_grid_fiedler(Kg,xg,V,ee,de,s)
  p = numel(Kg.kron);                              % number of Kronecker factors
  ng = [apxGrid('size',xg)',1];                                 % grid dimension
  dhyp = [];                              % dhyp(i) = trace( V*diag(de)*V'*dKi )
  for i=1:p
    z = reshape(de,ng); Vi = V{i};
    for j=1:p, if i~=j, z = apxGrid('tmul',ee{j}(:)',z,j); end, end
    if isnumeric(Vi)
      Q = bsxfun(@times,Vi,z(:)')*Vi';
      dhci = Kg.kron(i).dfactor(Q);
    else
      kii = Kg.kron(i).factor.kii;
      [junk,ni] = apxGrid('expand',xg{i}); di = numel(ni);
      xs = cell(di,1);                % generic (Strang) circular embedding grid
      for j=1:di, n2 = floor(ni(j)-1/2)+1; xs{j} = [1:n2,n2-2*ni(j)+2:0]'; end
      Fz = real(fftn(reshape(z,[ni(:)',1])));  % imaginary part of deriv is zero
      rep = 2*s*ones(1,di); if di==1, rep = [rep,1]; end           % replication
      Fzw = repmat(Fz,rep); % replicate i.e. perform transpose of circ operation
      [junk,xw] = apxGrid('circ',kii,ni,s);    % get Whittle circ embedding grid
      [junk,dkwi] = kii(apxGrid('expand',xw));             % evaluate derivative
      dhci = dkwi(Fzw(:));
    end
    dhyp = [dhyp; dhci];
  end

function dhyp = ldB2_deriv_grid(dhyp, Kg,xg,Mx, alpha,a,b)
  if nargin>4, dhyp.cov = dhyp.cov-apxGrid('dirder',Kg,xg,Mx,alpha,alpha)/2; end
  if nargin>6, dhyp.cov = dhyp.cov-apxGrid('dirder',Kg,xg,Mx,a,b);           end

function q = linsolve(p,mvm,varargin) % solve q = mvm(p) via conjugate gradients
  [q,flag,relres,iter] = conjgrad(mvm,p,varargin{:});                 % like pcg
  if ~flag,error('Not converged after %d iterations, r=%1.2e\n',iter,relres),end

% Solve x=A*b with symmetric A(n,n), b(n,m), x(n,m) using conjugate gradients.
% The method is along the lines of PCG but suited for matrix inputs b.
function [x,flag,relres,iter,r] = conjgrad(A,b,tol,maxit)
if nargin<3, tol = 1e-10; end
if nargin<4, maxit = min(size(b,1),20); end
x0 = zeros(size(b)); x = x0;
if isnumeric(A), r = b-A*x; else r = b-A(x); end, r2 = sum(r.*r,1); r2new = r2;
nb = sqrt(sum(b.*b,1)); flag = 0; iter = 1;
relres = sqrt(r2)./nb; todo = relres>=tol; if ~any(todo), flag = 1; return, end
on = ones(size(b,1),1); r = r(:,todo); d = r;
for iter = 2:maxit
  if isnumeric(A), z = A*d; else z = A(d); end
  a = r2(todo)./sum(d.*z,1);
  a = on*a;
  x(:,todo) = x(:,todo) + a.*d;
  r = r - a.*z;
  r2new(todo) = sum(r.*r,1);
  relres = sqrt(r2new)./nb; cnv = relres(todo)<tol; todo = relres>=tol;
  d = d(:,~cnv); r = r(:,~cnv);                           % get rid of converged
  if ~any(todo), flag = 1; return, end
  b = r2new./r2;                                               % Fletcher-Reeves
  d = r + (on*b(todo)).*d;
  r2 = r2new;
end

% Upper determinant bound on log |K*diag(W)+I| using Fiedler's 1971 inequality.
% K = kron( kron(...,K{2}), K{1} ), W = diag(w) both symmetric psd.
% The bound is exact for W = w*ones(N,1). Here, E = eig(K) are the
% eigenvalues of the matrix K.
% See also Prob.III.6.14 in Matrix Analysis, Bhatia 1997.
%
% Given nxn spd matrices C and D with ordered nx1 eigenvalues c, d 
% then det(C+D) <= prod(c+flipud(d))=exp(ub).
function [ub,dE,dW,ie,iw] = logdet_fiedler(E,W)
  [E,ie] = sort(E,'descend'); [W,iw] = sort(W,'descend');         % sort vectors
  N = numel(E); n = numel(W); k = n/N*E; % dimensions, approximate spectrum of K
  if n>N, k = [k;zeros(n-N,1)]; else k = k(1:n); end  % extend/shrink to match W
  kw1 = k.*W+1; ub = sum(log(kw1))/2;                     % evaluate upper bound
  dW = zeros(n,1); dW(iw) = k./(2*kw1); m = min(n,N);      % derivative w.r.t. W
  dE = zeros(N,1); dE(ie(1:m)) = W(1:m)./(N*2/n*kw1(1:m));  % deriative w.r.t. E

% Approximate l = log(det(B))/2, where B = I + sqrt(W)*K*sqrt(W) and compute
% the derivatives d l / d hyp w.r.t. covariance hyperparameters and
% d l / d W the gradient w.r.t. the precision matrix W.
%
% Large-scale Log-det Computation through Stochastic Chebyshev Expansions
% Insu Han, Dmitry Malioutov, Jinwoo Shin, ICML, 2015.
%
% Chebyshev polynomials T[0],..,T[d], where T[0](x)=1, T[1](x)=x, and
%                                           T[i+1](x)=2*x*T[i](x)-T[i-1](x).
% dT[0](x)=0, dT[1](x)=1, dT[i+1](x)=2*T[i](x)+2*x*dT[i](x)-dT[i-1](x)
%
% W       is the precision matrix
% K(z)    yields mvm K*z
% dK(y,z) yields d y'*K*z / d theta
%
% Copyright (c) by Insu Han and Hannes Nickisch 2016-09-27.
function [ldB2,emax,dhyp,dW] = logdet_sample(W,K,dK, m,d, maxit,emax,seed)
  sW = sqrt(W); n = numel(W); emin = 1;          % size and min eigenvalue bound
  B = @(x) x + bsxfun(@times,sW, K(bsxfun(@times,sW,x) ));%B=I+sqrt(W)*K*sqrt(W)
  if nargin<6, maxit = 50; end
  if nargin<7 || numel(emax)~=1                % evaluate upper eigenvalue bound
    if n==1, emax = B(1); else
      opt.maxit = maxit;   % number of iterations to estimate the largest eigval
      opt.issym = 1; opt.isreal = 1; % K is real symmetric and - of course - psd
      cstr = 'eigs(B,n,1,''lm'',opt)';              % compute largest eigenvalue
      if exist('evalc'), [txt,emax] = evalc(cstr); else emax = eval(cstr); end
    end
  end
  if nargin<5, d = 15; end
  if nargin<4, m = 10; end
  d = round(abs(d)); if d==0, error('We require d>0.'), end
  a = emin+emax; del = 1-emax/a; ldB2 = n*log(a)/2;% scale eig(B) to [del,1-del]
  if emax>1e5
    fprintf('B has large condition number %1.2e\n',emax)
    fprintf('log(det(B))/2 is most likely overestimated\n')
  end
  sf = 2/a/(1-2*del); B = @(x) sf*B(x) - 1/(1-2*del)*x;% apply scaling transform

  xk = cos(pi*((0:d)'+0.5)/(d+1));                              % zeros of Tn(x)
  fk = log(((1-2*del)/2).*xk+0.5);        % target function, [-1,1]->[del,1-del]
  Tk = ones(d+1,d+1); Tk(:,2) = xk;                             % init recursion
  for i=2:d, Tk(:,i+1) = 2*xk.*Tk(:,i) - Tk(:,i-1); end   % evaluate polynomials
  c = 2/(d+1)*(Tk'*fk); c(1) = c(1)/2;          % compute Chebyshev coefficients
  if nargin>7 && numel(seed)>0, randn('seed',seed), end               % use seed
  V = sign(randn(n,m)); dhyp = 0; dW = 0; % bulk draw Rademacher variables, init
  p1 = [1; zeros(d,1)]; p2 = [0;1;zeros(d-1,1)]; % Chebyshev->usual coefficients
  p = c(1)*p1 + c(2)*p2;
  for i=2:d, p3 = [0;2*p2(1:d)]-p1; p = p + c(i+1)*p3; p1 = p2; p2 = p3; end
  if nargout<3                       % use bulk mvms with B, one for each j=1..d
    U = p(1)*V; BjV = V; for j=1:d, BjV = B(BjV); U = U + p(j+1)*BjV; end
    ldB2 = ldB2 + sum(sum(V.*U))/(2*m);
  else                               % deal with one Rademacher vector at a time
    Bv = zeros(n,d+1);                            % all powers Bv(:,j) = (B^j)*v
    for i=1:m
      v = V(:,i); Bv(:,1) = v;
      for j=1:d, Bv(:,j+1) = B(Bv(:,j)); end
      ldB2 = ldB2 + (v'*Bv*p)/(2*m);
      sWBv = bsxfun(@times,sW,Bv);
      for j=1:d                   % p(1)*I + p(2)*B + p(3)*B^2 + .. + p(d+1)*B^d
        for k=1:ceil(j/2)
          akj = sf*p(j+1)/m; if k==ceil(j/2) && mod(j,2)==1, akj = akj/2; end
          dhyp = dhyp + akj * dK(sWBv(:,j-k+1),sWBv(:,k));
        end
      end
      if nargout>3, u = bsxfun(@times,1./sW,Bv); w = K(sWBv);       % precompute
        for j=1:d
          dWj = sum( u(:,j:-1:1).*w(:,1:j) + w(:,j:-1:1).*u(:,1:j), 2 );
          dW = dW + sf*p(j+1)/(4*m) * dWj;
        end
      end
    end
  end