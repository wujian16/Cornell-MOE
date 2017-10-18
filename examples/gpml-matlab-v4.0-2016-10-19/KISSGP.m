% Jian Wu, October 1st, 2017
function [results] = KISSGP(hypers)
sd = 3; rand('seed',sd), randn('seed',sd)       % set a seed for reproducability

a = 0.3; b = 1.2; f = @(x) a*x + b + sin(x);               % underlying function
n = 1e5; sn = 0.5;          % number of training points, noise standard deviation
x = 2*rand(n,1)-1; x = 1+4*x+sign(x); y = f(x)+sn*randn(n,1);      % sample data

cov = {@covSEiso}; hyp.cov = log([hypers(1); hypers(2)]);
mean = {@meanZero}; hyp.mean = [];
lik = {@likGauss};    hyp.lik = log(hypers(3)); inf = @infGaussLik;

ng = 1000; xg = linspace(-6,8,ng)'; covg = {@apxGrid,{cov},{xg}};% grid prediction
opt.cg_maxit = 500; opt.cg_tol = 1e-5; opt.pred_var = 100;          % parameters
inf = @(varargin) infGrid(varargin{:},opt);

[postg,nlZg,dnlZg] = infGrid(hyp,mean,covg,lik,x,y,opt);  % fast grid prediction
results = [nlZg; dnlZg.cov; dnlZg.lik]/n;
end
