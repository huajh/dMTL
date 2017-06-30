function [v, G] = quadloss(Z, Y)
%QUADLOSS Quadratic loss function
%
%   v = QUADLOSS(D);
%   v = QUADLOSS(Z, Y);
%
%   [v, G] = QUADLOSS(D);
%   [v, G] = QUADLOSS(Z, Y);
%
%       Evaluates the quadratic loss, defined to be
%
%           v = (1/2) * ||z - y||^2.
%
%       Input arguments:
%       - D:        The matrix of difference vectors [D := Z - Y]
%       - Z:        The matrix of predicted vectors [d x n]
%       - Y:        The matrix of expected vectors [d x n]
%
%       Output arguments:
%       - v:        The vector of loss values [1 x n]
%       - G:        The gradient vectors w.r.t. Z [d x n]
%
%

% Created by Dahua Lin, on Jan 15, 2012
%

%% main

if nargin == 1
    G = Z;
else
    G = Z - Y;
end

if size(G, 1) == 1
    v = 0.5 * G.^2;
else
    v = 0.5 * sum(G.^2, 1);
end