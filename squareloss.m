function [cost, grad] = squareloss(f, y)
%QUADLOSS Quadratic loss function
%
%       Evaluates the quadratic loss, defined to be
%
%           cost = (1/2) * ||F - Y||^2.
%
%       Input arguments:
%       - f:        The vector of predicted vectors [n x 1]
%       - y:        The vector of expected vectors  [n x 1]
%
%       Output arguments:
%       - cost:       
%       - Grad:        The gradient vectors w.r.t. Z [n x 1]
%
%

% Created by Junhao Hua, on July 7, 2015
%

%% main

    if nargin == 1
        grad = f;
    else
        grad = f - y;
    end

    cost = 0.5*sum(grad.^2,1);
    
end    