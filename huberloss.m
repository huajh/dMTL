
function [v, G] = huberloss(Z, Y, r)
%HUBERLOSS Huber loss function
%
%   v = HUBERLOSS(E, r);
%   v = HUBERLOSS(Z, Y, r);
%
%   [v, G] = HUBERLOSS(E, r);
%   [v, G] = HUBERLOSS(Z, Y, r);
%
%       Evaluates the huber loss function, defined to be
%
%           v = (1/2) * e^2,        when |e| < r
%             = r * e - r^2 / 2.    when |e| >= r
%
%       If e is a vector, then v is the sum of the loss values at
%       all components. The derivative is given by
%
%           g = e,                  when |e| < r
%             = r * sign(e),        when |e| >= r
%

% Created by Dahua Lin, on Jan 15, 2012
%

%% main

if nargin == 2
    E = Z;
    r = Y;
elseif nargin == 3
    E = Z - Y;
end

Ea = abs(E);
Eb = min(Ea, r);

v = 0.5 * Eb .* (2 * Ea - Eb);

if size(v, 1) > 1
    v = sum(v, 1);
end

if nargout >= 2
    G = Eb .* sign(E);
end
