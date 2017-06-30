function [ x ] = quad_kanpsack_singular( sigma, eta, h)
%QUAD_KANPSACK_BRUCKER Summary of this function goes here

%
% x = quad_kanpsack_brucker(sigma, eta, h) returns the vector x which is the solution
%   to the following constrained minimization problem:
%
%    min   sum_i^n sigma_i^2/(eta + x_i)
%    s.t.  sum_i^n x_i = h, 0<=x_i<=1
%           
%
% Author: Junhao Hua (huajh7@gmail.com) on July 27, 2015

%
% Reference:
%   Brucker, Peter. "An O (n) algorithm for quadratic knapsack problems."
%   Operations Research Letters 3.3 (1984): 163-166.
%
%   J. Chen, L. Tang, J. Liu, and J. Ye, ¡°A convex formulation for learning
%   a shared predictive structure from multiple tasks,¡± IEEE Transactions
%   on Pattern Analysis and Machine Intelligence, vol. 35, no. 5, pp. 1025¨C
%   1038, 2013.

% critical parameters: lower and upper endpoints, tu <= tl

% dimension 
n = numel(sigma);

% tu <= tl
all_tu = sigma.^2./(eta+1)^2;
all_tl = sigma.^2./eta^2;


% distinct critical parameter values ,Ascending
all_t = union(all_tl,all_tu);
r = size(all_t,1);

x = sol(sigma, eta,all_t(1));
max_z = sum(x);
if max_z == h
    % optimal solution
    return;
end
x = sol(sigma, eta,all_t(r));
min_z = sum(x);
if min_z == h
    % optimal solution
    return;
end

if max_z < h || min_z > h
    fprintf('No feasible solution exists');
    return;
end
tmin = all_t(1);
tmax = all_t(r);
I = (1:n)'; % index

opt_num = 0;
opt_den = h;
while (~isempty(I))
    tl = median(all_tl(I));    
    tu = median(all_tu(I(all_tl(I)>=tl)));
    for t = [tl,tu]
        if (tmin < t && t < tmax)
           x = sol(sigma, eta,t);
           z = sum(x);
           if z == h
               % optimal solution  x    
               return;       
           elseif z > h
               tmin = max(tmin,t);
           else
               tmax = min(tmax,t);
           end      
        end
    end
    k = size(I,1);
    cache = I;
    for j=1:k
        i = cache(j);
        if all_tl(i) <= tmin
            I = setdiff(I,i);
            % 0            
        end
        if tmax <= all_tu(i)
            I = setdiff(I,i);
            % 1            
            opt_den = opt_den - 1;
        end
        if all_tu(i) <= tmin && tmin <= tmax && tmax <= all_tl(i)
            I = setdiff(I,i);
            opt_num = opt_num + sigma(i);
            opt_den = opt_den + eta;
        end        
    end
end

opt = (opt_num/opt_den)^2;
x = sol(sigma,eta, opt);
end

function x = sol(sigma,eta, t)    
    x = min(1,max(0,sigma./sqrt(t) - eta));
end

