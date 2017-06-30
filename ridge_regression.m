function [ u ] = ridge_regression( X,y,lambda )
%RIDGE_REGRESSION Summary of this function goes here
%   Detailed explanation goes here

    maxiter = 500;
    [nsample,nfeat] = size(X);
    u = zeros(nfeat,1);    
    cost = 0;
    costs = [];
   % lambda = 0.05;
    tau = 0.05;
    for t=1:maxiter
        
        [cost,grad] = cost_func(u,X,y,lambda);
        
        u = u - tau*grad; 
        
        costs = [costs,cost];
        
        if t>200 && abs(costs(t) - costs(t-1))/costs(t-1) < 1e-8
            break;
        end           
    end
    plot(1:numel(costs),costs,'-');
    title('cost');
end


function [J,grad] = cost_func(w,X,y,lambda)
%   
%   minimize the regularized squared loss
    
    [nsample,nfeat] = size(X);
    J = 1/(2*nsample)*sum(sum((X*w-y).^2)) + lambda/2*sum(sum(w.^2));
    
    grad = 1./nsample*X'*(X*w - y) + lambda*w;

end