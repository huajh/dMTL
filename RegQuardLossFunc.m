function [ best_w, best_lambda, min_err ] = RegQuardLossFunc( X,y )
%REGEMPRISK Summary of this function goes here
%
%   solve the regularized empirical risk
%   min |X*w - y|^2 + \lambda |w|^2
%   
%   X nsample x nfeat
%   Y nsample x 1
   
%  using leave-one-out cross-validation

   [nsample,nfeat] = size(X);
    K = 5;
    interval = floor(nsample/K); 
    
    lambdas = [0.05,0.1,0.2,0.5,1,4,10];
    best_w = [];
    min_err = inf;
    err_list = zeros(length(lambdas),1);
    best_lambda = 0;
    for i=1:length(lambdas);
        lambda = lambdas(i);
        err = zeros(K,1);
        for k = 1:K
            val_int = (k-1)*interval+1:min(k*interval, nsample);
            train_int = [1:(k-1)*interval+1,min(k*interval+1, nsample):nsample];        
            train_x = X(train_int, :);          
            train_y = y(train_int,1);          
            val_x = X(val_int,:);
            val_y = y(val_int,1);  
    
            options = optimset('GradObj', 'on', 'MaxIter', 100);
            init_val = randn(nfeat,1);
            w = fmincg (@(w)(cost_func(w, train_x,train_y,lambda)),init_val, options);
            pred_y = val_x*w;            
            err(k) = sum((pred_y-val_y).^2)/sum(y.^2);
            fprintf('EMSE=%.4f\n',err(k));
           % [pred_y, val_y, pred_y-val_y]
        end
        fprintf('lambda = %f | cross-validation error = %.4f\n',lambda, mean(err));
        err_list(i) = mean(err);
        if min_err > mean(err)
            min_err = mean(err);
            best_w = w;
            best_lambda = lambda;
        end
    end
    plot(lambdas,err_list,'-o');
    %set(gca,'xscale','log');
    fprintf('Best: lambda = %f | cross-validation error = %.4f\n',best_lambda, min_err);
end


function [J,grad] = cost_func(w,X,y,lambda)
%   
%   minimize the regularized squared loss
    
    J = 1/2*sum(sum((X*w-y).^2)) + lambda/2*sum(sum(w.^2));
    
    grad = X'*(X*w - y) + lambda*w;

end
