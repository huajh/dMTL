function [ pred_y ] = predicthuber(  theta, X  )
    
    n = size(X,1);    
    
    y = X*theta(2:end) + theta(1);   
    
    pred_y = ones(n,1);
    
    pred_y(y<0) = -1;    
end

