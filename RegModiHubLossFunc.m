function [ w ] = RegModiHubLossFunc( X,y,lambda )
%REGMODIHUBLOSSFUNC Summary of this function goes here
%
%  Regularized Modified Huber loss
%
%   min_w 1/N*L(X*w,y)+\lambda/2 |w|^2
%
%   L(f,y) = max(0,1-yf)^2  when yf>=-1
%          = -4yf           when yf<-1
%
%   X : N x p
%   y : N x 1
%   w : p x 1

    if nargin < 3
        lambda =  0.1;
    end
        
    [nsample,nfeat] = size(X);    
    options = optimset('GradObj', 'on', 'MaxIter', 200);
   % init_val = randn(nfeat,1);   
    init_val = zeros(nfeat,1);
    
     [w,cost] = fmincg (@(w)(cost_func(w,X,y,lambda)),init_val, options);
    
%     max_interIters = 100;
%     tau = 0.001;
%     u0 = init_val;
%     cost_list = [];
%     for t=1:max_interIters        
%         [cost,Gu] = cost_func(u0,X,y,lambda);
%         u = u0 - tau*Gu;
%         u = u + (t-1)/(t+2)*(u - u0);
%         u0 = u;
%         cost_list = [cost_list,cost];  
%     end    
%     figure;
%     plot(1:numel(cost_list),cost_list,'-');
%     title('cost');    
%     w = u;
    
%     dist_y = X*w;
%     pred_y(dist_y<0) = 1;
%     pred_y(dist_y>=0) = -1;
%     pred_y = pred_y(:);
%     acc = sum(y ~= pred_y)/nsample;
%     fprintf('acc=%f\n',acc);

end

function [cost,grad] = cost_func(w,X,y,lambda)
%
%
    [nsample,nfeat] = size(X);
        
    [ cost0, grad0 ] = modihuberloss(X*w,y);
    
    w(1) = 0;
    cost = 1/nsample*cost0 + lambda/2*(w'*w);    
    grad = 1/nsample*X'*grad0 + lambda*w;
end