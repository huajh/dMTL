function [ mis_class_rate ] = single_modihubloss( nodes,option )
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
    nodenum = numel(nodes); 
    [~,nfeat] = size(nodes{1}.data);
    single_U = zeros(nfeat,nodenum);
    lambda = option.beta + option.alpha;
    
    options = optimset('GradObj', 'on', 'MaxIter', 200);
    init_val = zeros(nfeat,1);
    for i =1:nodenum
        X = nodes{i}.data;
        y = nodes{i}.gnd;
       % w = accgrad(X,y,init_val, lambda);
        [w,~] = fmincg (@(w)(cost_func(w,X,y,lambda)),init_val, options);
        single_U(:,i) = w;
    end

    [mis_class_rate] = mis_class_rate_func(nodes,single_U);  

end

function [mis_class_rate] = mis_class_rate_func(nodes,single_U)
    nodenum = numel(nodes); 
    misclassificationrate = zeros(nodenum,1);
    for i=1:nodenum
        w = single_U(:,i);
        test_X = nodes{i}.test_data;
        test_y = nodes{i}.test_gnd;
        [nsample,~] = size(test_X);
        y = test_X*w;
        pred_y = -1*ones(nsample,1);
        pred_y(y>=0) = 1;
        pred_y = pred_y(:);
        misclassificationrate(i) = sum(test_y ~= pred_y)/nsample;
 %       fprintf('misclassification rate = %f\n', misclassificationrate(i));
    end
    mis_class_rate = mean(misclassificationrate);
end

function w = accgrad(X,y, init_val,lambda)
    max_interIters = 100;
    tau = 0.1;
    u0 = init_val;
    cost_list = [];
    for t=1:max_interIters        
        [cost,Gu] = cost_func(u0,X,y,lambda);
        u = u0 - tau*Gu;
        u = u + (t-1)/(t+2)*(u - u0);
        u0 = u;
        cost_list = [cost_list,cost];  
    end    
    figure;
    plot(1:numel(cost_list),cost_list,'-');
    title('cost');    
    w = u;
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