function [ cost, grad ] = modihuberloss(f,y)
%MODIHUBERLOSS 
%  
%   Modified Huber loss
%   L(f,y) = max(0,1-yf)^2  when yf>=-1
%          = -4yf           when yf<-1
%
%   expected ouput:f        N x 1
%   ground turth: y (+/-1)  N x 1
%   
% Created by Junhao Hua, on June 5, 2015

    E = f.*y;
    idx = find(E>=-1);
    loss = -4*E;
    loss(idx) =  max(0,1-E(idx)).^2;
    
    cost = sum(loss);
        
    grad = 2*(f-y);
    grad(E>=1) = 0;
    grad1 = -4*y;
    idx = find(E<=-1);    
    grad(idx) = grad1(idx);
    
end

