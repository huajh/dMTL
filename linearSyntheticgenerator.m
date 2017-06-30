function [ nodes,syn_param ] = linearSyntheticgenerator( ntask, nfeat,options)
%LINEARSYNTHETICGENERATOR Summary of this function goes here
%   Detailed explanation goes here
    
%   The construction of linear synthetic data for regression
% 
%
% Created by Junhao Hua (huajh7@gmail.com), on June 10, 2015


  %  ntask =200;
  %  nfeat = 10;
        
    ntrain = options.ntrain;
    ntest = options.ntest;
    nsample = ntrain + ntest;
    r = options.info_num;
    
    %feature dimension
    %h = 5;

    mu = zeros(r,1);

    % a sum of h rank-one matrices    
    Sigma = wishrnd(eye(r),r);

    %Sigma = diag(rand(dim,1));
    %Sigma = diag([1,0.64,0.49,0.36,0.25]);
    %Sigma = diag([1,0.64,0.49,0.36,0.25,0.5,0.6,0.2]);
    %Sigma = diag([0.5,0.25,0.1,0.05,0.15,0.1,0.15]);
    % nfeat x ntask
    % W = [w1,w2,...,wn];

    % nfeat x ntask
    W = mvnrnd(mu,Sigma,ntask)';

    % irrelevant dimension
    irr = nfeat - r;
    W = [W; zeros(irr,ntask)];
    idx = randperm(nfeat);
    W = W(idx,:);
    for i=1:ntask
        X = rand(nsample,nfeat);
        %X = mvnrnd(zeros(nfeat,1),1*eye(nfeat),nsample); % nfeat x nsample
        y = X*W(:,i) + normrnd(0,0.05,nsample,1); % standrad derviation= 0.1
        nodes{i}.data =X(1:ntrain,:);
        nodes{i}.gnd = y(1:ntrain,:);
        nodes{i}.test_data = X(ntrain+1:end,:);
        nodes{i}.test_gnd = y(ntrain+1:end,:);
    end
    syn_param.W = W;
    syn_param.Sigma = Sigma;   

end

