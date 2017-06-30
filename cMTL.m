function [ nodes,res] = cMTL( nodes,options)
%DMTL Summary of this function goes here
%   
%   centralized Multi-Task Learning (cMTL) via ADMM
%   
%   consensus on Z = \sum u_i u_i^T
%
%   using Block Coordinate Descent
%
% nodes 
%   - data : nsample x nfeat
%   - gnd  : nsample x 1
%
%
%
% Created by Junhao Hua (huajh7@gmail.com), on May 1, 2016

    nodenum = numel(nodes);      
    [~,nfeat] = size(nodes{1}.data);
    
    % tuning parameter    
    maxiter = options.max_iters;      
    
    h = options.subdim;        
    eta = options.eta;
    alpha = options.alpha;                          
          
    for k=1:nodenum
        nodes{k}.u = zeros(nfeat,1);           
        nodes{k}.M = zeros(nfeat,nfeat);
        nodes{k}.Z = zeros(nfeat,nfeat);
        nodes{k}.Omega = zeros(nfeat,nfeat);
        nodes{k}.avgZ = zeros(nfeat,nfeat);
        nodes{k}.alpha = alpha;        
    end
    
	cost_list = zeros(maxiter,1);                 
    
    for t=1:maxiter                    
                        
        % Compute the #estimates# U
        
        costs = zeros(1,nodenum);
        for k=1:nodenum            
           % compute the vector                 
            [costs(k),nodes{k}] = optimize_u(nodes{k},eta);
        end
        
        cost_list(t) = mean(costs); 
        err = testerror(nodes);            
        fprintf('Iter=%d | cost=%f | testerr=%f\n',t, cost_list(t),err); 
        
        %%%        
        avgZ = zeros(nfeat, nfeat);
        for k=1:nodenum
            u = nodes{k}.u;
            avgZ = avgZ + u*u';
        end
        avgZ = 1./nodenum*avgZ;
        
        % Compute the #Common Structure# M

        M = optimize_M(avgZ,eta,h);
        
        costs = zeros(1,nodenum);
        for k=1:nodenum
            %  nodes{k}.Z =avgZ;
            % compute subspace
            nodes{k}.M = M;
            
            % compute the cost
            [costs(k), ~] = cost_func_u(nodes{k}.u,nodes{k},eta);            
                                  
        end                                    
        cost_list(t) = mean(costs); 
        err = testerror(nodes);
 
        fprintf('Iter=%d | cost=%f | err=%f\n',t, cost_list(t),terr);    
 
        
        if t>5 && abs(cost_list(t) - cost_list(t-1))/cost_list(t-1) < 1e-8
            break;
        end

    end
    
    res.testerr = testerror(nodes);
    
end

function [M] = optimize_M(Z,eta,h)
    
    [P1,D] = eig(Z);
    D = diag(D);
    all_postive = 1;
    if all_postive
        idx = D>0;
        D = D(idx);
        P1 = P1(:,idx);
    end    
    x = quad_kanpsack_singular(sqrt(D)+1e-15, eta, h);    
    M = P1*diag(x)*P1';  
end

function [cost,node] = optimize_u2(node,eta)

    costfunc = @(u) cost_func_u(u,node,eta);
    options = optimset('MaxIter', 200);
    u0 = node.u;
    %u0 = zeros(size(node.u));
    [u, costs] = fmincg(costfunc, u0, options);
    node.u = u;
    if numel(costs) == 0
        cost = 0;
    else
        cost = costs(end);
    end
end


function [cost,node] = optimize_u(node,eta)
    % # linear search
    maxiter =100;
    old_u = node.u;    
    %old_u = zeros(size(node.u));
    tau = 1;
    beta = 0.5;
    old_cost = 0;
    for i=1:maxiter        
        u = node.u + (i-2)/(i+1)*(node.u - old_u);
        old_u = node.u;
        [objfunc0, Gu] = cost_func_u(u, node, eta);
        node.u = u;
        cost = objfunc0;
        Flag = 0;
        while(1)
            u0 = node.u - tau*Gu;
            objfunc = cost_func_u(u0,node,eta);
            diffu = u0-node.u;
            cmp_err = objfunc0 + trace(Gu'*diffu) + 1/(2*tau)*norm(diffu,'fro')^2 - objfunc;
            if cmp_err > 0
                node.u = u0;
                cost = objfunc;
                break;
            else
                tau = beta*tau;
            end
            if  sum(diffu.^2)/sum(u0.^2) < 1e-12 % this shows that, the gradient step makes little improvement
                Flag = 1;
                break;
            end
        end
        if Flag || (i>10 && abs(old_cost - cost)/cost < 1e-12)
            break;
        end
        old_cost = cost;
    end
end

function [cost, Gu] = cost_func_u(u,node,eta)
%    cost 
%    Gu_t : gradient of u of source domain data ( p x 1)     

    alpha = node.alpha;
    X = node.data;
    Y = node.gnd;             
    
    [nsample,nfeat] = size(X);      

    [ cost0, grad0 ] = modihuberloss(X*u, Y);        

    T = (1+eta)*eye(nfeat) - node.M;
    cost = 1./nsample*cost0 + alpha*u'*T*u;        
    Gu = 1./nsample*X'*grad0 + 2*alpha*T*u;

end

function err = testerror(nodes)

    nodenum = numel(nodes);    
    testerrs = zeros(nodenum,1);   
    for k=1:nodenum
        w = nodes{k}.u;
        testX = nodes{k}.test_data;                
        test_y = nodes{k}.test_gnd;
        pred_y = predicthuber(w,testX);
        testerrs(k) = 1 - mean(pred_y == test_y);
    end
    err = mean(testerrs);
   
end

function [ pred_y ] = predicthuber(  theta, X  )
    
    n = size(X,1);    
    
    y = X*theta;
    
    pred_y = ones(n,1);
    
    pred_y(y<0) = -1;    
end

