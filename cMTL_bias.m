function [ nodes,res] = cMTL_bias( nodes,options)
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
% Created by Junhao Hua (huajh7@gmail.com), on May 13, 2016

    nodenum = numel(nodes);      
    [~,nfeat] = size(nodes{1}.data);
    
    % tuning parameter    
    maxiter = options.max_iters;      
    
    h = options.subdim;        
    eta = options.eta;
    alpha = options.alpha;               
          
    for k=1:nodenum
        nodes{k}.u = zeros(nfeat+1,1);           
        M = zeros(nfeat,nfeat);
        nodes{k}.Z = zeros(nfeat,nfeat);
        nodes{k}.Omega = zeros(nfeat,nfeat);
        nodes{k}.avgZ = zeros(nfeat,nfeat);
        nodes{k}.alpha = alpha;        
    end
    
	cost_list = zeros(maxiter,1); 
    testerrs = zeros(maxiter,1);
    
    for t=1:maxiter                    
                         
        % Compute the #estimates# U
        
        costs = zeros(1,nodenum);     
         
        T = (1+eta)*eye(nfeat) - M;
         
        for k=1:nodenum                                        
            [costs(k),nodes{k}.u] = optimize_u(nodes{k},T);                     
        end         
        
        cost_list(t) = mean(costs); 
        terr = testerror(nodes);                    
     %   fprintf('cmtl: Iter=%d | cost=%f | TestErrorRate (%%) =%f\n',t, cost_list(t),terr);   
        %%%      
        
        avgZ = zeros(nfeat, nfeat);
        for k=1:nodenum
            u = nodes{k}.u(2:end);
            avgZ = avgZ + u*u';
        end
        avgZ = 1./nodenum*avgZ;
        
        % Compute the #Common Structure# M
        
        M = optimize_M(avgZ,eta,h);
        T = (1+eta)*eye(nfeat) - M;
        costs = zeros(1,nodenum);
        for k=1:nodenum
            % compute the cost
            u = nodes{k}.u;
            X = nodes{k}.data;
            y = nodes{k}.gnd;
            alpha = nodes{k}.alpha;
            [costs(k), ~]  = cost_func_u(u,X,y,T,alpha);
        end                                    
        cost_list(t) = mean(costs); 
        terr = testerror(nodes);
        testerrs(t) = terr;        
        
%         if t==1
%             fprintf('cmtl: Iter=%d | cost=%f | TestErrorRate(%%) =%f\n',t, cost_list(t), terr);    
%         end
        if t>1 && mod(t,1) == 0
            costrate = abs(cost_list(t) - cost_list(t-1))/cost_list(t-1);
%              fprintf('cmtl: Iter=%d | costrate=%f | TestErrorRate(%%) =%f\n',t, costrate, terr);    
        end     
        
        if t > 2 && std(testerrs(t-2:t)) < 1e-5
            break;
        end
        
        if t>1 && costrate < 1e-5
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
    x = quad_kanpsack_singular(sqrt(D)+1e-16, eta, h);    
    M = P1*diag(x)*P1';  
end

% function [cost,node] = optimize_u2(node,T)
%     
%     u0 = node.u;    
%     X = node.data;
%     y = node.gnd;
%     alpha = node.alpha;
%     
%     costfunc = @(u) cost_func_u(u,X,y,T,alpha);
%     options = optimset('MaxIter', 200);
%     
%     %u0 = zeros(size(node.u));
%     [u, costs] = fmincg(costfunc, u0, options);
%     node.u = u;
%     if numel(costs) == 0
%         cost = 0;
%     else
%         cost = costs(end);
%     end
% end

function [cost,final_u] = optimize_u(node,T)
    % # linear search
    maxiter =200;
    u = node.u;
    X = node.data;
    y = node.gnd;
    alpha = node.alpha;
    old_u = u;    
    old_u = zeros(size(node.u));
    tau = 1;
    beta = 0.5;
    old_cost = 0;
    for i=1:maxiter        
        uhalf = u + (i-2)/(i+1)*(u - old_u);
        old_u = u;
        [objfunc0, Gu] = cost_func_u(uhalf,X,y,T,alpha);
        u = uhalf;
        cost = objfunc0;
        Flag = 0;
        while(1)
            u0 = u - tau*Gu;
            objfunc = cost_func_u(u0,X,y,T,alpha);
            diffu = u0-u;
            cmp_err = objfunc0 + trace(Gu'*diffu) + 1/(2*tau)*norm(diffu,'fro')^2 - objfunc;
            if cmp_err > 0
                u = u0;
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
    final_u = u;
end

function [cost, Gu] = cost_func_u(u,X,y,T,alpha)
%    cost 
%    Gu_t : gradient of u of source domain data ( p x 1)     

    n = size(X,1);
    X = [ones(n,1) X];           
    
    nsample = size(X,1);  
    
    [ cost0, grad0 ] = modihuberloss(X*u, y);        
    u_cut = u(2:end);
    Tucut = T*u_cut;
    cost = 1./nsample*cost0 + alpha*u_cut'*Tucut;        
    Gu = 1./nsample*X'*grad0 + 2*alpha*[0; Tucut];

end

function err = testerror(nodes)

    nodenum = numel(nodes);    
    errs = zeros(nodenum,1);   
    for k=1:nodenum
        w = nodes{k}.u;
        testX = nodes{k}.test_data;                
        test_y = nodes{k}.test_gnd;
        pred_y = predicthuber(w,testX);
        errs(k) = 1 - mean(pred_y == test_y);
    end
    err = 100*mean(errs);    
   
end

