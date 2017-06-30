function [ nodes,res] = dMTL_bias( nodes,neighbors,options)
%DMTL Summary of this function goes here
%   
%   distributed Multi-Task Learning (dMTL) via ADMM
%   
%   consensus on Z = \sum u_i u_i^T
%
%   using Block Coordinate Descent
%
% nodes 
%   - data : nsample x nfeat
%   - gnd  : nsample x 1
%
%   Last Update: 2016/1/12
%
% Created by Junhao Hua (huajh7@gmail.com), on July 26, 2015

    nodenum = numel(nodes);      
    [~,nfeat] = size(nodes{1}.data);
    
    % tuning parameter    
    alpha = options.alpha;
    h = options.subdim;    
    eta = options.eta;
    maxiter = options.max_iters;      
    %options.loop = 3;  
    
    for k=1:nodenum
        nodes{k}.u = zeros(nfeat+1,1);           
        nodes{k}.M = zeros(nfeat,nfeat);
        nodes{k}.Z = zeros(nfeat,nfeat);
        nodes{k}.Omega = zeros(nfeat,nfeat);
        nodes{k}.W = zeros(nfeat,nfeat);
        nodes{k}.alpha = alpha;        
    end

    cost_list = zeros(maxiter,1);  
    for t=1:maxiter      
        tic;
        % Compute the #estimates# U
        
        costs = zeros(1,nodenum);     

        for k=1:nodenum     
            T = (1+eta)*eye(nfeat) - nodes{k}.M;                  
            [costs(k),nodes{k}.u] = optimize_u(nodes{k},T);            
         
        end
        
        cost_list(t) = mean(costs);
        terr = testerror(nodes);
     %   fprintf('dmtl: Iter=%d | cost=%f | TestErrorRate (%%) =%f\n',t, cost_list(t),terr);
                
        nodes = dist_strategy(nodes,neighbors,options);
        
%           for k=1:nodenum     
%               nodes{k}.Z = postisemidef(nodes{k}.Z);
%           end
         
        for k=1:nodenum
            % compute subspace
            nodes{k}.M = optimize_M(nodes{k}.Z,eta,h);
            
            % compute the cost
            u = nodes{k}.u;
            X = nodes{k}.data;
            y = nodes{k}.gnd;
            alpha = nodes{k}.alpha; 
            T = (1+eta)*eye(nfeat) - nodes{k}.M;
            [costs(k), ~]  = cost_func_u(u,X,y,T,alpha);                     
        end     
        cost_list(t) = mean(costs); 
        terr = testerror(nodes);
        
        if t==1
            fprintf('dmtl: Iter=%d | cost=%f | TestErrorRate(%%) =%f\n',t, cost_list(t), terr);    
        end
        if t>1 && mod(t,1) == 0
            costrate = abs(cost_list(t) - cost_list(t-1))/cost_list(t-1);
             fprintf('dmtl: Iter=%d | costrate=%f | TestErrorRate(%%) =%f\n',t, costrate, terr);    
        end        
        toc;
        if t>1 && costrate < 1e-5
            break;
        end          

    end    
    
    res.testerr = testerror(nodes);
            
end

function nodes = dist_strategy(nodes,neighbors,options)

    nfeat = size(nodes{1}.data,2);
    rho = options.rho;
    nodenum = numel(nodes);

    all_Z = zeros(nodenum,nfeat*nfeat);

    % admm
    trials = options.loop;
    for i = 1:trials
        avg_Zstar = zeros(nfeat,nfeat);
        for k=1:nodenum
            % compute consensus variable
            nei_num = numel(neighbors{k});
            u = nodes{k}.u(2:end);
            z_star = u*u';
            nodes{k}.Z = ( z_star - 2*rho*nodes{k}.Omega + 2*rho*nodes{k}.W)./(1+2*rho*nei_num);
            all_Z(k,:) = nodes{k}.Z(:)';
            avg_Zstar = avg_Zstar + z_star;
        end
        % Compute the #Lagrange multipliers# (dual variables)
        for k=1:nodenum
            nei = neighbors{k};
            nei_num = numel(nei);
            W = zeros(nfeat,nfeat);
            for j=1:nei_num
                W = W + (nodes{k}.Z+nodes{nei(j)}.Z)/2;
                nodes{k}.Omega = nodes{k}.Omega + (nodes{k}.Z - nodes{nei(j)}.Z)/2;
            end
            nodes{k}.W = W;
        end
    end


end

function [M] = postisemidef(Z)
    [P1,D] = eig(Z);
    D = diag(D);
    idx = D>eps;  
    D = D(idx);
    P1 = P1(:,idx);
    M = P1*diag(D)*P1';  
    M = (M+M')./2;
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

function [cost,final_u] = optimize_u(node,T)
    % # linear search
    maxiter =200;
    u = node.u;
    X = node.data;
    y = node.gnd;
    alpha = node.alpha;
    %old_u = u;    
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
