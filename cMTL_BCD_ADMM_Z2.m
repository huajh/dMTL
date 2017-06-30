function [ nodes,Results] = cMTL_BCD_ADMM_Z2( nodes,syn_param,options)
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
% Created by Junhao Hua (huajh7@gmail.com), on July 26, 2015

    nodenum = numel(nodes);      
    [~,nfeat] = size(nodes{1}.data);
    
    % tuning parameter    
    alpha = options.alpha;
    rho = options.rho;
    h = options.subdim;    
    eta = options.eta;
    maxiter = options.max_iters;      
   
    
    % did not use a bias term
    is_bias = 0;
            
    if is_bias
        dim_x = nfeat - 1;
    else
        dim_x = nfeat;
    end
          
    for k=1:nodenum
        nodes{k}.u = randn(nfeat,1);           
        nodes{k}.M = zeros(dim_x,dim_x);
        nodes{k}.Z = zeros(dim_x,dim_x);
        nodes{k}.Omega = zeros(dim_x,dim_x);
        nodes{k}.avgZ = zeros(dim_x,dim_x);
        nodes{k}.alpha = alpha;        
    end

    
	cost_list = [];         
    diff_M = [];      
    exp_var = zeros(maxiter+1,1);
    
    
    [exp_var(1)] = train_test_msd(nodes);
    
    for t=1:maxiter                    
        
        costs = zeros(1,nodenum);
        all_M = zeros(nodenum,dim_x*dim_x);
        
        % Compute the #estimates#
        
        for k=1:nodenum            
           % compute the vector                 
            [cost,nodes{k}] = optimize_u(nodes{k},eta,is_bias);
        end
        trials = 10;
        for i =1:trials
            for k=1:nodenum
                % compute consensus variable
                nodes{k}.Z = (nodenum*(nodes{k}.u*nodes{k}.u')...
                    + rho/2*(nodes{k}.avgZ-nodes{k}.Omega))./(1+rho/2);
            end

            % Compute the #Lagrange multipliers# (dual variables)        
            avgZ = zeros(dim_x,dim_x);
            for k=1:nodenum            
                avgZ = avgZ + nodes{k}.Z;
            end
            avgZ = 1./nodenum*avgZ;
            for k=1:nodenum
                nodes{k}.avgZ = avgZ;
                nodes{k}.Omega = nodes{k}.Omega + nodes{k}.Z - avgZ;
            end  
        end              
        for k=1:nodenum
          %  nodes{k}.Z =avgZ;
            % compute subspace
            nodes{k}.M = optimize_M(nodes{k}.Z,eta,h);
            
            % compute the cost
            [cost, ~] = cost_func_u(nodes{k}.u,nodes{k},eta,is_bias);            
            
            costs(k) = cost;            
            all_M(k,:) = nodes{k}.M(:)';                        
         end                            
        
        cost_list = [cost_list, mean(costs)];                        
        diff_M = [diff_M, norm(bsxfun(@minus, all_M, mean(all_M,1)))];  
        fprintf('Iter=%d | nodenum=%d | Cost=%f | var_m=%f \n',t,nodenum, cost_list(t), diff_M(t));
                
        [exp_var(t+1)] = train_test_msd(nodes);
    end    
    
    Results.exp_var = exp_var;
      
%     figure;
%     f1 = plot(1:numel(cost_list),cost_list,'r-'); 
%     legend(f1,['rho = ' num2str(rho)]); 
%     ylabel('cost');
%     xlabel('Iterations');
%     
%     figure;
%     f2 = plot(1:numel(diff_M),diff_M,'r-');   
%     legend(f2,'M'); 
%     ylabel('estimates variance');
%     xlabel('Iterations');
% %             
end


function [M] = optimize_M(U,eta,h)
    if size(U,2) == 1
        P1 = U./sum(U.^2);
        D = sum(U.^2);
    else
        [P1,D,~] = svd(U);
    end   
    D = diag(D);
    idx = D>0;
    D = D(idx);
    P1 = P1(:,idx);
    x = quad_kanpsack_singular(D, eta, h);    
    M = P1*diag(x)*P1';  
end

function [cost,node] = optimize_u(node,eta,is_bias)
    % 1 : fixed step size
    % 2 : linear search
    opt_type = 2;  
    if opt_type == 1

        % # fixed step size
        max_interIters = 400;
        tau = 0.001;
        old_u = node.u;
        old_cost = 0;
        for i=1:max_interIters
            % searching points
            u = node.u + (i-2)/(i+1)*(node.u - old_u);
            old_u = node.u;
            % a feasible solution
            [cost,Gu] = cost_func_u(u, node, eta,is_bias);
            node.u = u - tau*Gu;
            if sum((node.u - u).^2) < eps
                break;
            end
%             if i>50 && abs(old_cost - cost)/cost < 1e-10
%                 break;
%             end
            old_cost = cost;
        end

    elseif opt_type == 2

        % # linear search
        max_interIters = 400;
        old_u = node.u;
        tau = 1;
        beta = 0.5;
        old_cost = 0;
        for i=1:max_interIters
            u = node.u + (i-2)/(i+1)*(node.u - old_u);
            old_u = node.u;
            node.u = u;
            [objfunc0, Gu] = cost_func_u(u, node, eta,is_bias);
            Flag = 0;
            while(1)
                u0 = node.u - tau*Gu;
                objfunc = cost_func_u(u0,node,eta,is_bias);
                diffu = u0-node.u;
                cmp_err = objfunc0 + trace(Gu'*diffu) + 1/(2*tau)*norm(diffu,'fro')^2 - objfunc;
                if  norm(diffu,'fro')^2 < eps % this shows that, the gradient step makes little improvement
                    Flag = 1;
                    break;
                end
                if cmp_err > 0
                    break;
                else
                    tau = beta*tau;
                end
            end

            if Flag
                break;
            end
            cost = objfunc;
            node.u = u;
            if i>200 && abs(old_cost - cost)/cost < 1e-12
                break;
            end
            old_cost = cost;
        end
    end

end

function [cost, Gu] = cost_func_u(u,node,eta,is_bias)
%    cost 
%    Gu_t : gradient of u of source domain data ( p x 1)     

    alpha = node.alpha;
    X = node.data;
    Y = node.gnd;             
    
    [nsample,nfeat] = size(X);  
    
    task_type = 2;
    if task_type == 1
        [ cost0, grad0 ] = modihuberloss(X*u, Y);        
    elseif task_type == 2
        [ cost0, grad0 ] = squareloss(X*u, Y);
    end
    
    if is_bias
        u_cut = u(2:end);
        T = (1+eta)*eye(nfeat-1) - node.M;
        cost = 1./nsample*cost0 + alpha*u_cut'*T*u_cut;        
        Gu = 1./nsample*X'*grad0 + alpha*[0; T*u_cut];
    else
        T = (1+eta)*eye(nfeat) - node.M;
        cost = 1./nsample*cost0 + alpha*u'*T*u;        
        Gu = 1./nsample*X'*grad0 + alpha*T*u;       
    end
end

function [exp_var] = train_test_msd(nodes)
    nodenum = numel(nodes);
    % test error & transient MSD    
    exp_var = zeros(nodenum,1);
    
    
    for k=1:nodenum
        w = nodes{k}.u;
        X = nodes{k}.test_data;
        test_y = nodes{k}.test_gnd;
        pred_y = X*w;
        
        % explained variance
       % exp_var(k) = 1 - sum((pred_y - test_y).^2) / sum((test_y - mean(test_y)).^2);
        
        %
        X = nodes{k}.data;
        y = nodes{k}.gnd; 
        pred_y = X*w;
        exp_var(k) = 1 - sum((pred_y - y).^2) / sum((y - mean(y)).^2);
    end
    
    exp_var = mean(exp_var);
   
end

