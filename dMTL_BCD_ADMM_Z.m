function [ nodes,Results] = dMTL_BCD_ADMM_Z( nodes,neighbors,syn_param,options)
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
    
    % did not use a bias term
    is_bias = 0;
            
    if is_bias
        dim_x = nfeat - 1;
    else
        dim_x = nfeat;
    end
    options.dim_x = dim_x;      
    for k=1:nodenum
        nodes{k}.u = randn(nfeat,1);           
        nodes{k}.M = zeros(dim_x,dim_x);
        nodes{k}.Z = zeros(dim_x,dim_x);
        nodes{k}.Omega = zeros(dim_x,dim_x);
        nodes{k}.W = zeros(dim_x,dim_x);
        nodes{k}.alpha = alpha;        
    end

    
    cost_list = [];     
    single_cost_list = zeros(nodenum, maxiter*2);
    diff_M = [];      
    trans_msd = zeros(maxiter+1,1);
    test_err = zeros(maxiter+1,1);
    exp_var = zeros(maxiter+1,1);
    [trans_msd(1), test_err(1), exp_var(1)] = train_test_msd(nodes,syn_param,options);
    
    all_u = zeros(nfeat,maxiter,nodenum);    
    for t=1:maxiter                    
        
        costs = zeros(1,nodenum);
        sing_costs = zeros(nodenum,nodenum);
        all_M = zeros(nodenum,dim_x*dim_x);        
                
        % Compute the #estimates#
        
        parfor k=1:nodenum            
           % compute the vector                 
            [costs(k),nodes{k}] = optimize_u(nodes{k},eta,is_bias);
            all_u(:,t,k) = nodes{k}.u;
        end     
        cost_list = [cost_list, sum(costs)]; 
        
        for k=1:nodenum
            for i = 1:nodenum
                sing_costs(k,i) = costfunc_single(nodes{i},nodes{k}.M,eta,is_bias);                                 
            end
        end
        single_cost_list(:,2*t-1) = sum(sing_costs,2);
        
        
        nodes = dist_strategy(nodes,neighbors,options,t);
        
%           for k=1:nodenum     
%               nodes{k}.Z = postisemidef(nodes{k}.Z);
%           end
         
        parfor k=1:nodenum
          %  nodes{k}.Z =avgZ;
            % compute subspace
            nodes{k}.M = optimize_M(nodes{k}.Z,eta,h);
            
            % compute the cost
            [costs(k), ~] = cost_func_u(nodes{k}.u,nodes{k},eta,is_bias);            
                        
            all_M(k,:) = nodes{k}.M(:)';                        
        end                                      
        cost_list = [cost_list, sum(costs)];    
        
        for k=1:nodenum
            for i = 1:nodenum
                sing_costs(k,i) = costfunc_single(nodes{i},nodes{k}.M,eta,is_bias);                                 
            end
        end
        single_cost_list(:,2*t) = sum(sing_costs,2);        
        
        diff_M = [diff_M, norm(bsxfun(@minus, all_M, mean(all_M,1)))];  
        
        fprintf('Iter=%d | nodenum=%d | Cost=%f | var_m=%f \n',t,nodenum, cost_list(t), diff_M(t));
                
        % test error & transient MSD
        [trans_msd(t+1), test_err(t+1),exp_var(t+1)] = train_test_msd(nodes,syn_param,options);
    end    
    
    Results.test_err = test_err;
    Results.trans_msd = trans_msd;
    Results.exp_var = exp_var;
%     figure(2);
%     hold on;   
%     for d = 1:nfeat
%         tmp = all_u(d,:,1);
%         plot(1:numel(tmp),tmp,'r-');        
%     end  
%     ylabel('norm2_diff');
%     xlabel('Iterations');
%     hold off;
     
%     figure;
%     hold on;
%     fig1 = plot(1:numel(cost_list),cost_list,'r-','LineWidth',1.5);        
%     for k=1:nodenum
%         list = single_cost_list(k,:);
%         plot(1:numel(list),list,'b-'); 
%     end
    
%     legend(fig1,'averge-cost');
%     title('cost - dMTL-BCD-ADMM-Z');    
%     xlabel('Iterations');
%     hold off;

            
end

function nodes = dist_strategy(nodes,neighbors,options,t)
    
    dim_x = options.dim_x;
    rho = options.rho;
    nodenum = numel(nodes);
    
    diff_Z = [];
    mean_Z = [];
    mean_U = [];
    all_Z = zeros(nodenum,dim_x*dim_x);    
    
    dist_type = 1;
    % admm
    trials = options.loop;
    if dist_type == 1
%         for k=1:nodenum
%             nodes{k}.Omega = zeros(dim_x,dim_x);
%             nodes{k}.W = zeros(dim_x,dim_x);
%         end
        for i = 1:trials
            avg_Zstar = zeros(dim_x,dim_x);
            for k=1:nodenum
                % compute consensus variable
                nei_num = numel(neighbors{k});
                z_star = nodes{k}.u*nodes{k}.u';
                nodes{k}.Z = ( z_star - 2*rho*nodes{k}.Omega + 2*rho*nodes{k}.W)./(1+2*rho*nei_num);
                all_Z(k,:) = nodes{k}.Z(:)';
                avg_Zstar = avg_Zstar + z_star;
            end
            avg_Zstar = avg_Zstar./nodenum;
            % Compute the #Lagrange multipliers# (dual variables)
            for k=1:nodenum
                nei = neighbors{k};
                nei_num = numel(nei);
                W = zeros(dim_x,dim_x);
                for j=1:nei_num
                    W = W + (nodes{k}.Z+nodes{nei(j)}.Z)/2;
                    nodes{k}.Omega = nodes{k}.Omega + (nodes{k}.Z - nodes{nei(j)}.Z)/2;
                end
                nodes{k}.W = W;
            end
            %diff_Z = [diff_Z, norm(bsxfun(@minus, all_Z, avg_Zstar(:)'))];
            diff_Z = [diff_Z, norm(bsxfun(@minus, all_Z, mean(all_Z,1)))];
        end       
        
    elseif dist_type == 2
        %diffusion
        for i = 1:trials
            avg_Zstar = zeros(dim_x,dim_x);
            alpha = 0.2;
            old_nodes = nodes;
            UU = zeros(dim_x,dim_x,nodenum);
            for k=1:nodenum
                z_star = nodes{k}.u*nodes{k}.u';
                UU(:,:,k) = (1-alpha) * old_nodes{k}.Z + alpha*z_star;
                avg_Zstar = avg_Zstar + z_star;
            end
            avg_Zstar = avg_Zstar./nodenum;
            for k=1:nodenum
                nei_num = numel(neighbors{k});
                Z = sum(UU(:,:,neighbors{k}),3);
                nodes{k}.Z = (Z + UU(:,:,k))./(nei_num+1);
                all_Z(k,:) = nodes{k}.Z(:)';
            end
            diff_Z = [diff_Z, norm(bsxfun(@minus, all_Z, avg_Zstar(:)'))];
        end
    elseif dist_type == 3
        %  simple averaing
        avg_Zstar = zeros(dim_x,dim_x);
        for k=1:nodenum
            nodes{k}.Z = nodenum*(nodes{k}.u*nodes{k}.u');
            avg_Zstar = avg_Zstar + nodes{k}.Z;
        end
        avg_Zstar = avg_Zstar./nodenum;
        for i =1:trials;
            UU = zeros(dim_x,dim_x,nodenum);
            for k=1:nodenum
                UU(:,:,k) = nodes{k}.Z;
            end
            for k=1:nodenum
                nodes{k}.Z = mean(UU(:,:,neighbors{k}),3);
                all_Z(k,:) = nodes{k}.Z(:)';
            end
            diff_Z = [diff_Z, norm(bsxfun(@minus, all_Z, avg_Zstar(:)'))];
        end
    end
%     for k=1:nodenum
%         min_eigval = min(eig(nodes{k}.Z));
%         if min_eigval < -eps
%             fprintf('Z is negative definite: %.16f \n', min_eigval);
%         end
%     end
%     figure(1);
%     hold on;    
%     f2 = plot(1+trials*(t-1):t*trials,diff_Z,'b-o');    
%     legend(f2,'Z');
%     ylabel('estimates variance');
%     xlabel('Iterations');
%  
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
    
    %[P1,D,~] = svd(Z);  
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

function [cost,node] = optimize_u(node,eta,is_bias)
    % 1 : fixed step size
    % 2 : linear search
    opt_type = 1;  
    costs = [];
    if opt_type == 1

        % # fixed step size
        max_interIters = 400;
        tau = 0.02;
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
            costs = [costs,cost];
        end

    elseif opt_type == 2

        % # linear search
        maxiter =500;
        old_u = node.u;
        tau = 1;
        beta = 0.6;
        old_cost = 0;
        for i=1:maxiter
            u = node.u + (i-2)/(i+1)*(node.u - old_u);
            old_u = node.u;
            node.u = u;
            [objfunc0, Gu] = cost_func_u(u, node, eta);
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
                if  sum(diffu.^2)/sum(u0.^2) < 1e-10 % this shows that, the gradient step makes little improvement
                    Flag = 1;                    
                    break;
                end
            end           
            if Flag || (i>10 && abs(old_cost - cost)/cost < 1e-8)
                break;
            end
            old_cost = cost;
        end
%     fig1 = plot(1:numel(costs),costs,'-r');
%     legend(fig1,'cost');
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
        Gu = 1./nsample*X'*grad0 + 2*alpha*[0; T*u_cut];
    else
        T = (1+eta)*eye(nfeat) - node.M;
        cost = 1./nsample*cost0 + alpha*u'*T*u;        
        Gu = 1./nsample*X'*grad0 + 2*alpha*T*u;       
    end
end

function cost = costfunc_single(node,M,eta,is_bias)
%    cost  

    alpha = node.alpha;
    u = node.u;
    X = node.data;
    Y = node.gnd;             
    
    [nsample,nfeat] = size(X);  
    
    task_type = 2;
    if task_type == 1
        [ cost0, ~ ] = modihuberloss(X*u, Y);        
    elseif task_type == 2
        [ cost0, ~ ] = squareloss(X*u, Y);
    end
    
    if is_bias
        u_cut = u(2:end);
        T = (1+eta)*eye(nfeat-1) - M;
        cost = 1./nsample*cost0 + alpha*u_cut'*T*u_cut;        
    else
        T = (1+eta)*eye(nfeat) - M;
        cost = 1./nsample*cost0 + alpha*u'*T*u;             
    end
end

function [train_msd, test_err,exp_var] = train_test_msd(nodes,syn_param, options)
    nodenum = numel(nodes);
    % test error & transient MSD
    train_msd = zeros(nodenum,1);
    exp_var = zeros(nodenum,1);
    test_err = zeros(nodenum,1);    
    for k=1:nodenum
        w = nodes{k}.u;
        X = nodes{k}.test_data;
        test_y = nodes{k}.test_gnd;
        pred_y = X*w;
        simul_case = 1;        
        if simul_case == 1
            %test_err(k) =norm((pred_y-test_y).^2,2)/norm(test_y.^2,2);    
            
            test_err(k) = 10*log10(sum((pred_y-test_y).^2,1));
            
            true_w = syn_param.W(:,k);
            train_msd(k) = 10*log10(sum((w - true_w).^2,1));
        elseif simul_case == 2
            exp_var(k) = 1 - sum((pred_y - test_y).^2) / sum((test_y - mean(test_y)).^2);
        end
    end
    test_err = mean(test_err);
    train_msd = mean(train_msd);
    exp_var = mean(exp_var);
end

