function [ nodes ] = dmodihuberloss( nodes, neighbors )
%dmodihubloss Summary of this function goes here
%   Detailed explanation goes here
%
%   Distributed modified Huber loss
%
% nodes 
%   - t_data : nsample x nfeat
%   - t_gnd  : nsample x 1
%
% neighobrs
%
%
% Created by Junhao Hua (huajh7@gmail.com), on June 8, 2015

    nodenum = numel(nodes);   
   
    [~,nfeat] = size(nodes{1}.t_data);
    % tuning parameter
    rho = 0.1;
    
    % 1 : fixed step size
    % 2 : linear search
    % 3 : fmincg
    opt_type = 1;    
    
    % accelerate projected gradient (APG) algorithm    
    maxiter = 200;      
    % initialization
    for k=1:nodenum
        nodes{k}.u_t = randn(nfeat,1);
        nodes{k}.alpha = zeros(nfeat,numel(neighbors{k}));
        nodes{k}.w = zeros(nfeat,numel(neighbors{k}));
        nodes{k}.msgs = [];
        nodes{k}.lambda_t = 0.1;
    end
    
    cost_list = [];        
    err_dual_list  = [];
    diff_u = [];
    for t=1:maxiter                    

        costs = zeros(1,nodenum);
        all_U = [];
        % Compute the #estimates#
        for k=1:nodenum
            if opt_type == 1
                max_interIters = 20;       
                tau = 0.01;
                for i=1:max_interIters                                        
                    [cost,Gu_t] = cost_func(nodes{k}.u_t, nodes{k},rho);
                    node.u_t = nodes{k}.u_t - tau*Gu_t;
                    nodes{k}.u_t = node.u_t + (i-1)/(i+2)*(node.u_t - nodes{k}.u_t);
                end                
            elseif opt_type == 2                                
                max_interIters = 20;       
                for i=1:max_interIters                         
                    [node,cost] = linear_search(nodes{k},rho);
                    nodes{k}.u_t = node.u_t + (i-1)/(i+2)*(node.u_t - nodes{k}.u_t);
                end                
            elseif opt_type == 3
                options = optimset('GradObj', 'on', 'MaxIter',10);         
                [u_t,cost] = fmincg (@(w)(cost_func(w, nodes{k},rho)),nodes{k}.u_t, options);
                nodes{k}.u_t = u_t;
                if isempty(cost)
                    cost = 0;
                else
                    cost = cost(end);
                end                
            end
            all_U = [all_U; nodes{k}.u_t'];
            costs(k) = cost;
        end                    
        
        % Broadcast/Receive the #messages# to/from the neighbors
        for k=1:nodenum            
            nei = neighbors{k};
            msgs = cell(numel(nei),1);            
            for j=1:numel(nei)
                msgs{j}.u_t = nodes{nei(j)}.u_t;
            end
            nodes{k}.msgs = msgs;
        end
        
        % Compute the #Lagrange multipliers# (dual variables)
        for k=1:nodenum
            msgs = nodes{k}.msgs;
            neig_num = numel(msgs);
            for j=1:neig_num
                nodes{k}.alpha(:,j) = nodes{k}.alpha(:,j) + 1/2*(nodes{k}.u_t-msgs{j}.u_t);          
                nodes{k}.w(:,j) = (nodes{k}.u_t + msgs{j}.u_t )/2;
            end              
        end

        cost_list = [cost_list, sum(costs)];                          
        diff_u = [diff_u, norm(bsxfun(@minus, all_U, mean(all_U)))];            
        err_dual_list = [err_dual_list, norm(nodes{1}.alpha,'fro')];   
%           
        fprintf('Iter=%d | Cost=%f | var=%f | dual_err=%f \n',...
            t,cost_list(t), diff_u(t), err_dual_list(t));
        
        if t>1 && abs(cost_list(t) - cost_list(t-1))/cost_list(t) < 1e-6
            break;
        end
        
    end
    
    figure;
    plot(1:numel(err_dual_list),err_dual_list,'g--');
    title('dual error');
    
    figure;
    plot(1:numel(cost_list),cost_list,'r-'); 
    title('cost function');
    
    figure;
    plot(1:numel(diff_u),diff_u,'-');     
    title('estimates variance');
end

function [node0,cost] = linear_search(node,rho)
    tau = 1;
    beta = 0.5;    
    
    [objfunc0,Gu_t] = cost_func(node.u_t,node,rho);            
    maxiter = 13;
    node0 = node;
    for t=1:maxiter        
        node0.u_t = node.u_t - tau*Gu_t;  
        objfunc = cost_func(node0.u_t,node0,rho);            
        diffUt = node0.u_t-node.u_t;
        
        cmp_err = objfunc0 + trace(Gu_t'*diffUt) + 1/(2*tau)*norm(diffUt,'fro')^2 - objfunc;
        if cmp_err > 0
            break;
        else
            tau = beta*tau;            
        end            
    end
%     fprintf ('t=%d | tau=%f \n',t,tau);    
    cost = objfunc;
end

function [cost,Gu_t] = cost_func(u_t,node,rho)

%    cost 
%    Gu_t : gradient of u of target data ( p x 1)    
    lambda_t = node.lambda_t;    
    neig_num = numel(node.msgs);  
        
    % target domain data
    X = node.t_data;
    Y = node.t_gnd;
    %u_t = node.u_t;    
    [nsample,nfeat] = size(X);    
    [ cost0, grad0 ] = modihuberloss(X*u_t, Y);    
    
    u_t2 = u_t;  u_t2(1) = 0;
    cost = 1./nsample*cost0 + lambda_t/2*(u_t2'*u_t2);    % do not penalty the extra term (1)
    
    tmp = 2*norm(bsxfun(@minus, u_t, node.w - node.alpha),'fro')^2;
    cost = cost + 1*rho/2*tmp;
    Gu_t = 1./nsample*X'*grad0 + lambda_t*u_t2 + 2*rho*(neig_num*u_t+sum(node.alpha,2) - sum(node.w,2));        
      
end


