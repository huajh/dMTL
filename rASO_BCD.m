function  [U,M,Results] = rASO_BCD(nodes,X,Y,syn_param,options)
%RASO_APG Summary of this function goes here
%   
%   relaxed Alternating Structure Optimization (ASO) using 
%      using Block Coordinate Descent
%
%   X : nsample x nfeat x ntasks
%   Y : nsample x ntasks
%
%   U = [u1,u2,...,u_m]  nfeat x ntasks
%   Theta : h x nfeat, h < nfeat
%   M = Theta'*Theta : h x h 
%
% Created by Junhao Hua (huajh7@gmail.com), on June 7, 2015
    
    [nsample,nfeat,ntasks] = size(X);
    
    % tuning parameter
   
    alpha = options.alpha;
    eta = options.eta;
    h = options.subdim;
    maxiter = options.max_iters;

    
    
    % did not use a bias term
    is_bias = 0;
    if is_bias
        dim_x = nfeat - 1;
    else
        dim_x = nfeat;
    end
    % initialization
    U = randn(nfeat,ntasks);    
    M = zeros(dim_x,dim_x);
             
    costs = [];
    all_M = [];
    trans_msd = zeros(maxiter+1,1);
    test_err = zeros(maxiter+1,1);
    exp_var = zeros(maxiter+1,1);
    
    %%%%%%%%%%%
    % test error & transient MSD
    [trans_msd(1), test_err(1),exp_var(1)] = train_test_msd(nodes,ntasks,U,syn_param,options);
        
    for t = 1:maxiter 
        % 
        [cost,U] = optimize_u(X,Y,U,M,eta,alpha,is_bias);
        [M] = optimize_M(U,eta,h);
        
        costs = [costs,cost];
        
        [cost,~] = cost_func(U,X,Y,M,eta,alpha,is_bias);              
        
        costs = [costs,cost];
        
%         fprintf('t=%d | cost=%f \n',t, cost);
        
        
        all_M = [all_M, norm(M,'fro').^2];
%         if t>500 && abs(costs(t) - costs(t-1))/costs(t-1) < 1e-12
%             break;
%         end       
        
        % test error & transient MSD 
        [trans_msd(t+1), test_err(t+1),exp_var(t+1)] = train_test_msd(nodes,ntasks,U,syn_param,options);
        
    end    
    Results.trans_msd = trans_msd;
    Results.test_err = test_err;
    Results.exp_var = exp_var;
    
%     figure;    
%     plot(1:numel(costs),costs,'-b');    
%     title('cost - rASO-BCD');    
%     
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


function [cost,U] = optimize_u(X,Y,U_old,M,eta,alpha,is_bias)
    % 1 : fixed step size
    % 2 : linear search
    opt_type =1;  
    costs = [];
    U = U_old;
    if opt_type == 1

        % # fixed step size
        max_interIters = 400;
        tau = 0.02;  
        old_cost = 0;
        
        for t=1:max_interIters
            % searching points
            U_half = U + (t-2)/(t+1)*(U-U_old); 
            U_old = U;
            % a feasible solution                      
            [cost,Gu] = cost_func(U_half,X,Y,M,eta,alpha,is_bias);     
            U = U_half - tau*Gu;            
            costs = [costs,cost];
            if norm(U - U_half,'fro')^2 < eps || (old_cost - cost).^2/cost.^2 < 1e-6
                break;
            end            
        end
    elseif opt_type == 2

        % # linear search
        max_interIters = 200;
        tau = 1;
        beta = 0.5;
        old_cost = 0;
        for t=1:max_interIters
            U_half = U + (t-2)/(t+1)*(U-U_old); 
            U_old = U;
            [objfunc0,Gu] = cost_func(U_half,X,Y,M,eta,alpha,is_bias);              
            Flag = 0;
            while(1)
                U = U_half - tau*Gu;
                [objfunc,~] = cost_func(U,X,Y,M,eta,alpha,is_bias);                      
                diffU = U-U_old;
                cmp_err = objfunc0 + trace(Gu'*diffU) + 1/(2*tau)*norm(diffU,'fro')^2 - objfunc;
                if  norm(diffU,'fro')^2 < eps % this shows that, the gradient step makes little improvement
                    Flag = 1;
                    break;
                end
                if cmp_err > 0
                    break;
                else
                    tau = beta*tau;
                end
            end
            cost = objfunc; 
            costs = [costs,cost];            
            if Flag || (old_cost - cost).^2/cost.^2 < 1e-10
                break;
            end                        
            old_cost = cost;
        end
    end
%      plot(1:numel(costs),costs,'-r');
%      title('cost of u'); 
     cost = costs(end);
end


function [cost,Gu] = cost_func(U,X,Y,M,eta,alpha,is_bias)

%    cost 
%    Gu : gradient of U ( p x ntasks)
%
    [nsample,nfeat,ntasks] = size(X);
    if is_bias
        T = (1+eta)*eye(nfeat-1) - M;
    else
        T = (1+eta)*eye(nfeat) - M;
    end
    
    Gu = zeros(nfeat,ntasks);
    cost = 0;
    for i = 1:ntasks
        u = U(:,i);           
        task_type = 2;
        if task_type == 1 % classification
            [ cost0, grad0 ] = modihuberloss(X(:,:,i)*u,  Y(:,i));        
        elseif task_type == 2 % regression
            [ cost0, grad0 ] = squareloss(X(:,:,i)*u,  Y(:,i));
        end        
        if is_bias
            u_cut = u(2:end);
            cost = cost + 1./nsample*cost0 + alpha*u_cut'*T*u_cut; 
            Gu(:,i) = 1./nsample*X(:,:,i)'*grad0 +2*alpha*[0; T*u_cut];
        else
            cost = cost + 1./nsample*cost0 + alpha*u'*T*u; 
            Gu(:,i) = 1./nsample*X(:,:,i)'*grad0 + 2*alpha*T*u;
        end
    end
end

function [train_msd, test_err,exp_var] = train_test_msd(nodes,ntasks,U,syn_param,options)
    %%%%%%%%%%%
    % test error & transient MSD
    train_msd = zeros(ntasks,1);
    exp_var = zeros(ntasks,1);
    test_err = zeros(ntasks,1);
    for k=1:ntasks        
        w = U(:,k);
        test_X = nodes{k}.test_data;
        test_y = nodes{k}.test_gnd;
        pred_y = test_X*w;        
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
    train_msd = mean(train_msd);
    test_err = mean(test_err);
    exp_var = mean(exp_var);
end




