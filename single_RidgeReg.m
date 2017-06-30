function  [Results]= single_RidgeReg(nodes, syn_param, lambda)
    nodenum = numel(nodes); 
    [ntrain,nfeat] = size(nodes{1}.data);
    single_U = zeros(nfeat,nodenum);    
    for k =1:nodenum
        X = nodes{k}.data;
        y = nodes{k}.gnd;
        w = (X'*X + 2*ntrain*lambda*eye(nfeat))\(X'*y);
        single_U(:,k) = w;
    end
    % test error
    train_msds = zeros(nodenum,1);
    test_err = zeros(nodenum,1);
    test_err2 = zeros(nodenum,1);
    exp_var = zeros(nodenum,1);
    simul_case = 2;  
    for k=1:nodenum
        w = single_U(:,k);

        % test error/MSD
        test_X = nodes{k}.test_data;
        test_y = nodes{k}.test_gnd;
        pred_y = test_X*w;     
        
              
        if simul_case == 1
            %test_err(k) =norm((pred_y-test_y).^2,2)/norm(test_y.^2,2);    
            test_err(k) = 10*log10(sum((pred_y-test_y).^2,1));  
            test_err2(k) = sum((pred_y-test_y).^2,1);
            true_w = syn_param.W(:,k);
            train_msds(k) = 10*log10(sum((w - true_w).^2,1));
        elseif simul_case == 2
            exp_var(k) = 1 - sum((pred_y - test_y).^2) / sum((test_y - mean(test_y)).^2);
        end
    end
    if simul_case == 1
        Results.train_msd = mean(train_msds);
        Results.test_err = mean(test_err);
        Results.test_err2 = mean(test_err2);
    elseif simul_case == 2
        Results.exp_var = mean(exp_var);
    end

end