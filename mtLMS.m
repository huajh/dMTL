function [nodes,res ] = mtLMS(  nodes,neighbors, syn_param, options )
%REGMTL Summary of this function goes here
%   Detailed explanation goes here


    nodenum = numel(nodes);      
    [nsample,nfeat] = size(nodes{1}.data);
    
    % tuning parameter    
    maxiter = options.max_iters;      
    mu = options.mu;
    eta = options.eta;
    
    train_msds = zeros(maxiter,1);
    test_msds = zeros(maxiter,1);         
    
    rhos = Uniform_Weight(nodenum,neighbors);   
    for k=1:nodenum
        nodes{k}.u = zeros(nfeat,1);
    end
    U = zeros(nfeat, nodenum);
    for t=1:maxiter         
        idx = randperm(nsample,1);
        newU = zeros(nfeat,nodenum);
        for k=1:nodenum
                x = nodes{k}.data(idx,:)';
                d = nodes{k}.gnd(idx);                 
                u = U(:,k);
                neis = neighbors{k};              
                uu = bsxfun(@minus, U(:,neis),u);
                innov = (d - x'*u)*x + eta*mean(uu,2);  % 1/degree
                newU(:,k) = u + mu*innov;                
        end        
        U = newU;
        for k=1:nodenum
            nodes{k}.u = U(:,k);
        end
        [train_msds(t), test_msds(t)] = train_test_msd(nodes,syn_param);  
        if mod(t,1000) == 0
            fprintf('t=%d, trans_msd=%f, test_msd=%f\n',t, train_msds(t),test_msds(t) );
        end 
    end
    
    res.test_msds = test_msds;
    res.train_msds = train_msds;

end


function [train_msd, test_msd] = train_test_msd(nodes,syn_param)
    nodenum = numel(nodes);
    % test error & transient MSD
    train_msds = zeros(nodenum,1);
    test_msds = zeros(nodenum,1);    
    
    
    for k=1:nodenum
        w = nodes{k}.u;
        X = nodes{k}.test_data;
        test_y = nodes{k}.test_gnd;
        pred_y = X*w;
                                    
        test_msds(k) = 10*log10(sum((pred_y-test_y).^2,1));
        
        true_w = syn_param.W(:,k);
        train_msds(k) = 10*log10(sum((w - true_w).^2,1));
    end
    
    test_msd = mean(test_msds);
    train_msd = mean(train_msds);
   
end
