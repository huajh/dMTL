function [ allerrs ] = test_classifiers2(nodenum, filename,trainrate )

    nodes = data_dispatcher(nodenum,filename, trainrate);
    allerrs = zeros(1,3);
   
    allerrs(1) = test_logit_func( nodes );
%    fprintf('logistic: TestErrorRate(%%) = %f\n',allerrs(1));
      
    allerrs(2) = test_huber_func( nodes );
%    fprintf('modifiedhuber: TestErrorRate(%%) = %f\n',allerrs(2));
 
    % tuning parameter
    % subspace: subdim 
    % regularization parameter: alpha, eta = beta/alpha.
    options.subdim = 4;
    options.eta = 0.001;
    options.alpha = 0.01;

    %%%%%%%%%%%%%%%%%%%
    options.max_iters = 10;
   
%     %[ ~,res] = cMTL_bias( nodes,options);  
%     [ ~,res] = dMTL_bias( nodes,neighbors,options);  
% %    fprintf('cMTL_bias: TestErrorRate(%%) = %f\n',res.testerr);
%   
%     allerrs(3) = res.testerr;
    
    options.rho = 1;
    options.loop = 6;
      
    load(['Network' num2str(nodenum) '.mat']);
    neighbors = Network.neighbors;
    tic;
    [ ~,res] = dMTL_bias( nodes,neighbors,options);
    toc;
    fprintf('dMTL_bias: TestErrorRate(%%) = %f\n',res.testerr);
    allerrs(3) = res.testerr;
end

