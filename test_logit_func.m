function [ err ] = test_logit_func( nodes )
    
    nodenum = numel(nodes);
    errs = zeros(nodenum,1);
    parfor i=1:nodenum
        X = nodes{i}.data;
        y = nodes{i}.gnd;
        testX = nodes{i}.test_data;
        testy = nodes{i}.test_gnd;
        y(y==-1) = 0;
        testy(testy==-1) = 0;
        lambda = 0;
        theta = logit( X,y ,lambda);    
        pred_y = predictlogit(theta,testX);
        errs(i) = 1 - mean(pred_y==testy);
        errs(i) = errs(i)*100;
    end
  
    err = mean(errs);

end

