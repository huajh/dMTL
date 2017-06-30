function [ err ] = test_huber_func( nodes,options )

    nodenum = numel(nodes);
    errs = zeros(nodenum,1);
    lambda = options.lambda;
    parfor i=1:nodenum
        X = nodes{i}.data;
        y = nodes{i}.gnd;
        testX = nodes{i}.test_data;
        testy = nodes{i}.test_gnd;        
        theta = huberclassifer( X,y ,lambda);    
        pred_y = predicthuber(theta,testX);
        errs(i) = 1 - mean(pred_y==testy);
        errs(i) = errs(i)*100;
    end
     
    err = mean(errs);
end

