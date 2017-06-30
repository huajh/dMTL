function pcadata(filename,score)

    load(sprintf('%s.mat',filename));
   
    % 
    %
    % feature:      fea nsample x nfeat
    % groundturth:  gnd nsample x 1
    %
   
    fea = double(fea);
    data = NormalizeFea(fea); % nsample x dim
    
    [nsample,dim] = size(data);

    xbar = mean(data,1);
    means = bsxfun(@minus, data, xbar);
    cov = means'*means/nsample;
    [V,D] = eig(cov);
    eigval = diag(D);
    [~,idx] = sort(eigval,'descend');
    eigval = eigval(idx);
    V = V(idx,:);
    p = 0;
    for i=1:dim
       perc = sum(eigval(1:i))/sum(eigval);
       if perc > score
           p = i;
           break;       
       end 
    end

    p
    E = V(1:p,:);

    fea = means*E';

    save(sprintf('%s_PCA.mat',filename),'fea','gnd');    

end

