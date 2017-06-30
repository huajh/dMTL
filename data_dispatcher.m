
function  nodes = data_dispatcher(nodenum, filename, trainrate)
        
    load(filename);  
    %fea; %nsample x nfeat
    %gnd; %nsample x 1 

%     fea = double(fea);
%     fea = NormalizeFea(fea); % nsample x dim    
    
    labels = unique(gnd);
    for i=1:numel(labels)
        idx = find(gnd==labels(i));
        objects{i}.fea = fea(idx,:);
    end

    for i=1:nodenum
        idx = randperm(numel(labels),2);        
        X = [objects{idx(1)}.fea; objects{idx(2)}.fea];
        nclass1 = size(objects{idx(1)}.fea,1);
        nclass2 = size(objects{idx(2)}.fea,1);
        nsample = nclass1 + nclass2;
        y = -1*ones(nsample,1);
        y(1:nclass1) = 1;
        mix = randperm(nsample);
        X = X(mix, :);
        y = y(mix);
        ntrain = ceil(nsample*trainrate);
        nodes{i}.data = X(1:ntrain,:);
        nodes{i}.gnd = y(1:ntrain);
        nodes{i}.test_data = X(ntrain+1:end,:);
        nodes{i}.test_gnd = y(ntrain+1:end);
    end
end