
function nodes = load_school_data(school_name)
% task_indexes = starting index for each task
% 
% Splits are in school_$i$_indexes.mat (i=1,...,10)
% 
% tr = training set indexes (for x in school_b.mat)
% tst = test set indexes (for x in school_b.mat)
% tr_indexes = starting index for each task in tr
% tst_indexes = starting index for each task in tst

    addpath('.\school_splits');
    load('school_b.mat');
    load(school_name);

    ntask = numel(task_indexes);

    tr_indexes(ntask+1) = size(tr,2) + 1;
    tst_indexes(ntask+1) = size(tst,2) + 1;
    for t=1:ntask
        train_x = x(:,tr); % (featnum + 1) x number of students
        train_y = y(tr,1);

        nodes{t}.data =train_x(1:end-1,tr_indexes(t):tr_indexes(t+1)-1)'; % ntrain x featnum
        nodes{t}.gnd = train_y(tr_indexes(t):tr_indexes(t+1)-1);

        test_x = x(:,tst);
        test_y = y(tst,1);    
        nodes{t}.test_data = test_x(1:end-1,tst_indexes(t):tst_indexes(t+1)-1)';
        nodes{t}.test_gnd = test_y(tst_indexes(t):tst_indexes(t+1)-1,1);
    end
end