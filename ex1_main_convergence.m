
close all;
clear;

nodenum = 20;
load(['Network' num2str(nodenum) '.mat']);
neighbors = Network.neighbors;


%%%%%%%%%%%%%%%%%%%%%%%%%%%
% what's the difference ?
loops = [3,6,9]; %
%%%%%%%%%%%%%%%%%%%%%%%%%%%

ntask = nodenum;
nfeat = 20;
options.subdim = 5;
options.info_num = 5; % the dimension of the informative feature

options.alpha = 1e-2;
options.eta = 1e-2;
options.lambda = 1e-4;

%options.lambda = options.alpha*(1+options.eta);
options.rho = 1;

options.ntrain = 20;
options.ntest = 50;

options.max_iters = 100;

trails = 50;

dmtl_train_msd = zeros(trails,options.max_iters+1,numel(loops));
raso_train_msd = zeros(trails,options.max_iters+1);
sing_train_msd = zeros(trails,1);

dmtl_test_err = zeros(trails,options.max_iters+1,numel(loops));
raso_test_err = zeros(trails,options.max_iters+1);
sing_test_err = zeros(trails,1);

for tr = 1:trails  
    tic;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fprintf(['trail = ' num2str(tr) '\n']);
    % Linear Synthetic Data test
    [ nodes,syn_param ] = linearSyntheticgenerator(ntask, nfeat,options );
    %[ nodes,syn_param ] = linsyngenerator2( ntask, nfeat,options);
    [ntrain,nfeat] = size(nodes{1}.data);
    X_train = zeros(ntrain,nfeat,ntask);
    Y_train = zeros(ntrain,ntask);
    for i=1:ntask
        X_train(:,:,i) =  nodes{i}.data;
        Y_train(:,i) = nodes{i}.gnd;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Ridge Regression    
    res = single_RidgeReg(nodes, syn_param,options.lambda);       
	sing_train_msd(tr) = res.train_msd;
    sing_test_err(tr) = res.test_err; 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    [raso_U,raso_M,res2] = rASO_BCD(nodes, X_train,Y_train,syn_param, options);
    raso_train_msd(tr,:) = res2.trans_msd;
    raso_test_err(tr,:) = res2.test_err;
    for iter = 1:numel(loops)
        options.loop = loops(iter);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        fprintf('dMTL_BCD_ADMM_Z\n');

        [ nodes,res1] = dMTL_BCD_ADMM_Z( nodes,neighbors,syn_param,options);
      
        dmtl_train_msd(tr,:,iter) = res1.trans_msd;                
        dmtl_test_err(tr,:,iter) = res1.test_err;        
    end
    clear nodes;
    toc;
end

options.max_iters = 100;
max_iter = options.max_iters;
figure;
%subplot(2,1,1);
hold on;
fig11 = plot(0:max_iter, mean(dmtl_train_msd(:,:,1),1),'-','Color',[0,0.5,0],'LineWidth',1.5);        
fig12 = plot(0:max_iter, mean(dmtl_train_msd(:,:,2),1),'-r','LineWidth',1.5);        
fig13 = plot(0:max_iter, mean(dmtl_train_msd(:,:,3),1),'-','Color',[0.6, 0.2,0],'LineWidth',1.5);        
fig2 = plot(0:max_iter, mean(raso_train_msd(:,:),1),':b','LineWidth',1.5); 
fig3 = plot([0,max_iter],[mean(sing_train_msd),mean(sing_train_msd)],'--k','LineWidth',1.5);

%legend([fig2,fig3],'cMTL','independent');
legend([fig11,fig12,fig13,fig2,fig3],'dMTL, L=3','dMTL, L=6','dMTL, L=9','cMTL','independent');
ylabel('Transient network MSDs (dB)');
xlabel('Iterations, t');
title('ntrain=20');
set(gcf, 'Color', 'w');

raso_final = mean(raso_train_msd(:,:),1);
sing_final = mean(sing_train_msd);
fprintf('multi - single = %f\n',raso_final(end) - sing_final);

%saveas(gcf,'ex1_train_msd.fig');

%subplot(2,1,2);
figure;
hold on;
fig11 = plot(0:max_iter, mean(dmtl_test_err(:,:,1),1),'-','Color',[0,0.5,0],'LineWidth',1.5);        
fig12 = plot(0:max_iter, mean(dmtl_test_err(:,:,2),1),'-r','LineWidth',1.5);        
fig13 = plot(0:max_iter, mean(dmtl_test_err(:,:,3),1),'-','Color',[0.6, 0.2,0],'LineWidth',1.5);    
fig2 = plot(0:max_iter, mean(raso_test_err(:,:),1),':b','LineWidth',1.5); 
fig3 = plot([0, max_iter],[mean(sing_test_err),mean(sing_test_err)],'--k','LineWidth',1.5);
legend([fig11,fig12,fig13,fig2,fig3],'dMTL, L=3','dMTL, L=6','dMTL, L=9','cMTL','independent');
ylabel('Transient network MSDs (dB)');
xlabel('Iterations, t');
title('Test');
set(gcf, 'Color', 'w');

%saveas(gcf,'ex1_test_msd.fig');


% save ex1_20.mat options loops...
%     dmtl_train_msd raso_train_msd sing_train_msd ...
%     dmtl_test_err raso_test_err sing_test_err
