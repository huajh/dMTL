function [ Network ] = CreateNetworks(Conf)
%CREATENETWORKS Summary of this function goes here
%   Detailed explanation goes here

    num = Conf.nodenum;
    square = Conf.square;
    commdist = Conf.commdist;
    
    loc = square*rand(num,2) - square/2;
        
     
    Dists = Euclid_Dist(loc(:,1),loc(:,2));
    % without self-loop
    Dists = Dists + 10*commdist*eye(num);
    Neighbors = cell(num,1);
    maxDegree = 0;
    edges = 0;
    lap_matrix = zeros(num);
    for i=1:num
        Neighbors{i} = find(Dists(i,:)<=commdist);
        num_nei = length(Neighbors{i});
        if num_nei > maxDegree
            maxDegree = num_nei;
        end
        edges = edges + num_nei;
        lap_matrix(i,Neighbors{i}) = -1;
        lap_matrix(i,i) =num_nei;
    end
    eig_val = eig(lap_matrix);
    eig_val = sort(eig_val,'ascend');
    algeb_conn = eig_val(2); % algebraic connectivity
    avg_deg = sum(diag(lap_matrix))/num;   % average values

    Network.neighbors = Neighbors;
    Network.loc = loc;
    Network.nodenum = num;      
    Network.maxdegree = maxDegree;
    Network.algeb_conn = algeb_conn;
    Network.avg_deg = avg_deg;
    Network.edges = edges/2; %%undirected graph
    Network.square = square;
    Network.commdist = commdist;
end

function dist = Euclid_Dist(X,Y)
    len = length(X);
    xx = repmat(X,1,len);
    yy = repmat(Y,1,len);    
    dist = sqrt((xx-xx').^2+(yy-yy').^2);
end