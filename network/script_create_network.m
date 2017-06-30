
close all;
is_createNetwork = true;
nodenum = 1;
if is_createNetwork
    algeb_conn = -1;
    avg_deg =  -1;
    while(avg_deg < 4 ||  algeb_conn < 0.005)
        Conf.square = 2.5;%sqrt(0.24*nodenum);
        Conf.nodenum= nodenum;
        Conf.commdist = 0.8;        
        Network = CreateNetworks(Conf);
        algeb_conn = Network.algeb_conn;
        avg_deg = Network.avg_deg;
    end
    save(['Network' num2str(nodenum) '.mat'], 'Network');
else
    load(['Network' num2str(nodenum) '.mat']);
end

DrawNetworks(Network);

% summary statistics
fprintf('max degree=%d | edges=%d  \naverage_degree=%.2f alge_conn = %.2f\n',...
    Network.maxdegree,Network.edges ,Network.avg_deg, Network.algeb_conn);



