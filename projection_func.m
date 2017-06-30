function [ M ] = projection_func( S0, h )
%SOLVE_EIGEN_OPT Summary of this function goes here
%   
    [p,~] = size(S0);
    [P1,D,~] = svd(S0);    
    x = quad_kanpsack_brucker( 1/2*ones(p,1),diag(D),ones(p,1),h,zeros(p,1),ones(p,1));
    M = P1*diag(x)*P1';    
end

