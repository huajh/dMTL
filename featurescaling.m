function [ X_norm,Xmin,Xmax ] = featurescaling( X )
%FEATURESCALING Summary of this function goes here
%   Detailed explanation goes here
Xmin = min(X,[],1);
Xmax = max(X,[],1);

X_norm = bsxfun(@minus, X,Xmin);

X_norm = bsxfun(@rdivide,X_norm,Xmax-Xmin);

end

