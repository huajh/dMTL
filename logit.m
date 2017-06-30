function [ theta ] = logit( X,y ,lambda)
%
%  logistic regression

    [nsample, nfeat] = size(X);

    % Add ones to the X data matrix

    X = [ones(nsample, 1) X];

    initial_theta = zeros(nfeat + 1, 1);
    options = optimset('GradObj', 'on', 'MaxIter', 100);
    theta = fmincg (@(t)(lrCostFunction(t, X, (y == 1), lambda)),initial_theta, options);
end


function [J, grad] = lrCostFunction(theta, X, y, lambda)

    m = length(y); 

    z = X*theta;
    theta(1) = 0;
    J = -1./m*(y'*log(sigmoid(z))+(1-y)'*log(1-sigmoid(z)))+lambda./(2*m)*(theta'*theta);
    grad = 1./m * (sigmoid(z)-y)'*X + lambda/m*theta';

    grad = grad(:);

end

function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = 1.0 ./ (1.0 + exp(-z));
end
