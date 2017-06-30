function [ pred ] = predictlogit( theta,X )

    m = size(X, 1); % Number of training examples

    pred = zeros(m, 1);
    
    X = [ones(m, 1) X];
    
    prob = sigmoid(X*theta);

    pred(prob >=0.5) = 1;

end

function g = sigmoid(z)

    g = 1.0 ./ (1.0 + exp(-z));
    
end

