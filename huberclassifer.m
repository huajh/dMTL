function [ theta ] = huberclassifer( X,y, lambda )

    % Modified Huber loss
    [n, p] = size(X);    
    X = [ones(n, 1) X];
    
    %initial_theta = zeros(p + 1, 1);    
    %options = optimset('MaxIter', 200);
    %theta = fmincg (@(t)(costfunc(t, X, (y == 1), lambda)),initial_theta, options);         

    % # linear search
    maxiter =1000;    
    old_u = zeros(p+1,1);
    u = old_u;
    tau = 1;
    beta = 0.5;
    old_cost = 0;    
    for i=1:maxiter        
        uhalf = u + (i-2)/(i+1)*(u - old_u);
        old_u = u;        
        [objfunc0, Gu] = costfunc(uhalf, X, y, lambda);
        u = uhalf;
        cost = objfunc0;
        Flag = 0;
        while(1)
            u0 = u - tau*Gu;
            [objfunc, ~] = costfunc(u0, X, y, lambda);
            diffu = u0 - u;
            cmp_err = objfunc0 + trace(Gu'*diffu) + 1/(2*tau)*norm(diffu,'fro')^2 - objfunc;
            if cmp_err > 0
                u = u0;
                cost = objfunc;
                break;
            else
                tau = beta*tau;
            end
            if  sum(diffu.^2)/sum(u0.^2) < 1e-12 % this shows that, the gradient step makes little improvement
                Flag = 1;
                break;
            end
        end
        if Flag || (i>10 && abs(old_cost - cost)/cost < 1e-12)
            break;
        end
        old_cost = cost;
    end
    
    theta = u;
end

function [J, grad] = costfunc(theta, X, y, lambda)

    m = length(y); 
        
    [ cost0, grad0 ] = modihuberloss(X*theta, y); 
    theta(1) = 0;
    J = 1./m*cost0 + lambda*theta'*theta;
    grad = 1./m * X'*grad0 + 2*lambda*theta;        
end

