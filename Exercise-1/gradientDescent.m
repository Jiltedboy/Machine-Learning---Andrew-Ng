function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    %htheta = X * theta;
    %temp0 = theta(1) - alpha / m * sum((htheta - y) .* X(:,1));
    %temp1 = theta(2) - alpha / m * sum((htheta - y) .* X(:,2)); 
    prediction_error = (X * theta)- y; %Hypthesis htheta(x) - Actual Y  , X is mXn , theta is nX1 
                                       % Prediciton will result in mX1 vector
    temp0=theta(1)-alpha/m * sum((prediction_error).*X(:,1)); %.X(:,1) Pick 1st column from data(ex1data1.txt)(that we are loading)
    temp1=theta(2)-alpha/m * sum((prediction_error).*X(:,2)); % X(:,2)Second column of data
    
    % How its possible to multiply prediction_error -> (mX1) to X(:,1) or X(:,2) -> (mX1) ?
    % Ans is using element wise multiplication . See a example.
    %>> a=[1;2;3]  -> 3x1 vector
    %>> b=[4;5;6]  -> 3x1 vector
    %>> a*b - > error: operator *: nonconformant arguments (op1 is 3x1, op2 is 3x1)
    %Now, >> a.*b ->[4;10;18] 

    theta=[temp0;temp1]; %Simultaneously  updating theta
    

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);


end

end
