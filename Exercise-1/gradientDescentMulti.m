function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %
    predictions=X*theta; % mXn * nX1 -> mx1 vector
    error=predictions-y ;% mx1 vector

    delta= X'*error; %  Why X transpose ? X originally mXn , error-> mx1 so , to multiply both move X before error 
                     %  so that when you do X transpose it wil be nXm matrix then X'*error -> nXm * mX1 ->nX1 vector
    
    % I struggled a lot here to make it right. What I understood here that you need to fit the formulas mathematically in
    % correct format. Best way to write the algorithm is on notebook and then make a small matrix case and try to see how
    % its fitting there. If any one has suggestion . Feel Free to provide.
    
    theta= theta- (alpha/m) *delta;
   
                    
    
    
    
    








    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
