function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%

%First method
hthetaX = sigmoid(X*theta);   % sigmoid already calculated in previous assignment(sigmoid.m)
index = find(hthetaX >= 0.5); % Find the value where hthetaX is equal to or above 0.5
p(index,1) = 1;               % As index will return a vector , used (index,1) stating that p is also a vector
                              % and assign the index place to value 1, inside the p(zeros(m,1))
                              % As p is initital vector of 0 , the 1 will be assigned at index place and the 0 would 
                              % bw available at the rest of the place that indicates hthetaX<0.5 at that index.
                              
                              

%Second Method
p = round(sigmoid(X*theta)); % Sigmoid always returns value between 0 to 1.
                             % using round function, it can be rounded to 0 or 1 
                             





% =========================================================================


end
