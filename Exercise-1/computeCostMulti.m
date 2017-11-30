function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

 htheta=X*theta;  % X is mXn matrix , theta is nx1(Used all over in this course)vector i.,e colum vector
                  %  htheta will result in mX1 vector.
 J=(1/(2*m))* sum(((htheta-y).^2)); % .^2 elementwise square



% =========================================================================

end
