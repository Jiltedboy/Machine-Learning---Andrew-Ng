function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

htheta=X*theta;

% Unregularized Cost J
J=(1/(2*m)) * sum((htheta-y).^2);

%Unregularized grad
grad=(1/m)* (X'*(htheta-y));

% Set thetat1 =0 

first_theta_zero= [0; theta(2:length(theta));];  % Setting theta(0)=0 as per the instruction for exercise in pdf. Example..
                                                   % >> a=[2;3;4]
                                                   % >>  a=[0; a(2:length(a));] %says set index 1 as 0 and
                                                   %   from index 2 to lenghth of the a(matxrix) put as it is
                                                   % >>  a will change to [0;3;4]
                                                   % Otherway to do below ---
                                                   %tempTheta = theta;
                                                   %tempTheta(1) = 0;
%Regularized Cost using vectorization
J=J+(lambda/(2*m))*sum(first_theta_zero' * first_theta_zero);

%Regularized Gradient
grad=grad+(lambda/m)*first_theta_zero;



% =========================================================================

grad = grad(:);

end
