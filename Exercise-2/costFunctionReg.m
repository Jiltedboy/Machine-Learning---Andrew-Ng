function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

first_theta_zero= [0; theta(2:length(theta));];  % Setting theta(0)=0 as per the instruction for exercise in pdf. Example..
                                                   % >> a=[2;3;4]
                                                   % >>  a=[0; a(2:length(a));] %says set index 1 as 0 and
                                                   %   from index 2 to lenghth of the a(matxrix) put as it is
                                                   % >>  a will change to [0;3;4]
                                                                                                                                                                                                                                                                                                                        
J=(-1/m)*sum(y.*log(sigmoid(X*theta))+(1-y).*log(1-sigmoid(X*theta))) + (lambda/(2*m))*sum(first_theta_zero.^2); 

%Using Vectorization . We want the vector product to be a scalar
%J=(-1/m)*(y'*log(sigmoid(X*theta))+(1-y)'*log(1-sigmoid(X*theta))) + (lambda/(2*m))*(first_theta_zero'*first_theta_zero);

prediction= (sigmoid(X * theta))-y;
grad= (1/m)*(X'*prediction)+(lambda/m)*first_theta_zero; % Why X'*prediction ?   As prediction would be mX1 vector, 
                                                         % so to multiply  prediction to X( X is mXn -> It came from ex2data.txt),
                                                         % X should be transpose then it would be nXm .
                                                         % This will help to multiply  X'*prediction (nXm * mX1)



% =============================================================

end
