function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%



% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Feed Forward Propogation
a1=[ones(m,1) X]; %5000x401
z2=a1 * Theta1';  %5000x401 * (25x401)' = 5000x25
a2=sigmoid(z2);
a2=[ones(m,1) a2];
z3=a2* Theta2';
a3=sigmoid(z3);

y_matrix= eye(num_labels)(y,:); %This is Important. Suppose if we have Y - > [1,3,2,4,6,5] (Count is 6)
                                %To represent Y , we have to represent each numeric in a Vector. Lets see for the above Y
                                %Example. for 1 - > [1 0 0 0 0 0] -> row 1 6X1 Vector
                                %         for 3 - > [0 0 1 0 0 0] -> row2  6X1 Vector 
                                %         for 2 ->  [0 1 0 0 0 0] -> row3  6X1 Vector
                                %         for 4 ->  [0 0 0 1 0 0] -> row 4 and so on .
                                % Keep in mind, if you have Y , the row size(num_labels) should be at least same or greater than the  
                                % Largest numeric inside Y, Example, Y-[1 2 9] , You have to choose at least 9 row size(num_labels)
% Cost Calculation
J=(1/m)*sum(sum(-y_matrix.*log(a3)-(1-y_matrix).*log(1-a3)));


%Regularizaton

% Exclude Bias i.e., 1st column but keep all as it is. Example how to perform.
%a=[2 3 4;5 6 7]
%>> a(:,2:end)
%ans =

%   3   4
%   6   7
Theta1_temp= Theta1(:, 2:end); 
Theta2_temp = Theta2(:, 2:end);
%Motive is to get a scalar sum of all theta in network.
%Using Theta1_temp(:) -> It converts matrix in a vecotr that helps. 
%Let see an example.
%a=[1 2;3 4] b=[2 3; 5 6]
%a(:)   and a(:)'-> 1 3 2 4
 % 1
 % 3
 % 2
 % 4
%
Entire_Theta_sum=(Theta1_temp(:)' * Theta1_temp(:)+ Theta2_temp(:)' * Theta2_temp(:));
%Regularized Cost.
J=J+ (lambda/(2*m))*Entire_Theta_sum;


%Back Propogation
d3=a3-y_matrix;
%size(delta3)- > 16X4
d2=(d3*(Theta2(:,2:end))).*sigmoidGradient(z2);

Delta1=d2'*a1;
Delta2=d3'*a2;

Theta1_grad=(1/m)*Delta1;
Theta2_grad=(1/m)*Delta2;

Theta1(:,1)=0;
Theta2(:,1)=0;

Theta1_grad=(1/m)*Delta1 +(lambda/m)*Theta1;
Theta2_grad=(1/m)*Delta2 +(lambda/m)*Theta2;






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
grad

end
