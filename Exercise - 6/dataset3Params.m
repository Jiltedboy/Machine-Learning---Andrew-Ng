function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_OR_Sigma_Possibilities = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30]; 
C_Temp = C;
sigma_Temp= sigma;
Error_Asumption = 10^6; % Select any big integer , its like initializtion .

for i = 1:length(C_OR_Sigma_Possibilities)
	for j = 1:length(C_OR_Sigma_Possibilities)
		model = svmTrain(X, y, C_OR_Sigma_Possibilities(i), @(x1, x2) gaussianKernel(x1, x2, C_OR_Sigma_Possibilities(j)));
    predictions= svmPredict(model,Xval);
    Error=mean(double(predictions ~= yval)); %Compute the error between your predictions and yval
    
    if Error < Error_Asumption
      Error_Asumption=Error;
      C_Temp=C_OR_Sigma_Possibilities(i);
      Sigma_Temp =  C_OR_Sigma_Possibilities(j);
     end
   end
end
  
C= C_Temp;
sigma= Sigma_Temp;
       






% =========================================================================

end
