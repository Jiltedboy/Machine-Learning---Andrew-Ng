function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       
mu = mean(X); % mean(X) will return row vector (1Xn). Example below
              % a=[2 4 6;8 10 12] 
              % >> mean(a)
                    ans = 5   7   9
  
sigma = std(X); % 1xn vector . Example below..
                % >> std(a)
                      ans =
                      4.2426   4.2426   4.2426                
X_norm=(X-mu)./sigma; % ./ to divide each element in X-mu by sigma









% ============================================================

end
