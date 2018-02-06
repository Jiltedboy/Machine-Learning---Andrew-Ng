function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

m=size(X,1);      # Size of the training Example X
for i=1:m         #Iterate Through every example
  minDistance = 10^6;  # Initialize any negative / Larg Random Size
  for k=1:K             # Iterate Through initail set of centroid as initialized . Check ex7.m
    distance= sum((X(i,:)-centroids(k,:)).^2); # Calculate the distance between the example and the centroid positions 
                                               # Square them and do the sum as per the formula 
                                               
    #Compare the distance with minDistance and assign distance to minDistance if distance < minDistance
    #to find the index of the closed centroid to the example
    if distance < minDistance                  
      minDistance=distance;
      idx(i)=k;
    end
  end
  
end
idx  



% =============================================================

end

