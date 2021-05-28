function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

    % You need to set these values correctly
    X_norm = X;
    mu = zeros(1, size(X, 2));  %store mean here
    sigma = zeros(1, size(X, 2)); %store standard deviation here

    % ====================== YOUR CODE HERE ======================
    % Instructions: First, for each feature dimension, compute the meanrow
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

    m = size(X, 1); % number of rows, size of data set
    % mean for each column, find the size and sum and then divide
    %mu = sum(X, 1)/m; % sum each column of X and divide by size
    mu = mean(X);
    %disp('Mean: ');
    %mu

    %% Standard Deviation
    sigma = std(X)  %computes standard deviation for each column, returns row vector

    for i=1:m,
      X_norm(i,:) = (X(i,:) - mu)./sigma;
    end;
    %disp('X_norm');
    %disp(X_norm);


% ============================================================

end
