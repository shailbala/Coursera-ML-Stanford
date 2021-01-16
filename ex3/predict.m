function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)
%   :::::::::::::::::::::::::SEE NOTES, if confusion::::::::::::::::::::
% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1); %in our case, it is 10 for digits 1-9 and 0

% Return the following variables correctly 
p = zeros(size(X, 1), 1);

% make predictions using pre-learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
    X = [ones(m, 1) X]; %adding bias unit of value 1
    % hidden layer = a
    
    a = sigmoid(Theta1*X');
    % in a, each column is one set of nodes for each data set
    % total q columns corresponds to q rows of X (total no. of data sets)
    
    % add a bias unit as top row
    a = [ ones(1,columns(a)) ; a];
    
    %% FINAL LAYER, h
    h = sigmoid(Theta2*a);
    h = h';
    
    %save the index of the max element, ~ has the max values
    [~, p] = max(h, [], 2);
% =========================================================================


end
