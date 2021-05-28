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

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
        %% expand y into y-matrix with 10 columns
        y_matrix = eye(num_labels)(y,:);
        % size(y_matrix)  % size is m x 10
        
        % 2 layers
        %% X has size m x 400, add bias layer
        a1 = [ones(m,1) X];  %% now size of X = m rows, 401 columns
        z2 = a1*Theta1'; %% Theta1 has size 25 x 401
        % size(z2); m x 25
        a2 = sigmoid(z2); % layer 2 activation nodes
        
        %% Add a bias unit to a2, new size m x 26
        a2 = [ones(m,1) a2];  % m = no. of training examples, no. of rows
        z3 = a2*Theta2';  %% Theta2 has size 10 x 26
        
        a3 = sigmoid(z3);  %% Size m x 10, same as size of y
        h = a3;
        
        %% why using .*?, See:
        %https://www.coursera.org/learn/machine-learning/discussions/all/threads/AzIrrO7wEeaV3gonaJwAFA
        J = (1/m)*sum(sum((-y_matrix.*log(h)) -((1-y_matrix).*log(1-h))));
        
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
        % % 2 layers, a1 size m x 401, z2 has size m x 25
        % Theta1 has size 25 x 401
        % a2, new size m x 26, 
        
        
        %$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        %Code not working from here
        a2 = [ones(m,1) a2];  % m = no. of training examples, no. of rows
        z3 = a2*Theta2';  %% Theta2 has size 10 x 26
        
        a3 = sigmoid(z3);  %% Size m x 10, same as size of y
        h = a3;
        delSum2 = 0;
        delSum3 = 0;

        delta3 = a3 - y_matrix;  % Size m x 10
        delta2 = (delta3*Theta2)(2:end).*sigmoidGradient(z2); % Size 25 x 401

        delSum2 = delSum2 + delta2'*a1(:, 2:end); % size 400 x 25
        delSum3 = delSum3 + delta3'*a2(:, 2:end); % size 25 x 10

        Theta1_grad = delSum2'/m;
        Theta2_grad = delSum3'/m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
% -------------------------------------------------------------

        % regularization for cost
        reg = (lambda/(2*m))*(sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
        %size(J)
        J = J+reg;
        
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
