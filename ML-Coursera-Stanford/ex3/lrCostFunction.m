function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta, 1);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%  Use this function: sigmoid(X * theta)
    h = sigmoid(X*theta);
    J = (1/m)*(-y'*log(h) -(1-y)'*log(1-h)) + (lambda/(2*m))*sum(theta(2:n,:).^2);
    
    grad = ((1/m)*X'*(h-y)); %unregularized gradient
    g = (lambda/m)*theta; %regularization term
    grad(2:n,:) = grad(2:n,:) + g(2:n,:); %combine both for theta2 to thetaN
% =============================================================

grad = grad(:);

end
