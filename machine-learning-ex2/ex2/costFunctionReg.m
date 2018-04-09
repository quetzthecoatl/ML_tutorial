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
stheta = size(theta)


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

   h = sigmoid(X*theta);
    diff = h - y;
	fprintf("h = ")
	disp(h)
	%x = X(:,2);
	%fprintf("x = ")
	%disp(x)
	grad = (X'*(diff))/m + (lambda/m).*theta;
	temp = (X'*(diff))/m
	grad(1) = temp(1)

	J = (-y'*log(h) - (1-y)'*log(1-h))/m;
	theta_subset = theta(2:stheta(1,1))
	regularized_add = (lambda/(2*m))*(sum(theta_subset.^2))
	J = J + regularized_add




% =============================================================

end
