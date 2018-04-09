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


% Part 1: Feedforward the neural network and return the cost 

X = [ones(m, 1) X];
a2 = sigmoid(Theta1*X')';
m_1 = size(a2, 1);
a2 = [ones(m_1, 1) a2];
h = sigmoid(Theta2 * a2');
[val, p] = max(h);
y_labels = zeros(num_labels, m); 
%convert y from vector to matrix of every column having one 1 and rest all zeroes
for i=1:m,
  y_labels(y(i),i)=1;
end
J = (sum(sum((-y_labels).*log(h) - (1-y_labels).*log(1-h) )))/m;

%regularization - do not regularize bias
m_t1 = size(Theta1, 2);
m_t2 = size(Theta2, 2);
Theta1_unbiased = Theta1(:,2:m_t1);
Theta2_unbiased = Theta2(:,2:m_t2);

regularized_add = (lambda/(2*m)) * (sum(sum(Theta1_unbiased.^2)) + sum(sum(Theta2_unbiased.^2)));
J = J + regularized_add;


%Part 2: Implement the backpropagation algorithm to compute the gradients

for i=1:m,
a1 = X(i,:);
z2 = Theta1*a1';
a2 = sigmoid(z2);
a2 = [1 ; a2];
z3 = Theta2*a2; %transpose giving error - check
h = sigmoid(z3);

yi = y_labels(:,i);
d_3 = h - yi;
%bias
z2 = [1 ; z2];
d_2 = (Theta2'*d_3).*sigmoidGradient(z2);

%Accumulate the gradient from this example using the following formula. Note that you should skip or remove (2)0 . In Octave/MATLAB, removing (2) 0 corresponds to delta 2 = delta 2(2:end).

d_2 = d_2(2:end); 

Theta2_grad = Theta2_grad + d_3*a2';
Theta1_grad = Theta1_grad + d_2*a1; %transpose giving error - check

endfor

%Part 3: Implement regularization with the cost function and gradients.

Theta1_grad(:, 1) = Theta1_grad(:, 1)/m;	
Theta1_grad(:, 2:end) = (Theta1_grad(:, 2:end) + (lambda*Theta1(:, 2:end)))/m;
		
Theta2_grad(:, 1) = Theta2_grad(:, 1)/m;	
Theta2_grad(:, 2:end) = (Theta2_grad(:, 2:end) + (lambda*Theta2(:, 2:end)))/m;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
