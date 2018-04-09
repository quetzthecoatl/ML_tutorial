function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
sz = size(z);
for i=1:sz(1,1)
  for j=1:sz(1,2)
    %fprintf('i = %f and j = %f, and current element = %f\n', i, j,z(i,j));
	g(i,j) = 1/(1+exp(-1*z(i,j)));
  endfor
endfor



% =============================================================

end
