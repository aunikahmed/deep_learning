function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  
  
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
  theta=[theta, zeros(n,1)];
 
  var1 = exp(theta' * X);
  sumOverColVar1 = sum(var1,1);
  var2 = bsxfun(@rdivide, var1 , sumOverColVar1);
  
  var3 = log(var2);
 
  I = sub2ind(size(var3),  y, 1:size(var3,2));
  values = var3(I);
  f = - sum(values);  

  var4 = zeros(size(var3));
  var4(I) = 1; 
  gg = X * (var4 - var2)';
  g = - gg(:,1:9); 
  %
  
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  %}
  g=g(:); % make gradient a vector for minFunc

