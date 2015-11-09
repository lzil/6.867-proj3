function [ Z ] = neuralNetwork( m, X, W1, W1_0, W2, W2_0 )
%neuralNetwork Implements a neural network
%   m: number of hidden layers
%   X: input layer, d x 1
%   W1: weights 1, m x d
%   W1_0: weight 1 bias, m x 1
%   W2: weights 2, k x m
%   W2_0: weight 2 bias, k x 1

d = length(X);
z = zeros(m, 1);
for j = 1:m
    a = 0;
    for i = 1:d
        a = a + W1(j, i) * X(i, 1);
    end
    a = a + W1_0(j, 1);
    z(j, 1) = sigmoid(a);
end

K = length(W2_0);
Z = zeros(K, 1);
for k = 1:K
    a = 0;
    for j = 1:m
        a = a + W2(k, j) * z(j, 1);
    end
    a = a + W2_0(k, 1);
    Z(k, 1) = sigmoid(a);
end

end

function [ out ] = sigmoid( z )
out = 1 / (1 + exp(-z));
end