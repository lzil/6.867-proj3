function [ dw1, dw2 ] = gradNN( w1, w2, x, y, lambda )
[ hstuff, z, a1, a2] = h(x,w1,w2)
dJdh = -y'.*((hstuff).^(-1)) + (1-y)'.*((1-hstuff).^(-1)) + 2*lambda*normFunc(w1, w2)
delta2 = dJdh.*a2.*(exp(a2) + 1).^(-1)
dw2 = delta2*z'
delta1 = (delta2'*w2).*(a1.*(exp(a1) + 1).^(-1))'
dw1 = delta1'*x
end

function [ out ] = normFunc (w1, w2)
out = norm(w1, 'fro') + norm(w2, 'fro');
end

function [ Z, z, a1, a2 ] = h( X, W1, W2 )
%h Implements a neural network
%   X: input layer, d x 1
%   W1: weights 1, m x d
%   W1_0: weight 1 bias, m x 1
%   W2: weights 2, k x m
%   W2_0: weight 2 bias, k x 1

d = length(X);
m = size(W1,1);
z = zeros(m, 1);
a1 = zeros(m,1);
for j = 1:m
    a = 0;
    for i = 1:d
        a = a + W1(j, i) * X(i);
    end
    a1(j) = a;
    z(j) = sigmoid(a);
end

K = size(W2,1);
Z = zeros(K, 1);
a2 = zeros(K, 1);
for k = 1:K
    a = 0;
    for j = 1:m
        a = a + W2(k, j) * z(j);
    end
    a2(k) = a;
    sigmoid(a)
    Z(k) = sigmoid(a)
end

end

function [ out ] = sigmoid( z )
out = 1 / (1 + exp(-z));
end