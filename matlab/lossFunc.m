function [ out ] = lossFunc( w1, w1_0, w2, w2_0, X, Y )

out = 0;
Kval = max(Y);

for i = 1:size(X,1)
    y = zeros(Kval);
    y(Y(i,:)) = 1;
    x = X(i,:);
    for kthis = 1:Kval
        hk = h(x, w1, w1_0, w2, w2_0);
        out = out - y(kthis)*log(hk) - (1-y(kthis))*(1-log(hk));
    end
end
end

function [ Z, a1, a2 ] = h( X, W1, W1_0, W2, W2_0 )
%h Implements a neural network
%   X: input layer, d x 1
%   W1: weights 1, m x d
%   W1_0: weight 1 bias, m x 1
%   W2: weights 2, k x m
%   W2_0: weight 2 bias, k x 1

d = length(X);
m = length(W1_0);
z = zeros(m, 1);
a1 = zeros(m,1);
for j = 1:m
    a = 0;
    for i = 1:d
        a = a + W1(j, i) * X(i);
    end
    a = a + W1_0(j);
    a1(j) = a;
    z(j) = sigmoid(a);
end

K = length(W2_0);
Z = zeros(K, 1);
a2 = zeros(K, 1);
for k = 1:K
    a = 0;
    for j = 1:m
        a = a + W2(k, j) * z(j);
    end
    a = a + W2_0(k);
    a2(k) = a;
    Z(k) = sigmoid(a);
end

end

function [ out ] = sigmoid( z )
out = 1 / (1 + exp(-z));
end