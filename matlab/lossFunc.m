function [ out ] = lossFunc( w1, w2, X, Y )
Y = Y';

out = 0;
Kval = size(Y,1);

for i = 1:size(X,1)
    y = Y(:,i);
    x = X(i,:);
    for kthis = 1:Kval
        hk = h(x, w1, w2);
        out = out - y(kthis).*log(hk(kthis))' - (1-y(kthis)).*(1-log(hk(kthis)))';
    end
end
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
z = [z; 1];

K = size(W2,1);
Z = zeros(K, 1);
a2 = zeros(K, 1);
m = m + 1;
for k = 1:K
    a = 0;
    for j = 1:m
        a = a + W2(k, j) * z(j);
    end
    a2(k) = a;
    Z(k) = sigmoid(a);
end

end

function [ out ] = sigmoid( z )
out = (1+exp(-z)).^(-1);
end