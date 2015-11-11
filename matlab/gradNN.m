function [ dw1, dw2 ] = gradNN( w1, w2, x, y, lambda )
[ hstuff, z, a1, a2] = h(x,w1,w2);
dJdh = -y'.*((hstuff).^(-1)) + (1-y)'.*((1-hstuff).^(-1));

dsigmoid = sigmoid(a2) .* (1 - sigmoid(a2));

delta2 = dJdh.*dsigmoid;
dw2 = delta2*z';
dsigmoid = sigmoid(a1) .* (1 - sigmoid(a1));
m = size(w1,1);
k = size(w2,1);
d1 = zeros(m,1);
for j = 1:m
    s = 0;
    for i = 1:k
        s = s + delta2(i)*w2(i,j)*dsigmoid(j);
    end
    d1(j) = s;
end
dw1 = d1*x;

dw1 = dw1 + 2*lambda*[w1(:,1:end-1) zeros(size(w1,1),1)];
dw2 = dw2 + 2*lambda*[w2(:,1:end-1) zeros(size(w2,1),1)];
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
    sigmoid(a);
    Z(k) = sigmoid(a);
end

end

function [ out ] = sigmoid( z )
out = (1+exp(-z)).^(-1);
end