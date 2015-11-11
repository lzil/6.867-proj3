function [w1, w2] = gradDescent(lossFunc, grad, X, Y, w1, w2, lambda, step, maxIter, threshold)
Y = Y';
iterations = 0;
exampleCount = size(Y,1);
ow1 = zeros(size(w1));
ow2 = zeros(size(w2));
while abs(lossFunc(w1, w2, X, Y) - lossFunc(ow1, ow2, X, Y)) > threshold && iterations < maxIter
    ow1 = w1;
    ow2 = w2;
    exampleNum = ceil(rand*exampleCount);
    %exampleNum = mod(iterations, exampleCount) + 1; % may need to change this to random selection
    xexample = X(exampleNum, :);
    yexample = Y(exampleNum, :);
    [w1, w2] = gradDescentStep(grad, xexample, yexample, ow1, ow2, step, lambda);
    iterations = iterations + 1;
    lossFunc(w1,w2,X,Y)
end
end


function [w1, w2] = gradDescentStep(grad, x, y, w1, w2, step, lambda)
[gradw1, gradw2] = grad(w1, w2, x, y, lambda);
w1 = w1 - step*gradw1;
w2 = w2 - step*gradw2;
end