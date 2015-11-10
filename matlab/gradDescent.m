function [w1, w1_0, w2, w2_0] = gradDescent(lossFunc, grad, X, Y, w1, w1_0, w2, w2_0, lambda, step, maxIter, threshold)
iterations = 0;
exampleCount = size(Y,1);
ow1 = zeros(size(w1));
ow1_0 = zeros(size(w1_0));
ow2 = zeros(size(w2));
ow2_0 = zeros(size(w2_0));
while abs(lossFunc(w1, w1_0, w2, w2_0, X, Y) - lossFunc(ow1, ow1_0, ow2, ow2_0, X, Y)) > threshold && iterations < maxIter
    ow1 = w1;
    ow1_0 = w1_0;
    ow2 = w2;
    ow2_0 = w2_0;
    exampleNum = mod(iterations, exampleCount) + 1; % may need to change this to random selection
    xexample = X(exampleNum, :);
    yexample = Y(exampleNum, :);
    [w1, w1_0, w2, w2_0] = gradDescentStep(grad, xexample, yexample, ow1, ow1_0, ow2, ow2_0, step, lambda);
end
end


function [w1, w1_0, w2, w2_0] = gradDescentStep(grad, x, y, w1, w1_0, w2, w2_0, step, lambda)
[gradW] = grad(w1, w2, w1_0, w2_0, x, y, lambda);
w1 = gradW(1,:);
w2 = gradW(2,:);
w1 = w1 + step*gradw1;
w1_0 = w1_0 + step*gradw1_0;
w2 = w2 + step*gradw2;
w2_0 = w2_0 + step*gradw2_0;
end