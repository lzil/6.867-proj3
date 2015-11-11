function [ out ] = sigmoid( z )
out = (1+exp(-z)).^(-1);
end

