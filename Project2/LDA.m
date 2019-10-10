function [W] = LDA(X)
% Function for Linear Discriminant Analysis
% W: discriminant functions (d, c - 1)
% X: input data for LDA (N, d, c)
% where N is num of samples of each class, d is num of features, c is num 
% of classes.

N = size(X, 1);
d = size(X, 2);
c = size(X, 3);
m_total = mean(c, 1)';
m = zeros(c, d);

for i = 1 : c
    m(i, :) = mean(X(:, :, i), 1);
end

Sb = zeros(d, d);
for i = 1 : c
    Sb = Sb + N * (m(i)' - m_total) * (m(i)' - m_total)';
end

Sw = zeros(d, d);
for i = 1 : c
    St = zeros(d, d);
    for j = 1 : N
        St = St + (X(j, :, i) - m(i)') * (X(j, :, i) - m(i)')';
    end
    Sw = Sw + St;
end

[W, lambda] = eig(pinv(Sw) * Sb);
W = real(W(:, real(diag(lambda)) ~= 0));

return
end