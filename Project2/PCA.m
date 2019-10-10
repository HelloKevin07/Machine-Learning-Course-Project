function [U, Alpha, m] = PCA(X)
% Function for Princple Component Analysis
% U is the matrix of eigenvectors
% Alpha is the matrix of eigenvalues
% X is the input matrix for PCA (N, d)
% where N is number of samples, d is number of features.

m = mean(X, 1);
A = (X - m)';
[V, D] = eig(A' * A);
U = A * V;
Alpha = diag(D);
return
end