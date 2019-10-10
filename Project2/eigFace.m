function [y] = eigFace(x, U, mean)
% Function for face projection
% y: eigenface (k, 1)
% x: face vector(d, 1)
% U: matrix of eigenvectors (d, k)
% m: mean face
y = U' * (x - mean);
return
end