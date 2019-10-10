function [pred] = kNN(k, num_classList, X, C, y)
% Function for kNN Classifier 
% pred: predicted label (M, 1)
% k: number of nearest neighbor
% ClassList: list of classes
% X: training set (N, d)
% C: label corresponding to training set (N, 1)
% y: testing set (M, d)
% where N, M are number of samples, d is number of features.

num_y = size(y, 1);
pred = zeros(num_y, 1);

% candidate = zeros(k, num_y);
for i_y = 1 : num_y
    % Find the k nearest neighbors for each testing sample
    distance = sum((X - y(i_y, :)).^2, 2);
    [~, I] = sort(distance);
    class_ordered = C(I);
    candidate = class_ordered(1 : k);
    % Find the major class in k nearest neighbors
    vote_ClassList = zeros(num_classList, 1);
    for i_candidate = 1 : k
        vote_ClassList(candidate(i_candidate)) = vote_ClassList(candidate(i_candidate)) + 1;
    end
    [~, pred(i_y)] = max(vote_ClassList);
end
return
end


