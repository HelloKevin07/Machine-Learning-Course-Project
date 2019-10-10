% Unpack the data
src = 'att_faces';
dst = 'data';
unzip(src, dst);

% Data Preprocessing
% To get data matrix A (N, d), where N is num of samples, d is num of
% features (pixels)
dataMat = zeros(400, 56 * 46);
fList = dir('data/s*');
count = 1;
for k = 1 : length(fList)
    fpath = ['data/', fList(k).name];
    imgList = dir([fpath, '/*.pgm']);
    for n = 1 : length(imgList)
        img_raw = imread([fpath, '/', imgList(n).name]);
        img_raw = imresize(img_raw, 0.5);
        img_flatten = reshape(img_raw, 56 * 46, 1);
        dataMat(count, :) = img_flatten';
        count = count + 1;
    end
end
d_reduced = 50;
[U, Alpha, m] = PCA(dataMat);
plot(1:400, log(sort(abs(Alpha), 'descend')))
[Alpha_ordered, I] = sort(real(Alpha), 'descend');
TopKAlpha = Alpha_ordered(1 : d_reduced);
U_ordered = real(U(:, I));
TopKU = U_ordered(:, 1 : d_reduced);
% eigface = reshape(U_ordered(:, 1), 112, 92);
% mface = reshape(m, 112, 92);


% Cross Validation
% Dividing
num_fold = 5;
cvdataMat = zeros(400 / num_fold, d_reduced, num_fold);
cvlabelMat = zeros(400 / num_fold, 1, num_fold);
count = ones(num_fold, 1);
for k = 1 : length(fList)
    fpath = ['data/', fList(k).name];
    imgList = dir([fpath, '/*.pgm']);
    ind = randperm(10);
    tmp = 10 / num_fold;
    for v = 1 : tmp : 10
        for u = v : (v + tmp - 1)
            img_raw = imread([fpath, '/', imgList(ind(u)).name]);
            img_raw = imresize(img_raw, 0.5);
            img_flatten = reshape(img_raw, 56 * 46, 1);
            eigface = eigFace(double(img_flatten), TopKU, double(m'));
            ind_fold = (v + 1) / tmp;
            cvdataMat(count(ind_fold), :, ind_fold) = eigface';
            cvlabelMat(count(ind_fold), 1, ind_fold) = k;
            count(ind_fold) = count(ind_fold) + 1;
        end
    end
end

% Testing
score = zeros(num_fold, 1);
for i_fold = 1 : num_fold
    X_test = cvdataMat(:, :, i_fold);
    y_test = cvlabelMat(:, :, i_fold);
    X_train = [];
    y_train = [];
    for t = 1 : num_fold
        if t ~= i_fold
            X_train = [X_train; cvdataMat(:, :, t)];
            y_train = [y_train; cvlabelMat(:, :, t)];
        end
    end
    pred = kNN(1, 40, X_train, y_train, X_test);
    score(i_fold) = mean(pred == y_test);
end

acc = mean(score);
fprintf('%d-fold cross validation acc = %.3f\n', num_fold, acc);

