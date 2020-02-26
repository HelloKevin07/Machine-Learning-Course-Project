%% Prepare training data and testing data
dataPath = 'Img/Sample*';
numClass = 62;
numEachClass = 55;
numTrain = 50;
numTest = 5;
imgH = 900;
imgW = 1200;
resizeRatio = 1/10;
rszSize = floor(32 * 32);
X_train = zeros(numClass * numTrain, rszSize);
y_train = zeros(numClass * numTrain, numClass);
X_test = zeros(numClass * numTest, rszSize);
y_test = zeros(numClass * numTest, numClass);
dataList = dir(dataPath);
countTrain = 1;
countTest = 1;
for k = 1 : numClass
    
    classPath = ['Img/', dataList(k).name];
    imgList = dir([classPath, '/img*']);
    randIndex = randperm(numEachClass);
    
    for n = 1 : numTrain
        
        imgRaw = imread([classPath, '/', imgList(randIndex(n)).name], 'png');
        imgRaw = imresize(imgRaw, resizeRatio);
        imgBinary = imgRaw(:, :, 1) / 255;
        imgBinary = 1 - FindBound(imgBinary);
        imgFlatten = reshape(imgBinary, rszSize, 1);
        X_train(countTrain, :) = imgFlatten';
        y_train(countTrain, k) = 1;
        countTrain = countTrain + 1;
    end
    
    for n = (numTrain + 1) : numEachClass
        
        imgRaw = imread([classPath, '/', imgList(randIndex(n)).name], 'png');
        imgRaw = imresize(imgRaw, resizeRatio);
        imgBinary = imgRaw(:, :, 1) / 255;
        imgBinary = 1 - FindBound(imgBinary);
        
        imgFlatten = reshape(imgBinary, rszSize, 1);
        X_test(countTest, :) = imgFlatten';
        y_test(countTest, k) = 1;
        countTest = countTest + 1;
    end
end

%% PCA
d_reduced = 150;
[U, Alpha, m] = PCA(X_train);
semilogy(1 : length(Alpha), sort(abs(Alpha), 'descend'));
title('Eigenvalues (descended order)');
xlabel('i-th')
ylabel('value')
[Alpha_ordered, I] = sort(real(Alpha), 'descend');
TopKAlpha = Alpha_ordered(1 : d_reduced);
U_ordered = real(U(:, I));
TopKU = U_ordered(:, 1 : d_reduced);
%% D Reduction
X_train = (X_train - m) * TopKU;
X_test = (X_test - m) * TopKU;

%% Train
lr = 1;
numLayer1 = d_reduced;
numLayer2 = 100;
numLayer3 = numClass;
numLayers = [numLayer1, numLayer2, numLayer3];
W1 = rand(d_reduced + 1, numLayer2);

W2 = rand(numLayer2 + 1, numLayer3);

batchSize = 310;
numEpoch = 130;
numBatch = floor(size(X_train, 1) / batchSize);

for epoch = 1 : numEpoch
    randIndex_Train = randperm(size(X_train, 1));
    for k = 1 : numBatch
        batchIndex = (batchSize * (k - 1) + 1) : (batchSize * k);
        batchIndex = randIndex_Train(batchIndex);
        x_batch = X_train(batchIndex, :);
        y_batch = y_train(batchIndex, :);
        [W1, W2, loss] = BackPropOnce(x_batch, y_batch, W1, W2, lr);
        
    end
    fprintf("epoch: %d/%d batch: %d/%d loss: %.5f\n", epoch, numEpoch, k, numBatch, loss);
end

%% Test
accTrain = 0;
for n = 1 : size(X_train, 1)
    x = X_train(n, :)';
    y = y_train(n, :)';
    y_hat = ThreeLayerNN(W1, W2, x);
    [~, pred] = max(y_hat);
    [~, label] = max(y);
    accTrain = accTrain + (pred == label);
end
accTrain = accTrain / size(X_train, 1);
fprintf("Training set acc: %.3f\n", accTrain);

accTest = 0;
for n = 1 : size(X_test, 1)
    x = X_test(n, :)';
    y = y_test(n, :)';
    y_hat = ThreeLayerNN(W1, W2, x);
    [~, pred] = max(y_hat);
    [~, label] = max(y);
    accTest = accTest + (pred == label);
end
accTest = accTest / size(X_test, 1);
fprintf("Testing set acc: %.3f\n", accTest);


