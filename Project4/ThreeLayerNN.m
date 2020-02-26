function y = ThreeLayerNN(W1, W2, x)
    z1 = [x; 1];
    z2 = W1' * z1;
    z3 = [sigmoid(z2); 1];
    z4 = W2' * z3;
    y = softmax(z4);
end