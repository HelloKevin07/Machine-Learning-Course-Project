function [W1_n, W2_n, loss] = BackPropOnce(X, Y, W1, W2, lr)
    dW1 = 0 * W1;
    dW2 = 0 * W2;
    loss = 0;
    
    for n = 1 : size(X, 1)
        x = X(n, :)';
        y = Y(n, :)';
        z1 = [x; 1];
        z2 = W1' * z1;
        z3 = [sigmoid(z2); 1];
    
        z4 = W2' * z3;
        o = softmax(z4);
        
        g4 = o - y;
        g3 = W2 * g4;
        g2 = g3 .* z3 .* (1 - z3);
        
        dW1 = dW1 + z1 * g2(1 : end-1, :)';
        dW2 = dW2 + z3 * g4';
        loss = loss - y' * log(o);
    end
    loss = loss / size(X, 1);
    C = 1E-3;
    
    W1_n = (1 - C) * W1 - lr * dW1 / size(X, 1);
    W2_n = (1 - C) * W2 - lr * dW2 / size(X, 1);
    
end