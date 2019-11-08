function [acc] = KernelSVM(X_train, y_train, X_test, y_test, numClass, C)
    d = 2;
    n = size(X_train, 1);
    alphay = zeros(numClass, n);
    B = zeros(numClass, 1);
    
    for idx_class = 1 : numClass
        
        ind = (y_train == idx_class);
        y = -ones(n, 1);
        y(ind) = 1;
        
        H = y * y' .* PolynomialKernel(X_train, X_train', d);
        f = -ones(n, 1);
        Aeq = y';
        beq = 0;
        lb = zeros(n, 1);
        ub = C * ones(n, 1);
        
        alpha = quadprog(H, f, [], [], Aeq, beq, lb, ub);

        Kxxt = (X_train * X_train' + 1).^d;
        tmp = y - Kxxt * (alpha .* y);
        b = mean(tmp);
        
        alphay(idx_class, :) = (alpha .* y)';
        B(idx_class, :) = b;
    end
    
    n_test = size(X_test, 1);
    count = 0;
    for idx_sample = 1 : n_test
        sample = X_test(idx_sample, :)';
        Z = alphay * PolynomialKernel(X_train, sample, d) + B;
        [~, pred] = max(Z);
        count = count + (pred == y_test(idx_sample));
    end
    acc = count / n_test;
end