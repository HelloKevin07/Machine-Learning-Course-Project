function ImgBounded = FindBound(Img)
    [H, W] = size(Img);
    upper = H; mupper = H;
    down = 1; mdown = 1;
    left = W; mleft = W;
    right = 1; mright = 1;
    % find left
    for k = 1 : H
        for n = 1 : W
            if Img(k, n) < 0.5
                mleft = n;
                break;
            end
        end
        if mleft < left
            left = mleft;
        end
    end
    % find right
    for k = 1 : H
        for n = W : -1 : 1
            if Img(k, n) < 0.5
                mright = n;
                break;
            end
        end 
        if mright > right
            right = mright;
        end
    end
    % find upper
    for k = 1 : W
        for n = 1 : H
            if Img(n, k) < 0.5
                mupper = n;
                break;
            end
        end
        if mupper < upper
            upper = mupper;
        end
    end
    % find down
    for k = 1 : W
        for n = H : -1 : 1
            if Img(n, k) < 0.5
                mdown = n;
                break;
            end
        end
        if mdown > down
            down = mdown;
        end
    end
    
    ImgBounded = Img(upper : down, left : right);
    ImgBounded = imresize(ImgBounded, [32, 32]);
end