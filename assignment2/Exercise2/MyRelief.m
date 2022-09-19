function [W] = MyRelief(X,Y,m)
    if m > length(X(:,1))
        error("M > number of instances")
    end
    W = zeros(1,length(X(1,:)));
    [~, idx] = datasample(X, m, 'Replace', false);  
    for i = 1:m
        idx(1)
        sample = X(idx(i), :);
        label = Y(idx(i));
        tempX = X;
        tempX(idx(i),:) = [];
        tempY = Y;
        tempY(idx(i)) = [];
        sameclass = tempX((label == tempY), :);
        differentclass = tempX((label ~= tempY), :);
        [~ , sameidx] = min(sum(sample ~= sameclass,2));
        [~ , missidx] = min(sum(sample ~= differentclass,2));
        diffH = double(sameclass(sameidx,:) ~= sample);
        diffM = double(differentclass(missidx,:) ~= sample);
        W = W - diffH/m + diffM/m;
    end
end