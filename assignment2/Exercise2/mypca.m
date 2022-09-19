function [pc, eigenvalues] = mypca(A)

[~, column] = size(A);

for value = 1:column
    A(:, value) = A(:, value)-mean(A(:, value));
end

covarianceMatrix = cov(A);

[eigenvectors, eigenvalues] = eig(covarianceMatrix);

pc = A * eigenvectors;

[~, index] = sort(diag(eigenvalues), 'descend');

eigenvaluesSorted = eigenvalues(index, index);
% eigenvectorsSorted = eigenvectors(:, index);
pcSorted = pc(:, index);

pc = pcSorted;
eigenvalues = diag(eigenvaluesSorted);
% eigenvectors = eigenvectorsSorted;
end
