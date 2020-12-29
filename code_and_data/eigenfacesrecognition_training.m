function [k_eigenvectors] = eigenfacesrecognition_training(X, k)
%EIGENFACESRECOGNITION_TRAINING perform training for face recognition with 
%eigenfaces  
%   Calculate k first eigenvectors of the data included in X
%
%   Parameters:
%       X: training data in form of matrix with respect to which the  
%         eigenvectors are to be calculated. Point-per-row format (i.e. face  
%         pixel representation per row) is expected.
%       k: a single value parameter specifying the number of first eigenvectors  
%         to be returned
%    
%   Output:
%       k_eigenvectors: first k eigenvectors calculated with respect to the
%         training data X, returned as columns of a matrix
  
  T = (X * X') / size(X, 1);
  
  % S_T, V_T are not required, but they need to be captured to ensure proper 
  % format of U_T
  [U_T, S_T, V_T] = svd(T); 
  U = X' * U_T;

  % Crucial: The eigenvectors contained in U need to be normalized (so they 
  % have norms equal to 1). This arises due to the fact that eigen vectors 
  % received by svd for T (originally with norm=1) have been 
  % multiplied by sth (X matrix), so their norm is not 1 anymore.
  U = normc(U);

  % Return only first eigen vectors
  k_eigenvectors = U(:, 1:k);
endfunction
