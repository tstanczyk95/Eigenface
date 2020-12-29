function [accuracy] = eigenfacesrecognition_test(U, data_normalized, gnd, ...
  test_indices, training_indices)
%EIGENFACESRECOGNITION_TEST perform testing for face recognition with eigenfaces  
%   Calculate test accuracy on the test dataset, using the provided (k) 
%   eigenvectors. Derive the data descriptors and compare their distances
%   appropriately. Classify testing data using the Nearest Neighbour approach
%
%   Parameters:
%       U: matrix contatining eigenvectors (as its columns), based on which
%         data descriptors will be calculated
%       data_normalized: already normalized training data in form of matrix 
%         with respect to which descriptors will be derived and test accuracy 
%         will be measured. Point-per-row format (i.e. face pixel 
%         representation per row) is expected.
%       gnd: a vector of ground truth values indicating actual face numbers for  
%         each face image (provided within the scope of this assignment) 
%       test_indices: a vector of indices describing selected face images as 
%         testing data
%       training_indices: a vector of indices describing selected face images  
%         as training data
%         
%   Output:
%       accuracy: single (floating point) value from range [0, 1] indicating
%         the number of correctly classified (testing) face images over all the 
%         (testing) face images provided
  
  % Extract test data descriptors
  originalX_test = data_normalized(test_indices, :);
  X_test = originalX_test - mean(originalX_test, 1);
  W_test = X_test * U;
  
  % Extract training data descriptors
  originalX = data_normalized(training_indices, :);
  X = originalX - mean(originalX, 1);
  W = X * U; 
  
  correctlyClassifiedSum = 0;  
  for i = [1:size(test_indices, 1)]
    % Take single descriptor corresponding to a sample from the testing set
    testSampleSetIndex = i;
    testSample = W_test(testSampleSetIndex, :);

    % Calculate distances between the descriptors of samples from the training
    % set
    distances = pdist2(testSample, W, 'euclidean');

    % Find the training sample with shortest distance (its nearest neighbour)
    closestTrainingSampleDistanceIndex = find(distances == min(distances));
    closestTrainingSampleDatasetIndex = ...
      training_indices(closestTrainingSampleDistanceIndex);
  
    % Determine the dataset index of the testing sample
    testSampleDatasetIndex = test_indices(testSampleSetIndex);

    % Check whether the testing sample has been classified correctly
    correctlyClassified = ...
      gnd(closestTrainingSampleDatasetIndex) == gnd(testSampleDatasetIndex);
    correctlyClassifiedSum += correctlyClassified;

    %printf("Correctly classified? %d\n", correctlyClassified);
  endfor

  % Calucate the overall accuracy
  accuracy = correctlyClassifiedSum / size(test_indices, 1);
endfunction
