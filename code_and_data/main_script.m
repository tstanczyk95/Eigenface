clear all;
close all;

pkg load image;
pkg load statistics;
pkg load miscellaneous;

load ORL_32x32.mat;

% Data normalization
fea_normalized = (fea - min(min(fea))) / (max(max(fea)) - min(min(fea)));

% Prepare plot
figure('name', 'Various results compared');
hold on;

trainIndexFiles = ['3Train/3.mat'; '5Train/5.mat'; '7Train/7.mat'];
for i = [1:size(trainIndexFiles, 1)]
  % Each time, the same variable names will be used (overwritting)
  load(trainIndexFiles(i, :)); 
  printf("Loaded file: %s\n\n", trainIndexFiles(i, :));
  
  k_values = [5 10 15 20 25 30 35 40 45 50];
  accuracy_values = zeros(size(k_values));
  
  for j = 1:size(k_values, 2)
    % Extract training data
    originalX = fea_normalized(trainIdx, :);
    meanVector = mean(originalX, 1);
    X = originalX - meanVector;

    % Calculate first k eigenvectors
    U = eigenfacesrecognition_training(X, k_values(j));

    % Measure the accuracy
    testAccuracy = eigenfacesrecognition_test(U, fea_normalized, gnd, testIdx, trainIdx);
    printf("k=%d\nAccuracy: %.6f\n\n", k_values(j), testAccuracy);
    accuracy_values(j) = testAccuracy;
    
  endfor
  printf("- - - - - - - - - -\n\n");
  
  % Plot current data 
  plot(k_values, accuracy_values);
endfor

% Polish the plot
title("Various results comparison");
legend(trainIndexFiles, 'Location', 'SouthEast');
xlabel("k values");
ylabel("Accuracy");
hold off;















