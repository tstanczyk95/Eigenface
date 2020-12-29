clear all;
close all;

pkg load image;
pkg load statistics;
pkg load miscellaneous;

% k value can be modified, so as to receive other (visual) results
k = 20;

load ORL_32x32.mat;
load '7Train/7.mat';

% Data normalization
fea_normalized = (fea - min(min(fea))) / (max(max(fea)) - min(min(fea)));

% Extract training data
originalX = fea_normalized(trainIdx, :);
meanVector = mean(originalX, 1);
X = originalX - meanVector;

# Calculate first k eigenvectors
U = eigenfacesrecognition_training(X, k);

% Eigenvector min-max value normalization (for displaying purposes only)
U_min = min(min(U));
U_max = max(max(U));
U_norm = (U - U_min) / (U_max - U_min);
eigenvectorsTitle = strcat('Eigenvectors visualisation, k=', num2str(k));
figure('name', eigenvectorsTitle);
imshow(reshape(U_norm, 32, k * 32));

% Training data descriptors
% Result: nxk (e.g. 280x10) matrix , containing descriptors for each image 
% image descriptor set per row)
W = X * U; 

% Mean face
figure('name', 'Mean face');
imshow(reshape(meanVector, 32,32));

% Image reconstruction
X_rec = W * U' + meanVector;

randomIndexTrainingSet = randi([1, size(X, 1)]);

figure('name', 'Randomly selected training image');
originalImage = reshape(originalX(randomIndexTrainingSet,:), 32, 32);
imshow(originalImage);

reconstructionTitle = strcat('Reconstruction of the drawn image, k=', num2str(k));
figure('name', reconstructionTitle);
reconstructedImage = reshape(X_rec(randomIndexTrainingSet,:), 32, 32);
reconstructedImage = (reconstructedImage - min(min(X_rec))) / (max(max(X_rec)) - min(min(X_rec)));
imshow(reconstructedImage);








