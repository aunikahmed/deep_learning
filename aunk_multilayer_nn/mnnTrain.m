%% Convolution Neural Network Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started in building a single.
%  layer convolutional nerual network. In this exercise, you will only
%  need to modify cnnCost.m and cnnminFuncSGD.m. You will not need to 
%  modify this file.

%%======================================================================
%% STEP 0: Initialize Parameters and Load Data
%  Here we initialize some parameters used for the exercise.

% Configuration
imageDim = 28;
numClasses = 10;  % Number of classes (MNIST images fall into 10 classes)
unitsPerLayer = [28*28, 256, 10]; % Number of units per layer including 
                                   %input & output layer


% Load MNIST Train
 addpath ../common;
% images = loadMNISTImages('../common/train-images-idx3-ubyte');
% images = reshape(images,imageDim,imageDim,[]);
% labels = loadMNISTLabels('../common/train-labels-idx1-ubyte');
% labels(labels==0) = 10; % Remap 0 to 10

binary_digits = false;

[train,test] = mnn_load_mnist(binary_digits);

% Add row of 1s to the dataset to act as an intercept term.
train.y = train.y+1; % make labels 1-based.
test.y = test.y+1; % make labels 1-based.


% Initialize Parameters
[weight,bias] = mnnInitParams(unitsPerLayer);
[weightCell, biasCell] = mnn(train.X,train.y,unitsPerLayer,weight,bias);
disp('learning complete ');
val = getAccuracy(weightCell, biasCell, unitsPerLayer, test.X, test.y);

fprintf('Accuracy %d',val);