clc;
clear all;
close all;

% --- Configuration ---
imageFilePath = 'AP1.jpg'; % Make sure this path is correct
xmlFilePath = 'AP1.xml';   % Make sure this path is correct
image_read(imageFilePath);
inputImage=image_read(imageFilePath);

%%Preprocessing using ECLACHE
figure()
imshow(inputImage)
title('Input Image')
[EnhancedImage] = ECLACHE(inputImage);

figure()
imshow(EnhancedImage)
title('Preprocessed Image')

EnhancedImage = imresize(EnhancedImage, [224 224]); 

%%Feature extraction using modified swin transformer
load featuremodel.mat

Extracted_feature = predict(net, EnhancedImage);

feature1=Extracted_feature;
feature2=Extracted_feature;
feature3=Extracted_feature;

%%Feature fusion using pyramid fusion method
fused_features = convolutionalPyramidFusion(feature1, feature2, feature3);

%classification using Attention assisted dense gated convolutional network
%%classification model
load('proposedmodel.mat');  

[labe1, scores] = classify(net, fused_features);

% Display result
fprintf('Predicted label: %s\n', string(label));