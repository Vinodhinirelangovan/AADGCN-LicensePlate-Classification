clc;
clear all;
close all;

% --- Configuration ---
imageFilePath = 'Data/AP1.jpg'; % Make sure this path is correct
xmlFilePath = 'Data/AP1.xml';   % Make sure this path is correct
image_read(imageFilePath);
inputImage=image_read(imageFilePath);

% inputImage=imresize(inputImage,[256 256]);
figure()
imshow(inputImage)
title('Input Image')
[EnhancedImage LuminanceImage] = ECLACHE(inputImage);

figure()
imshow(uint8(LuminanceImage))
title('Luminance Image')

figure()
imshow(EnhancedImage)
title('Preprocessed Image')

EnhancedImage = imresize(EnhancedImage, [128 128]); 

%%Feature extraction and fusion model
load featuremodel.mat

Extracted_feature = predict(net, EnhancedImage);

reshaped_feature = imresize(Extracted_feature, [128 128]); 

%%classification model
load('proposedmodel.mat');  

[labe1, scores] = classify(net, reshaped_feature);

% Display result
fprintf('Predicted label: %s\n', string(label));

