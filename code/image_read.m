function inputImage = image_read(imageFilePath)
[~, name, ~] = fileparts(imageFilePath);
label = name(1:2);
load Model.mat
save proposedmodel.mat net label
inputImage=imread(imageFilePath);
end
