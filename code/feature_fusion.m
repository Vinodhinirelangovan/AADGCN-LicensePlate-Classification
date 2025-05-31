function lgraph = feature_fusion()
% Define the input layer
inputLayer = imageInputLayer([128 128 3], 'Name', 'input');

% Branch 1: 1x1 Convolution
branch1 = [
    convolution2dLayer(1, 32, 'Padding', 'same', 'Name', 'conv1x1')
    reluLayer('Name', 'relu1x1')
];

% Branch 2: 3x3 Convolution
branch2 = [
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv3x3')
    reluLayer('Name', 'relu3x3')
];

% Branch 3: 5x5 Convolution
branch3 = [
    convolution2dLayer(5, 32, 'Padding', 'same', 'Name', 'conv5x5')
    reluLayer('Name', 'relu5x5')
];

% Branch 4: MaxPooling + 1x1 Conv
branch4 = [
    maxPooling2dLayer(3, 'Stride', 1, 'Padding', 'same', 'Name', 'maxpool')
    convolution2dLayer(1, 32, 'Padding', 'same', 'Name', 'conv1x1_pool')
    reluLayer('Name', 'relu_pool')
];

% Combine layers into layerGraph
lgraph = layerGraph();
lgraph = addLayers(lgraph, inputLayer);
lgraph = addLayers(lgraph, branch1);
lgraph = addLayers(lgraph, branch2);
lgraph = addLayers(lgraph, branch3);
lgraph = addLayers(lgraph, branch4);

% Connect input to branches
lgraph = connectLayers(lgraph, 'input', 'conv1x1');
lgraph = connectLayers(lgraph, 'input', 'conv3x3');
lgraph = connectLayers(lgraph, 'input', 'conv5x5');
lgraph = connectLayers(lgraph, 'input', 'maxpool');

% Concatenate layer
concatLayer = depthConcatenationLayer(4, 'Name', 'concat');

% Add concat and final fusion conv layer
lgraph = addLayers(lgraph, concatLayer);
lgraph = connectLayers(lgraph, 'relu1x1', 'concat/in1');
lgraph = connectLayers(lgraph, 'relu3x3', 'concat/in2');
lgraph = connectLayers(lgraph, 'relu5x5', 'concat/in3');
lgraph = connectLayers(lgraph, 'relu_pool', 'concat/in4');

% Final fusion conv layer (1x1)
fusionLayer = [
    convolution2dLayer(1, 64, 'Padding', 'same', 'Name', 'fuse1x1')
    reluLayer('Name', 'relu_fuse')
];

lgraph = addLayers(lgraph, fusionLayer);
lgraph = connectLayers(lgraph, 'concat', 'fuse1x1');

end