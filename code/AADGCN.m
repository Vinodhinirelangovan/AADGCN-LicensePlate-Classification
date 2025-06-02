function lgraph = AADGCN(featureLength)
inputSize = [featureLength, 1, 1];  
numClasses = 35;

% Define layers
layers = [
    imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')
    
    % Convolutional layer
    convolution2dLayer([3 1], 64, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')

    % Dense block
    depthConcatenationLayer(2,'Name','concat1')
    convolution2dLayer([3 1], 64, 'Padding', 'same', 'Name', 'dense1_conv1')
    reluLayer('Name','dense1_relu1')
    convolution2dLayer([3 1], 64, 'Padding', 'same', 'Name', 'dense1_conv2')
    reluLayer('Name','dense1_relu2')

    % Gated Conv Layer
    convolution2dLayer([1 1], 64, 'Name', 'gate_conv')
    sigmoidLayer('Name', 'gate_sigmoid')
    multiplicationLayer(2, 'Name', 'gated_output')

    % Attention Module (Squeeze-and-Excitation style)
    globalAveragePooling2dLayer('Name','gap1')
    fullyConnectedLayer(16, 'Name', 'att_fc1')
    reluLayer('Name', 'att_relu')
    fullyConnectedLayer(64, 'Name', 'att_fc2')
    sigmoidLayer('Name', 'att_sigmoid')
    multiplicationLayer(2, 'Name', 'attention_output')
    
    % Global Average Pooling
    globalAveragePooling2dLayer('Name','gap')

    % Fully connected & output layer
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name','relu_fc')
    fullyConnectedLayer(numClasses, 'Name', 'fc_out')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','output')
];

% Create layer graph
lgraph = layerGraph(layers);

% Connect dense block outputs
lgraph = connectLayers(lgraph, 'relu1', 'concat1/in1');
lgraph = connectLayers(lgraph, 'dense1_relu2', 'concat1/in2');

% Connect gated output
lgraph = connectLayers(lgraph, 'concat1', 'gate_conv');
lgraph = connectLayers(lgraph, 'concat1', 'gate_sigmoid');
lgraph = connectLayers(lgraph, 'gate_conv', 'gated_output/in1');
lgraph = connectLayers(lgraph, 'gate_sigmoid', 'gated_output/in2');

% Connect attention module
lgraph = connectLayers(lgraph, 'gated_output', 'gap1');
lgraph = connectLayers(lgraph, 'gap1', 'att_fc1');
lgraph = connectLayers(lgraph, 'att_fc2', 'att_sigmoid');
lgraph = connectLayers(lgraph, 'gated_output', 'attention_output/in1');
lgraph = connectLayers(lgraph, 'att_sigmoid', 'attention_output/in2');

% Connect to GAP and final FC
lgraph = connectLayers(lgraph, 'attention_output', 'gap');

end
