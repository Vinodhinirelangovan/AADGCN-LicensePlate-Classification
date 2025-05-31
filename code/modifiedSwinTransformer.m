function lgraph = modifiedSwinTransformer(inputSize)
% Create a simplified modified Swin Transformer for feature extraction

% Define input layer
inputLayer = imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'zerocenter');

% Patch partitioning (embed patches from 4x4)
patchSize = [4 4];
numChannels = inputSize(3);
embedDim = 96; % embedding dimension

patchEmbedding = convolution2dLayer(patchSize, embedDim, ...
    'Stride', patchSize, ...
    'Name', 'patch_embed', ...
    'Padding', 'same');

% Layer normalization
layerNorm1 = layerNormalizationLayer('Name', 'ln1');

% Swin Transformer block (1 windowed attention block)
windowSize = 7;
numHeads = 4;

% Define custom attention block
attnLayer = swinTransformerBlock(embedDim, numHeads, windowSize, 'swin1');

% Feed-forward network
ffnLayers = [
    fullyConnectedLayer(embedDim*4, 'Name', 'ffn_fc1')
    reluLayer('Name', 'ffn_relu')
    fullyConnectedLayer(embedDim, 'Name', 'ffn_fc2')
];

% Layer normalization
layerNorm2 = layerNormalizationLayer('Name', 'ln2');

% Global average pooling for feature extraction
gap = globalAveragePooling2dLayer('Name', 'gap');

% Create layer graph
lgraph = layerGraph();
lgraph = addLayers(lgraph, inputLayer);
lgraph = addLayers(lgraph, patchEmbedding);
lgraph = addLayers(lgraph, layerNorm1);
lgraph = addLayers(lgraph, attnLayer);
lgraph = addLayers(lgraph, ffnLayers);
lgraph = addLayers(lgraph, layerNorm2);
lgraph = addLayers(lgraph, gap);



end
