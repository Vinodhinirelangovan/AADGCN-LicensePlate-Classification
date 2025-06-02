function layers = modified_swin_transformer()
inputSize = [224 224 3];
numClasses = 35; 

layers = [
    imageInputLayer(inputSize,'Name','Input')

    convolution2dLayer(3, 64, 'Stride', 1, 'Padding', 'same', 'Name', 'PatchEmbedding')
    batchNormalizationLayer('Name','BN1')
    reluLayer('Name','ReLU1')
    
    % Simulated Swin Transformer Blocks
    swinBlock('swin1')
    additionLayer(2,'Name','add1') % For auxiliary loss

    swinBlock('swin2')
    additionLayer(2,'Name','add2') % For auxiliary loss

    swinBlock('swin3')

    % Cross Attention
    depthConcatenationLayer(2,'Name','CrossAttention')
    convolution2dLayer(1, 128, 'Name','CrossConv')
    reluLayer('Name','CrossReLU')

    % SE Block
    seBlock(128,'SEBlock')

    globalAveragePooling2dLayer('Name','GAP')
    fullyConnectedLayer(256, 'Name','FC1')
    reluLayer('Name','ReLU2')
    fullyConnectedLayer(numClasses, 'Name','FC2')
    softmaxLayer('Name','Softmax')
    classificationLayer('Name','Output')];

end