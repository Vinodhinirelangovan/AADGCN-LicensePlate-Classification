function layer = swinBlock(name)
    layer = [
        convolution2dLayer(3, 64, 'Padding','same','Name',[name '_conv'])
        batchNormalizationLayer('Name',[name '_bn'])
        reluLayer('Name',[name '_relu'])
    ];
end
