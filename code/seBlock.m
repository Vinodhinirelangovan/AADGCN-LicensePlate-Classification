function layers = seBlock(numChannels, name)
    layers = [
        globalAveragePooling2dLayer('Name',[name '_gap'])
        fullyConnectedLayer(round(numChannels/16), 'Name',[name '_fc1'])
        reluLayer('Name',[name '_relu'])
        fullyConnectedLayer(numChannels, 'Name',[name '_fc2'])
        sigmoidLayer('Name',[name '_sigmoid'])
        multiplicationLayer(2, 'Name',[name '_scale']) % SE scaling
    ];
end
