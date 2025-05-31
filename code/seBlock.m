classdef seBlock < nnet.layer.Layer
    properties
        Channels
        ReductionRatio = 16;
    end

    properties (Learnable)
        FC1Weights
        FC1Bias
        FC2Weights
        FC2Bias
    end

    methods
        function layer = seBlock(channels, name)
            layer.Name = name;
            layer.Description = 'Squeeze-and-Excitation Block';
            layer.Channels = channels;

            hiddenDim = floor(channels / layer.ReductionRatio);
            layer.FC1Weights = randn([1 1 channels hiddenDim], 'single') * sqrt(2/channels);
            layer.FC1Bias = zeros([1 1 hiddenDim], 'single');
            layer.FC2Weights = randn([1 1 hiddenDim channels], 'single') * sqrt(2/hiddenDim);
            layer.FC2Bias = zeros([1 1 channels], 'single');
        end

        function Z = predict(layer, X)
            % Global average pooling
            S = mean(mean(X,1),2);  % 1x1xC x N

            % FC → ReLU → FC → Sigmoid
            S = dlconv(S, layer.FC1Weights, layer.FC1Bias);
            S = relu(S);
            S = dlconv(S, layer.FC2Weights, layer.FC2Bias);
            S = sigmoid(S);

            Z = X .* S;  % Scale original input
        end
    end
end
