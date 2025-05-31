classdef swinTransformerBlock < nnet.layer.Layer
    properties
        EmbedDim
        NumHeads
        WinSize
    end

    properties (Learnable)
        QWeights
        KWeights
        VWeights
        OutputWeights
    end

    methods
        function layer = swinTransformerBlock(embedDim, numHeads, winSize, name)
            layer.Name = name;
            layer.Description = 'Swin Transformer Block';
            layer.EmbedDim = embedDim;
            layer.NumHeads = numHeads;
            layer.WinSize = winSize;

            scale = sqrt(2 / embedDim);
            layer.QWeights = randn([1 1 embedDim embedDim], 'single') * scale;
            layer.KWeights = randn([1 1 embedDim embedDim], 'single') * scale;
            layer.VWeights = randn([1 1 embedDim embedDim], 'single') * scale;
            layer.OutputWeights = randn([1 1 embedDim embedDim], 'single') * scale;
        end

        function Z = predict(layer, X)
            Q = dlconv(X, layer.QWeights, []);
            K = dlconv(X, layer.KWeights, []);
            V = dlconv(X, layer.VWeights, []);

            [H, W, C, N] = size(Q);
            Q = reshape(Q, [H*W, C, N]);
            K = reshape(K, [H*W, C, N]);
            V = reshape(V, [H*W, C, N]);

            attnScores = pagemtimes(Q, 'none', K, 'transpose') / sqrt(C);
            attnWeights = softmax(attnScores, 2);
            out = pagemtimes(attnWeights, V);

            out = reshape(out, [H, W, C, N]);
            Z = dlconv(out, layer.OutputWeights, []);
        end
    end
end
