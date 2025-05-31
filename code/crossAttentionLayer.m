classdef crossAttentionLayer < nnet.layer.Layer
    properties
        EmbedDim
    end

    properties (Learnable)
        QWeights
        KWeights
        VWeights
        OutputWeights
    end

    methods
        function layer = crossAttentionLayer(embedDim, name)
            layer.Name = name;
            layer.Description = 'Cross Attention Layer';
            layer.EmbedDim = embedDim;

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

            scores = pagemtimes(Q, 'none', K, 'transpose') / sqrt(C);
            attn = softmax(scores, 2);
            context = pagemtimes(attn, V);

            context = reshape(context, [H, W, C, N]);
            Z = dlconv(context, layer.OutputWeights, []);
        end
    end
end
