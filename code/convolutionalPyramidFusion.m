function fused_features = convolutionalPyramidFusion(feat1, feat2, feat3)
    
    % ---- Step 1: Initial Convolution ----
    feat1_conv = conv_layer(feat1);
    feat2_conv = conv_layer(feat2);
    feat3_conv = conv_layer(feat3);

    % ---- Step 2: Top-Down Pathway ----
    % P1 stays the same
    P1 = up_block(feat1_conv);
    
    % P2: Add P1 with upsampled feat2_conv
    up_feat1 = imresize(feat1_conv, size(feat2_conv(:,:,1)), 'nearest');
    P2_input = feat2_conv + up_feat1;
    P2 = up_block(P2_input);
    
    % P3: Add P2 with upsampled feat3_conv
    up_feat2 = imresize(P2_input, size(feat3_conv(:,:,1)), 'nearest');
    P3_input = feat3_conv + up_feat2;
    P3 = up_block(P3_input);

    % ---- Step 3: Concatenate and Final Output ----
    P1_up = imresize(P1, size(P3(:,:,1)), 'nearest');
    P2_up = imresize(P2, size(P3(:,:,1)), 'nearest');
    
    fused_features = cat(3, P1_up, P2_up, P3); % Concatenate along 3rd dimension
end

function out = conv_layer(in)
    h = fspecial('gaussian', [3, 3], 0.5); 
    out = zeros(size(in));
    for i = 1:size(in,3)
        out(:,:,i) = conv2(in(:,:,i), h, 'same');
    end
end

function out = up_block(in)
    % Conv + BN + ReLU + Upsample
    out = conv_layer(in);       % Conv
    out = batch_norm(out);      % BN
    out = max(0, out);          % ReLU
    out = imresize(out, 2, 'nearest');  % Upsample by factor of 2
end

function out = batch_norm(in)
    mu = mean(in(:));
    sigma = std(in(:));
    out = (in - mu) / (sigma + 1e-5);
end
