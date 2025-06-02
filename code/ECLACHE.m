 
function [enhancedColorImage,luminanceChannelDouble] = ECLACHE(inputImage)


% --- Configuration Parameters 
enable_denoising = true; % Set to true to apply denoising
denoising_sigma = 0.8;   % Standard deviation for Gaussian denoising filter (smaller = less blur)

clahe_clip_limit = 0.02; % CLAHE ClipLimit (0.01-0.03 common; higher = more contrast, more noise)
clahe_num_tiles = [8 8]; % CLAHE NumTiles (e.g., [8 8], [16 16]; smaller tiles = more local adaptation)

enable_sharpening = true; % Set to true to apply sharpening
sharpening_amount = 0.8;  % Sharpening amount (0.1-2.0 common; higher = more sharp, more artifacts)
% -------------------------------------------------------------------

fprintf('--- Starting Color Image Enhancement Process ---\n');

% Ensure the input is an RGB image
if size(inputImage, 3) ~= 3
    error('Input image must be an RGB color image (3 channels).');
end

% 1. Convert RGB to YCbCr color space and extract luminance channel
ycrcbImage = rgb2ycbcr(inputImage);
luminanceChannel = ycrcbImage(:,:,1);

% Convert luminance channel to double for calculations and scale to 0-255
luminanceChannelDouble = double(luminanceChannel);
luminanceChannelDouble = (luminanceChannelDouble - 16) / (235 - 16) * 255;
luminanceChannelDouble(luminanceChannelDouble < 0) = 0;
luminanceChannelDouble(luminanceChannelDouble > 255) = 255;
luminanceChannelDouble = round(luminanceChannelDouble);

%Denoising ---
if enable_denoising
    fprintf('Applying Gaussian denoising (sigma=%.1f)...\n', denoising_sigma);
    luminanceChannelDouble = imgaussfilt(luminanceChannelDouble, denoising_sigma);
    luminanceChannelDouble = max(0, min(255, round(luminanceChannelDouble))); % Keep in 0-255 range
end

[rows, cols] = size(luminanceChannelDouble);
numPixels = rows * cols;
intensityLevels = 256; % Assuming 8-bit equivalent luminance values (0-255)

%% Step 1, 2, 3: Apply Cumulative Histogram Equalization (CHE) to Luminance Channel

fprintf('Step 1-3: Applying Cumulative Histogram Equalization (CHE) to luminance channel...\n');

histogram = zeros(1, intensityLevels);
for i = 1:rows
    for j = 1:cols
        pixelValue = luminanceChannelDouble(i, j) + 1;
        histogram(pixelValue) = histogram(pixelValue) + 1;
    end
end
normalizedHistogram = histogram / numPixels;
cdf = cumsum(normalizedHistogram);
L_minus_1 = intensityLevels - 1;

cheLuminance = zeros(rows, cols);
for i = 1:rows
    for j = 1:cols
        originalPixelValue = luminanceChannelDouble(i, j);
        transformedPixelValue = round(L_minus_1 * cdf(originalPixelValue + 1));
        cheLuminance(i, j) = transformedPixelValue;
    end
end
cheLuminance = uint8(cheLuminance);
fprintf('   CHE applied to luminance. Ready for CLAHE.\n');

%% Step 4: Apply CLAHE to the CHE-processed luminance channel

fprintf('Step 4: Applying CLAHE to the CHE-processed luminance channel (ClipLimit=%.2f, NumTiles=[%d %d])...\n', ...
    clahe_clip_limit, clahe_num_tiles(1), clahe_num_tiles(2));

claheLuminance = adapthisteq(cheLuminance, 'ClipLimit', clahe_clip_limit, 'NumTiles', clahe_num_tiles);

fprintf('   CLAHE applied to luminance. Ready for final CHE.\n');

%% Step 5: Repeat steps 2 and 3 to create the final enhanced luminance channel

fprintf('Step 5: Applying final CHE to the CLAHE-processed luminance channel...\n');

claheLuminanceDouble = double(claheLuminance);

claheHistogram = zeros(1, intensityLevels);
for i = 1:rows
    for j = 1:cols
        pixelValue = claheLuminanceDouble(i, j) + 1;
        claheHistogram(pixelValue) = claheHistogram(pixelValue) + 1;
    end
end
normalizedClaheHistogram = claheHistogram / numPixels;
claheCdf = cumsum(normalizedClaheHistogram);

finalEnhancedLuminance = zeros(rows, cols);
for i = 1:rows
    for j = 1:cols
        originalPixelValue = claheLuminanceDouble(i, j);
        transformedPixelValue = round(L_minus_1 * claheCdf(originalPixelValue + 1));
        finalEnhancedLuminance(i, j) = transformedPixelValue;
    end
end
finalEnhancedLuminance = uint8(finalEnhancedLuminance);
fprintf('   Final CHE applied to luminance.\n');

%% Recombine and Convert back to RGB

fprintf('--- Recombining channels and converting back to RGB ---\n');

ycrcbImageEnhanced = ycrcbImage;
ycrcbImageEnhanced(:,:,1) = finalEnhancedLuminance;

enhancedColorImage = ycbcr2rgb(ycrcbImageEnhanced);

%Post-processing: Sharpening ---
if enable_sharpening
    fprintf('Applying sharpening (Amount=%.1f)...\n', sharpening_amount);
    enhancedColorImage = imsharpen(enhancedColorImage, 'Amount', sharpening_amount);
end

fprintf('Color image enhancement complete.\n');

end