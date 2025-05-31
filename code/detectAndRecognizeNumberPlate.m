function detectAndRecognizeNumberPlate(imageFilePath, xmlFilePath)
%detectAndRecognizeNumberPlate Reads an image and an XML annotation file,
%crops the number plate based on XML data, and performs OCR.
%
%   Inputs:
%     imageFilePath - Full path to the image file (e.g., 'AN1.jpg')
%     xmlFilePath   - Full path to the XML annotation file (e.g., 'AN1.xml')

% --- 1. Validate File Paths ---
if nargin < 2
    error('Both imageFilePath and xmlFilePath are required inputs.');
end

if ~exist(imageFilePath, 'file')
    error('Image file not found: %s', imageFilePath);
end

if ~exist(xmlFilePath, 'file')
    error('XML file not found: %s', xmlFilePath);
end

% --- 2. Read XML File and Extract Bounding Box and Name ---
fprintf('--- Reading XML File: %s ---\n', xmlFilePath);
try
    tree = xmlread(xmlFilePath);

    % Get the 'object' node. Assumes one object element for the number plate.
    objectNode = tree.getElementsByTagName('object').item(0);
    if isempty(objectNode)
        error('No ''object'' tag found in the XML file.');
    end

    % Extract the 'name' (number plate string)
    nameNode = objectNode.getElementsByTagName('name').item(0);
    if isempty(nameNode) || isempty(nameNode.getFirstChild)
        error('No ''name'' tag or data found within ''object'' in XML.');
    end
    numberPlateName = char(nameNode.getFirstChild.getData);

    % Extract the 'bndbox' coordinates
    bndboxNode = objectNode.getElementsByTagName('bndbox').item(0);
    if isempty(bndboxNode)
        error('No ''bndbox'' tag found within ''object'' in XML.');
    end

    % Helper function to safely extract numeric data from XML
    extractNumericData = @(parentNode, tagName) str2double(char(parentNode.getElementsByTagName(tagName).item(0).getFirstChild.getData));

    xmin = extractNumericData(bndboxNode, 'xmin');
    ymin = extractNumericData(bndboxNode, 'ymin');
    xmax = extractNumericData(bndboxNode, 'xmax');
    ymax = extractNumericData(bndboxNode, 'ymax');

    fprintf('Successfully extracted data from XML.\n');
    fprintf('  Ground Truth Number Plate Name: %s\n', numberPlateName);
    fprintf('  Ground Truth Bounding Box: [xmin:%d, ymin:%d, xmax:%d, ymax:%d]\n', xmin, ymin, xmax, ymax);

catch ME
    error('Error parsing XML file %s: %s', xmlFilePath, ME.message);
end

% --- 3. Read the Image and Crop the Number Plate Region ---
fprintf('\n--- Reading Image: %s ---\n', imageFilePath);
try
    img = imread(imageFilePath);
catch ME
    error('Error reading image file %s: %s', imageFilePath, ME.message);
end

% Define the bounding box in [x y width height] format for imcrop
% xmin and ymin are already top-left; width = xmax - xmin; height = ymax - ymin
bbox = [xmin ymin (xmax-xmin) (ymax-ymin)];

% Validate and adjust bbox to ensure it's within image bounds
imgWidth = size(img, 2);
imgHeight = size(img, 1);

bbox(1) = max(1, round(bbox(1))); % Ensure xmin is at least 1
bbox(2) = max(1, round(bbox(2))); % Ensure ymin is at least 1
bbox(3) = min(round(bbox(3)), imgWidth - bbox(1) + 1); % Adjust width if it goes beyond image right edge
bbox(4) = min(round(bbox(4)), imgHeight - bbox(2) + 1); % Adjust height if it goes beyond image bottom edge

% Ensure width and height are positive
if bbox(3) <= 0 || bbox(4) <= 0
    error('Calculated bounding box has zero or negative width/height. Check XML coordinates.');
end

cropped_plate = imcrop(img, bbox); % Crop the image using the extracted bounding box

fprintf('Image loaded and number plate cropped using XML data.\n');

% --- 4. Perform OCR on the Cropped Number Plate ---
fprintf('\n--- Performing OCR on cropped plate ---\n');
try
    results = ocr(cropped_plate); % Perform OCR on the cropped region
    ocrText = strtrim(results.Text); % Remove leading/trailing whitespace
    fprintf('OCR completed.\n');
    fprintf('Detected Number Plate (from OCR): %s\n', ocrText);
catch ME
    warning('Error during OCR: %s. No OCR result could be obtained.', ME.message);
    ocrText = 'N/A (OCR Error)';
end

% --- 5. Display Results ---
figure('Name', 'Number Plate Detection & OCR', 'NumberTitle', 'off');

% Display original image with detected bounding box
subplot(1, 2, 1);
imshow(img);
hold on;
rectangle('Position', bbox, 'EdgeColor', 'g', 'LineWidth', 2); % Draw bounding box
hold off;
title(sprintf('Original Image\n(Ground Truth: %s)', numberPlateName));

% Display cropped plate and OCR results
subplot(1, 2, 2);
imshow(cropped_plate);
hold on;
% Optionally, display character bounding boxes from OCR
if exist('results', 'var') && ~isempty(results.CharacterBoundingBoxes)
    for i = 1:size(results.CharacterBoundingBoxes, 1)
        rectangle('Position', results.CharacterBoundingBoxes(i,:), 'EdgeColor', 'r', 'LineWidth', 1);
    end
end
hold off;
title(sprintf('Cropped Number Plate\nOCR Result: %s', ocrText));

fprintf('\n--- Detection and OCR Complete ---\n');
fprintf('Ground Truth (from XML): %s\n', numberPlateName);
fprintf('OCR Result: %s\n', ocrText);

% Compare OCR result with ground truth (optional)
if strcmpi(ocrText, numberPlateName)
    fprintf('OCR result matches ground truth!\n');
else
    fprintf('OCR result does NOT exactly match ground truth.\n');
end

end % End of function