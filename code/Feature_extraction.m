function extracted_feature = Feature_extraction(xmlFilePath,EnhancedImage)

    tree = xmlread(xmlFilePath); % Read the XML file as a DOM node

    % Get the 'object' node. Assumes one object element for the number plate.
    objectNode = tree.getElementsByTagName('object').item(0); % Get the 'object' element
  

    % Get the 'bndbox' element
    bndboxNode = objectNode.getElementsByTagName('bndbox').item(0); % Get the 'bndbox' element
 

    % Helper function to safely extract numeric data from XML
    extractNumericData = @(parentNode, tagName) str2double(char(parentNode.getElementsByTagName(tagName).item(0).getFirstChild.getData));

    xmin = extractNumericData(bndboxNode, 'xmin');
    ymin = extractNumericData(bndboxNode, 'ymin');
    xmax = extractNumericData(bndboxNode, 'xmax');
    ymax = extractNumericData(bndboxNode, 'ymax');

    fprintf('Successfully extracted bounding box from XML: [xmin:%d, ymin:%d, xmax:%d, ymax:%d]\n', xmin, ymin, xmax, ymax);




    img = EnhancedImage; % Read the image


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

extracted_feature = imcrop(img, bbox); % Crop the image using the extracted bounding box

end