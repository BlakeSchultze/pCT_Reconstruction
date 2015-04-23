%% Example image reading/viewing/writing
close all;

% specify directory and filename and combine them into exact file path
directory = 'C:\Users\Blake\Documents\Visual Studio 2010\Projects\pCT_Reconstruction\Output\output_ESFPhant\';
MSC_filename = 'MSC_counts_h.txt';
MSC_filepath = [directory MSC_filename];
FBP_filename = 'FBP_image_h.txt';
FBP_filepath = [directory FBP_filename];

% reading in multiple images and storing them in single cell array
image_filepaths = {MSC_filepath, FBP_filepath};
images = import_text_images(image_filepaths);

% Displaying images from cell array, with optional full screen option
full_screen = true;
display_text_images(images, full_screen)

% Alternative method of importing single text image from file, determine
% its dimensions, and convert it to gray scale image
Hull = importdata(MSC_filepath);
[xmax, ymax] = size(Hull);
Hull = mat2gray(Hull);

% Extract single slice of an image and display image using by passing
% single image into display function as cell array by enclosing in {}
slice = 32;
Hull_slice = extract_image_slice(Hull, slice);
%figure, imshow(Hull_slice)
display_text_images({Hull_slice}, full_screen)

% Write image to disk, with optional 3rd parameter for specifying the image
% type and properties; these particular options are those used by default
% if no 3rd argument is present.  Usage of optional argument shown here
% simply for demonstration purposed, see Matlab documentation for supported
% image file types and options 
options = {'png', 'bitdepth', 8}
output_filename2 = 'MSC_counts2';
matrix_2_image(images{1}, output_filename2, options)

% Read image from text file and write directly to disk without displaying 
% it.  Same image options passed explicitly again through optional 3rd
% argument although these are the default options used when no 3rd argument
% is present
output_filename3 = 'FBP_image';
txt_2_image(FBP_filepath, output_filename3,  options)