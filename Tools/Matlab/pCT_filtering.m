close all;
clear all;
clc;

output_directory = 'C:\Users\Blake\Documents\Visual Studio 2010\Projects\pCT_Reconstruction\Output\';

% output_folder = 'output_ESFPhant\';
% start_iteration = 1;
% iterations = 20;
% output_folder = 'input_CTP404\';
output_folder = 'CTP404_4M\';
start_iteration = 1;
end_iteration = 12;

FBP_image_filename = 'FBP_image_h';
FBP_image_path = [output_directory, output_folder, FBP_image_filename];
plot_FBP_image = true;
write_FBP_image = false;
FBP_IMAGE = true;

output_image_type = 'png';
image_options = {output_image_type, 'Author', 'Me', 'Copyright', '2014', 'bitdepth', 16};

slices = 32;
start_slice = 16;
end_slice = 16;

filter_radius = 5;
middle_element = floor( filter_radius^2 /2 )+1;
if(plot_FBP_image)
    text_image = read_pCT_txt_image(FBP_image_path);
    %text_image = mat2gray(text_image, [0,2]);
    %h = fspecial('disk', 5);
    %filtered = imfilter( text_image, h);
    filtered = ordfilt2( text_image, middle_element, ones(filter_radius,filter_radius));
    %display_text_images({filtered}, true, 'Filtered FBP');
    %fileID = fopen([output_directory, output_folder, 'FBP_med',num2str(filter_radius),'.bin'], 'w');
    %fwrite(fileID, filtered', 'float');
    %fclose(fileID);
    for slice = start_slice:end_slice
        image_slice = extract_image_slice( text_image, slice );
        filtered_slice = extract_image_slice( filtered, slice );
        %display_text_images({image_slice}, true, 'FBP');
        display_text_images({filtered_slice}, true, 'Filtered FBP');
        if(write_FBP_image)
            matrix_2_image(image_slice, [FBP_image_path, '_Slice_', int2str(slice)],  image_options);
        end
    end
end





