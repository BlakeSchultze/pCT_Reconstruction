close all;
clear all;
clc;

output_directory = 'C:\Users\Blake\Documents\Visual Studio 2010\Projects\pCT_Reconstruction\Output\';


% output_folder = 'output_ESFPhant\';
% start_iteration = 1;
% iterations = 20;
%output_folder = 'input_CTP404\tests\ART\avgfilt-r1\90\DROP\0.0004\';
output_folder = 'my_merged\';
%output_folder = 'input_CTP404\tests\hybrid-0.0002, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9\'
%output_folder = 'input_CTP404\tests\0.0005\';
%%output_folder = 'beam-Sep2014\bin-Sensitom\';
start_iteration = 12;
end_iteration = 12;
% output_folder = 'input_CTP404\New folder\\';
% start_iteration = 1;
% iterations = 2;
% output_folder = 'input_CTP404\New folder (2)\\';
% start_iteration = 1;
% iterations = 3;
% output_folder = 'input_CTP404\New folder (3)\\';
% start_iteration = 1;
% iterations = 4;
% output_folder = 'input_CTP404\Nclosew folder (4)\\';
% start_iteration = 1;
% iterations = 1;
% output_folder = 'input_CTP404\New folder (5)\\';
% start_iteration = 1;
% iterations = 17;
% output_folder = 'input_CTP404\0001\';
% start_iteration = 1;
% iterations = 12;
% output_folder = 'input_CTP404\0001var\';
% start_iteration = 1;
% iterations = 20;
%output_folder = 'input_water_GeantNONUC\';
%start_iteration = 1;
% output_folder = 'input_CTP404\0001var2_fixedMLPpolys\';
% start_iteration = 1;
% iterations = 12;
% output_folder = 'input_CTP404\00005var\';
% start_iteration = 1;
% iterations = 20;
% output_folder = 'input_CTP404\000025var\';
% start_iteration = 1;
% iterations = 8;
% output_folder = 'input_CTP404\0001midvar\';
% start_iteration = 1;
% iterations = 20;
% output_folder = 'input_CTP404\.0001exp_r^2_0.5\';
% start_iteration = 1;
% iterations = 20;
% output_folder = 'input_CTP404\.0002exp_r^2_-0.5_R^2\';
% start_iteration = 8;
% iterations = 20;
% output_folder = 'input_CTP404\.0001_1-r^2_0.99_R^2\';
% start_iteration = 1;
% iterations = 20;
% output_folder = 'input_CTP404\.0001_1-.99r^2_R^2_r^2_gt_6^2\';
% start_iteration = 1;
% iterations = 20;

FBP_image_filename = 'FBP_image_h';
SC_image_filename = 'x_SC';
MSC_image_filename = 'x_MSC';
SM_image_filename = 'x_SM';
initial_iterate_filename = 'x_k0';
iterates_basename = 'x_';
final_image_filename = 'x';

FBP_image_path = [output_directory, output_folder, FBP_image_filename];
SC_image_path = [output_directory, output_folder, SC_image_filename];
MSC_image_path = [output_directory, output_folder, MSC_image_filename];
SM_image_path = [output_directory, output_folder, SM_image_filename];
initial_iterate_path = [output_directory, output_folder, initial_iterate_filename];
iterates_path = [output_directory, output_folder, iterates_basename];
final_image_path = [output_directory, output_folder, final_image_filename];

plot_FBP_image = false;
plot_SC_image = false;
plot_MSC_image = false;
plot_SM_image = false;
plot_initial_iterate = false;
plot_iterates = true;
plot_final_image = false;

write_FBP_image = false;
write_SC_image = false;
write_MSC_image = false;
write_SM_image = false;
write_initial_iterate = false;
write_iterates = true;
write_final_image = false;

output_image_type = 'png';
image_options = {output_image_type, 'Author', 'Me', 'Copyright', '2014', 'bitdepth', 16};

slices = 32;

FBP_IMAGE = true;
FBP = false;
SC = false;
MSC = false;
SM = false;
HULL = false;
X0 = true;
XK = true;
X = true;

av_files = dir(fullfile([output_directory, output_folder, iterates_basename,'*.txt']));
for file = 1:length(av_files)
    
end
start_slice = 15;
end_slice = 18;
if(plot_FBP_image)
    text_image = read_pCT_txt_image(FBP_image_path);
    text_image = mat2gray(text_image, [0,2]);
    for slice = start_slice:end_slice
        image_slice = extract_image_slice( text_image, slice );
        display_text_images({image_slice}, true, 'FBP');
        if(write_FBP_image)
            matrix_2_image(image_slice, [FBP_image_path, '_Slice_', int2str(slice)],  image_options);
        end
    end
end

start_slice = 17;
end_slice = 17;
if(plot_initial_iterate)
	text_image = read_pCT_txt_image(initial_iterate_path);
	text_image = mat2gray(text_image, [0,2]);
	for slice = start_slice:end_slice
        image_slice = extract_image_slice( text_image, slice );
        display_text_images({image_slice}, true);
        if(write_initial_iterate)
            matrix_2_image(image_slice, [initial_iterate_path, 'Slice_', int2str(slice)],  image_options);
        end
	end
end


start_slice = 15;
end_slice = 17;
if(plot_iterates)
	for iteration = start_iteration:end_iteration
        text_image = read_pCT_txt_image([iterates_path, int2str(iteration)]);
        text_image = mat2gray(text_image, [0,3.4]);
        for slice = start_slice:end_slice
            image_slice = extract_image_slice( text_image, slice );
            display_text_images({image_slice}, true, ['x_k : Iteration ', num2str(iteration), ' Slice ', num2str(slice)]);
            if(write_iterates)
                matrix_2_image(image_slice, [iterates_path, 'Slice_', int2str(slice), '_Iteration_', num2str(iteration)],  image_options);
            end
        end
	end
end

start_slice = 17;
end_slice = 17;
if(plot_final_image)
    text_image = read_pCT_txt_image(final_image_path);
    %text_image = mat2gray(text_image, [0,2]);
    for slice = start_slice:end_slice
        image_slice = extract_image_slice( text_image, slice )
        display_text_images({image_slice}, true);
        if(write_final_image)
            matrix_2_image(image_slice, [final_image_path, 'Slice_', int2str(slice)],  image_options)
        end
    end
end