close all;
clear all;
clc;

output_directory = 'C:\Users\Blake\Documents\Visual Studio 2010\Projects\pCT_Reconstruction\Output\';


% output_folder = 'output_ESFPhant\';
% start_iteration = 1;
% iterations = 20;
output_folder = {'input_CTP404\tests\x_after_add_collect_3\', 'input_CTP404\tests\x_add_order\'}
start_iteration = 1;
end_iteration = 3;
% output_folder = 'input_CTP404\New folder\\';
% start_iteration = 1;
% iterations = 2;
% output_folder = 'input_CTP404\New folder (2)\\';
% start_iteration = 1;
% iterations = 3;
% output_folder = 'input_CTP404\New folder (3)\\';
% start_iteration = 1;
% iterations = 4;
% output_folder = 'input_CTP404\New folder (4)\\';
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
iterates_path = {[output_directory, output_folder{1}, iterates_basename],[output_directory, output_folder{2}, iterates_basename]};
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


images = {};
start_slice = 17;
end_slice = 17;
compare_iteration = 3;
% if(plot_iterates)
% 	for iteration = start_iteration:end_iteration
%         
%         text_image1 = read_pCT_txt_image([iterates_path{1}, int2str(iteration)]);
%         text_image2 = read_pCT_txt_image([iterates_path{2}, int2str(iteration)]);
%         image_slice = extract_image_slice( text_image2, 17 );
%         first_column = 24;
%         text_slice = image_slice(101,first_column:101);
%         figure, plot( first_column:101, text_slice);
%         %text_image1 = mat2gray(text_image1, [0,2]);
%         %text_image2 = mat2gray(text_image2, [0,2]);
%         text_diff = text_image2 - text_image1;
%         max_diff = max(max(text_diff))
%         diff = sum(sum(text_diff.^2))
% %         for slice = start_slice:end_slice
% %             image_slice = extract_image_slice( text_image, slice );
% %             display_text_images({image_slice}, true, ['x_k : Iteration ', num2str(iteration), ' Slice ', num2str(slice)]);
% %             if(write_iterates)
% %                 matrix_2_image(image_slice, [iterates_path, 'Slice_', int2str(slice), '_Iteration_', num2str(iteration)],  image_options);
% %             end
% %         end
% 	end
% end
for iteration = 1:1
    %for iteration = start_iteration:end_iteration
    text_image2 = read_pCT_txt_image([iterates_path{2}, int2str(iteration)]);
    image_slice = extract_image_slice( text_image2, 17 );
    first_column = 26;
    text_slice = image_slice(101,first_column:101);
    %figure, plot( first_column:101, text_slice), hold on;
    rs = first_column:101;
    ds = dataset(rs',text_slice')
    modelfun = @(b,x)b(1) + b(2)*(x(:,1)).^(-2)
    %modelfun = @(b,x)b(1) + b(2)*x(:,1).^b(3) + b(4)*x(:,2).^b(5);
    beta0 = [1 100];
    mdl = NonLinearModel.fit(ds,modelfun,beta0)
    rs_new = rs;
    ypred = predict(mdl, rs_new')
    figure, plot(rs, text_slice, 'g', rs, ypred, 'r', 'LineWidth',2 ), hold on;
    
    modelfun = @(b,x)b(1) + b(2)*(x(:,1)).^(-1)
    %modelfun = @(b,x)b(1) + b(2)*x(:,1).^b(3) + b(4)*x(:,2).^b(5);
    beta0 = [1 100];
    mdl = NonLinearModel.fit(ds,modelfun,beta0)
    rs_new = rs;
    ypred = predict(mdl, rs_new')
    plot(rs, ypred, 'b', 'LineWidth',2 ), hold on;
    
    modelfun = @(b,x)b(1) + b(2)*(exp(-b(3)*x(:,1)))
    %modelfun = @(b,x)b(1) + b(2)*x(:,1).^b(3) + b(4)*x(:,2).^b(5);
    beta0 = [1 100 0.01];
    mdl = NonLinearModel.fit(ds,modelfun,beta0)
    rs_new = rs;
    ypred = predict(mdl, rs_new')
    plot(rs, ypred, 'c', 'LineWidth',2  ), hold on;
    
    modelfun = @(b,x)b(1) + b(2)*(exp(-b(3)*x(:,1).^2))
    %modelfun = @(b,x)b(1) + b(2)*x(:,1).^b(3) + b(4)*x(:,2).^b(5);
    beta0 = [.9 4 0.01];
    mdl = NonLinearModel.fit(ds,modelfun,beta0)
    rs_new = rs;
    ypred = predict(mdl, rs_new')
    plot(rs, ypred, 'k-o', 'LineWidth',2  ), hold on;
    
%     modelfun = @(b,x)b(1) + b(2)*(x(:,1)).^(-1) + b(3)*(exp(-b(4)*x(:,1)))
%     %modelfun = @(b,x)b(1) + b(2)*x(:,1).^b(3) + b(4)*x(:,2).^b(5);
%     beta0 = [10 100 3 0.1];
%     mdl = NonLinearModel.fit(ds,modelfun,beta0)
%     rs_new = rs;
%     ypred = predict(mdl, rs_new')
%     plot(rs, ypred, 'k', 'LineWidth',2  ), hold off;
end

% load carbig
% ds = dataset(Horsepower,Weight,MPG);
% modelfun = @(b,x)b(1) + b(2)*x(:,1).^b(3) + ...
%     b(4)*x(:,2).^b(5);
% beta0 = [-50 500 -1 500 -1];
% mdl = NonLinearModel.fit(ds,modelfun,beta0)
