close all;


%input_directory = 'C:\Users\Blake\Documents\Visual Studio 2010\Projects\pCT_Reconstruction\Output\input_CTP404\New folder (3)\';
input_directory = 'C:\Users\Blake\Documents\Visual Studio 2010\Projects\pCT_Reconstruction\Output\input_water_GeantNONUC\';
filename_base = 'MLP_image';
start_history = 11919120;
num_histories = 10;

slices = 32;

num_found_slices = 0;
%for slice = 1:slices
 text_image = read_pCT_txt_image([input_directory, filename_base, int2str(start_history)]);
 columns = size(text_image, 2)
 intersections = find( text_image == 4 | text_image == 3);
 [I,J] = ind2sub(size(text_image),intersections)
 intersections = floor(I / 200)+1
 found_slices = [intersections(1)];
 num_intersections = size(intersections, 1 );
 for i = 1:num_intersections
     %slice_temp = floor( intersections(i) / ( columns*columns) )
     %[column, row, slice] = voxel_2_indices(intersections(i), columns, rows);
     %intersections(i)
     slice_found = intersections(i);
     new_slice = find( found_slices == slice_found );
     if( isempty(new_slice))
        found_slices = [found_slices, slice_found];
     end
     %num_found_slices = num_found_slices + 1;
 end
 found_slices
 for i = 1:length(found_slices)
	slice = found_slices(i);
	image_slice = extract_image_slice( text_image, slice );
	figure, h = imshow(image_slice, [0 8]), colormap('hot');
	set( gcf, 'Name', 'Hull', 'Color', 'White', 'units', 'normalized', 'outerposition', [0 0 1 1] )
	title( ['MLP for history ', num2str(start_history), ' Slice # ', num2str(slice)], 'Color', 'k', 'FontSize', 28, 'FontName', 'Tekton Pro' );
	xlabel( 'Column', 'Color', 'k', 'FontName', 'Tekton Pro', 'FontSize', 20 );
	ylabel( 'Row', 'Color', 'k', 'FontName', 'Tekton Pro', 'FontSize', 20 );
 end
 
%end

