%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read paths (ai) from A and overlay these on the plot of the hull image  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;

% Execution options
plot_hull = true;
plot_object = true;
overlay_paths = true;
maintain_all_paths = true;
print_intersections = true;

% Specify the names of the files to be read from disk
bin_angle = 48;
v_bin = 14;
format = '%d';

paths_filename = 'C:\Users\Blake\Documents\Visual Studio 2010\Projects\pCT_Reconstruction\Output\input_water_GeantNONUC\MLP_paths_5.txt';
hull_filename = 'C:\Users\Blake\Documents\Visual Studio 2010\Projects\pCT_Reconstruction\Output\input_water_GeantNONUC\x_hull.txt';
object_filename = 'C:\Users\Blake\Documents\Visual Studio 2010\Projects\pCT_Reconstruction\Output\input_water_GeantNONUC\MLP_image5.txt';
%paths_filename = 'C:\Users\Blake\Documents\Visual Studio 2010\Projects\pCT_Reconstruction\Output\input_water_GeantNONUC\test_path.txt';
%hull_filename = 'C:\Users\Blake\Documents\Visual Studio 2010\Projects\pCT_Reconstruction\NEO_float(200,200).txt';

% Read hull used for MLP from disk
hull = importdata(hull_filename);
%hull = imread(hull_filename);
hull_dimensions = size(hull);

% Determine/define dimensions of hull image
slices = 32;
rows = hull_dimensions(1)/slices;
columns = hull_dimensions(2);
hull_size = [rows, columns, slices];

% Plot image of the hull used for MLP
if(plot_hull)
    figure, h = imshow(hull, [0 8]), colormap('hot');
    set( gcf, 'Name', 'Hull', 'Color', 'White', 'units', 'normalized', 'outerposition', [0 0 1 1] )
    title( 'Hull Used in MLP', 'Color', 'k', 'FontSize', 28, 'FontName', 'Tekton Pro' );
    xlabel( 'Column', 'Color', 'k', 'FontName', 'Tekton Pro', 'FontSize', 20 );
    ylabel( 'Row', 'Color', 'k', 'FontName', 'Tekton Pro', 'FontSize', 20 );
end
 
% Read hull used for MLP from disk
object = importdata(object_filename);
%object = imread(object_filename);
object_dimensions = size(object);

% Determine/define dimensions of hull image
object_slices = 32;
object_rows = object_dimensions(1)/object_slices;
object_columns = object_dimensions(2);
object_size = [object_rows, object_columns, object_slices];

% Plot image of the hull used for MLP
if(plot_object)
    figure, h = imshow(object, [0 8]), colormap('hot');
    set( gcf, 'Name', 'Object', 'Color', 'White', 'units', 'normalized', 'outerposition', [0 0 1 1] )
    title( 'Object With MLP Path Identified', 'Color', 'k', 'FontSize', 28, 'FontName', 'Tekton Pro' );
    xlabel( 'Column', 'Color', 'k', 'FontName', 'Tekton Pro', 'FontSize', 20 );
    ylabel( 'Row', 'Color', 'k', 'FontName', 'Tekton Pro', 'FontSize', 20 );
 end

% Read path data (A matrix) from disk
paths = importdata(paths_filename)
paths_dimensions = size(paths);
num_histories = paths_dimensions(1);
if( maintain_all_paths)
    start_path = 1;
    end_path = num_histories; 
else
    start_path = 5;
    end_path = 10;
end

% Determine the number of intersected voxels for each path
intersections = zeros(1, num_histories);
num_intersections = zeros(1, num_histories);
for path = start_path:end_path
    linear_index = find(isnan(paths(path,:)'), 1, 'first' );
    if(isempty(linear_index))
        num_intersections(path) = paths_dimensions(2);
    else
        [column, row] = ind2sub( paths_dimensions(2), linear_index);
        num_intersections(path) = column - 1;
    end
end 

% Convert intersected voxel indices for each history into [x,y,z]
% coordinates, draw these paths on hull plot, and accumulate voxel
% coordinates for each intersection and path 
lines = cell( 1, end_path - start_path + 1 );
RGB = hull;
for path = start_path : end_path
    if( overlay_paths )
        figure, h = imshow(hull, [0 8]), hold on;
        set( gcf, 'Name', 'Hull', 'Color', 'White', 'units', 'normalized', 'outerposition', [0 0 1 1] )
        title( ['Path ', num2str(path)], 'Color', 'k', 'FontSize', 28, 'FontName', 'Tekton Pro' );
        xlabel( 'Column', 'Color', 'k', 'FontName', 'Tekton Pro', 'FontSize', 20 );
        ylabel( 'Row', 'Color', 'k', 'FontName', 'Tekton Pro', 'FontSize', 20 );
    end
    line = [];
    for intersection = 1 : num_intersections(path)
        linear_index = paths( path, intersection );
        [row, column, slice] = ind2sub( hull_size, linear_index );
        %RGB(column, row) = .56;        
        if(overlay_paths)
            plot( row, column + rows*(slice-1), 's','LineWidth', 1, 'MarkerEdgeColor',...
                'g', 'MarkerFaceColor', 'r', 'MarkerSize', 1.5 );                     
        end        
        point = [column, row, slice];
        line = [line {point}];
    end
    %figure, imshow(RGB), colormap('jet');
    lines(path) = {line};
end

% Print the coordinates of the interesected voxels for each path
if( print_intersections )
    for path = 1 : size( lines, 2 )
        Path = [ num2str(path), sprintf('\n'), 'Intersected Voxels: ', num2str( size( lines{path}, 2 ) ) ]
        for point = 1 : size( lines{path}, 2 )
            Voxel = sprintf( 'Row = %d Column = %d Slice = %d', lines{path}{point}(:) )
        end
    end
end



% test_rows = [30, 100, 150]
% test_columns = [10, 20, 30]
% test_slices = [1 1 1]

% test_index1 = test_columns(1) + (test_rows(1)-1) * rows + (test_slices(1) - 1) * columns * rows
% test_index2 = test_columns(2) + (test_rows(2)-1) * rows + (test_slices(2) - 1) * columns * rows
% test_index3 = test_columns(3) + (test_rows(3)-1) * rows + (test_slices(3) - 1) * columns * rows
% test_index = test_rows + (test_columns-1) * rows + (test_slice - 1) * columns * rows

%A = importdata()
% fid = fopen(paths_filename);
% %C = textscan(fid, '%s', 'delimiter', sprintf('\n'));
% C = fscanf(fid, '%f')
% fclose(fid);