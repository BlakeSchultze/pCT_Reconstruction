%% Phantom Viewing
close all;

input_directory = 'C:\Users\Blake\Documents\Visual Studio 2010\Projects\pCT_Reconstruction\Input\WED_Input_Data\RStP_DICOM_PHANTOM\New folder\';
input_filename_end = '_RStP_dicom_phantom_txt125mm';

output_directory = 'C:\Users\Blake\Documents\Visual Studio 2010\Projects\pCT_Reconstruction\Input\WED_Input_Data\RStP_DICOM_PHANTOM\';
output_filename_base = 'RSP_Phantom_';
input_format = '%3d';
output_format = '%03d';

num_slices = 128;
all_slices = false;
binary_writes = true;
text_plots_on = false;
binary_plots_on = false;

if( all_slices )    
    slices = 1:num_slices;
else
    slices = [1, 2, 3, 4, 5];
end

for i = 1:length(slices)
    slice = slices(i);
    Phantom_slice = importdata([input_directory, num2str(slice, input_format), input_filename_end, '.txt']);
    if( plots_on )
        set( figure, 'Name', ['Phantom: Slice = ', num2str(slice)], 'Color', 'White', 'units', 'normalized', 'outerposition', [0 0 1 1] );
        imshow(Phantom_slice)
        title( ['Phantom: Slice = ', int2str(slice)], 'Color', 'k', 'FontSize', 28, 'FontName', 'Tekton Pro' );
        xlabel( 'Column', 'Color', 'k', 'FontName', 'Tekton Pro', 'FontSize', 20 );
        ylabel( 'Row', 'Color', 'k', 'FontName', 'Tekton Pro', 'FontSize', 20 );
    end
    if( binary_writes )
        %fid = fopen( [output_directory, output_filename_base, num2str(slice - 1, output_format), '.bin'], 'w' );
        fid = fopen( [output_directory, output_filename_base, int2str(slice - 1), '.bin'], 'w' );
        fwrite(fid, Phantom_slice);
        fclose(fid);
    end
end

if( binary_plots_on )
    for i = 1:length(slices)
        slice = slices(i);
        fileID = fopen( [output_directory, output_filename_base, num2str(slice - 1, output_format), '.bin'], 'r' );
        Phantom_slice = fread( fileID, [1024, 1024] );
        fclose(fid);
        set( figure, 'Name', ['Phantom: Slice = ', num2str(slice)], 'Color', 'White', 'units', 'normalized', 'outerposition', [0 0 1 1] );
        imshow(Phantom_slice)
        title( ['Phantom: Slice = ', int2str(slice)], 'Color', 'k', 'FontSize', 28, 'FontName', 'Tekton Pro' );
        xlabel( 'Column', 'Color', 'k', 'FontName', 'Tekton Pro', 'FontSize', 20 );
        ylabel( 'Row', 'Color', 'k', 'FontName', 'Tekton Pro', 'FontSize', 20 );
    end
end