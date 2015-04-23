function [column, row, slice] = voxel_2_indices(voxel, columns, rows)  
    column = 0;
    row = 0;
    slice = 0;
    
    while( voxel - columns * rows > 0 )
		voxel = voxel - columns * rows;
		slice = slice + 1;
    end
	% => bin = t_bin + angular_bin * T_BINS > 0
	while( voxel - columns > 0 )
		voxel = voxel - columns;
		row = row + 1;
    end
	% => bin = t_bin > 0
	column = voxel;
end


