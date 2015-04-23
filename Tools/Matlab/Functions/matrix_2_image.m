function matrix_2_image(matrix, output_filename,  varargin)
    if(nargin > 2 )
        output_filename = strcat(output_filename, '.', varargin{1}(1));
        options = [];
        for i = 1:size(varargin{:}, 2)
            options = [options, varargin{1}(i)];
        end
        imwrite( matrix, output_filename{1}, options{:} );
        index = find(output_filename{1} == '\', 1, 'last');
        output = output_filename{1}(index+1:end);
    else
        output_filename = strcat(output_filename, '.png');
        imwrite( matrix, output_filename, 'png', 'bitdepth', 8 );
        index = find(output_filename == '\', 1, 'last');
        output = output_filename(index+1:end);
    end
    status = strcat(output, ' successfully written to disk')
end