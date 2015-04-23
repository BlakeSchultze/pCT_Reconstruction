function txt_2_image(input_filename, output_filename,  varargin)
    text_image = importdata(input_filename);
    if(nargin > 2 )
        output_filename = strcat(output_filename, '.', varargin{1}(1));
        options = [];
        for i = 1:size(varargin{:}, 2)
            options = [options, varargin{1}(i)];
        end
        imwrite( text_image, output_filename{1}, options{:} );
        index = find(output_filename{1} == '\', 1, 'last');
        output = output_filename{1}(index+1:end);
    else
        output_filename = strcat(output_filename, '.png');
        imwrite( text_image, output_filename, 'png', 'bitdepth', 2 );
        index = find(output_filename == '\', 1, 'last');
        output = output_filename(index+1:end);
    end
    status = strcat(output, ' successfully written to disk')
end