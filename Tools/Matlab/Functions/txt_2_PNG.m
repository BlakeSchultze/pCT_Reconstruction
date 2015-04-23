function txt_2_PNG(input_filename, varargin)
    text_image = importdata([input_filename, '.txt']);  
    if( nargin == 1 )
        output_filename = input_filename;
    elseif( nargin == 2 )
        output_filename = varargin{1};
    end 
    imwrite( text_image, [output_filename,'.png'], 'png', 'bitdepth', 2 );
end
