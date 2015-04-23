function display_text_images(images, full_screen, varargin)
    num_images = size(images, 2);
    if( full_screen )
        for i = 1 : num_images
            %set( figure, 'Name', ['Image ', num2str(i)], 'units', 'normalized', 'outerposition', [0 0 1 1] );
            figure, imshow(images{i});
            figure_title = ['Image ', num2str(i)];
         	if( nargin > 2 )
                figure_title = [varargin{1}, ' ', figure_title];
            end
            set( gcf, 'Name', figure_title, 'units', 'normalized', 'outerposition', [0 0 1 1] );
        end 
    else
        for i = 1 : num_images
            figure, imshow(images{i}, [0 2]);
            figure_title = ['Image ', num2str(i)];
         	if( nargin > 2 )
                figure_title = [varargin{1}, ' ', figure_title];
            end
            set( gcf, 'Name', figure_title);            
        end 
    end
end