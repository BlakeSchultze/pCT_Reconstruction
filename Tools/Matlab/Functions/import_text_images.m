function [images] = import_text_images(filenames)
    num_images = size(filenames, 2);
    images = cell(1, num_images);
    for i = 1 : num_images
        image = importdata( filenames{i} );
        images(i) = {image};
    end 
end