function image_2_PNG(images, filenames, manual_naming, bitdepth)
    num_images = size(images, 2);
    for i = 1 : num_images
        if( manual_naming )
            imwrite(images{i}, [filenames{i},'.png'], 'png', 'bitdepth', bitdepth )
        else
            imwrite(images{i}, [filenames{i}(1:end-4),'.png'], 'png', 'bitdepth', bitdepth )
        end
    end 
end
