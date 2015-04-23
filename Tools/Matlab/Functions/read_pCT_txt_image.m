function [text_image] = read_pCT_txt_image(input_filename)
    text_image = importdata([input_filename, '.txt']);
    %text_image
    %max(max(text_image))
    %images = cell(1, num_images);
%     for i = 1 : num_images
%         figure, imshow(images{i});
%     end 
end