close all;

% Define angles to plot
all_angles = false;
angular_interval = 4;
num_angles = 360 / angular_interval;
if( all_angles == true )
    angles = 0 : angular_interval : 360 - angular_interval;
else
    single_angular_bin = floor( num_angles/2);
    angles = single_angular_bin * angular_interval;
end

% Define text file filenames to be read
filename_base_pre = 'post_cut_WEPLs_angle=186_v=16';
%filename_base_post = 'WEPL_dist_post_test2_';
format = '%03d';

% Define plot bounds and appearance parameters
xmin = 90;
xmax = 270;
xtick = 20;
ymin = 0.0;
ymax = 20.0;
ytick = 5.0;

WEPLs = importdata([filename_base_pre,'.txt']);
dimensions = size(WEPLs);
num_bins_to_search = 10;
num_members = zeros(num_bins_to_search,1);
%a = WEPLs(1:4,:)
for i = 1:num_bins_to_search
    linear_index = find(isnan(WEPLs(i,:)'), 1, 'first' )
    if(isempty(linear_index))
        num_members(i) = dimensions(2);
    else
        [column, row] = ind2sub( dimensions(2), linear_index);
        num_members(i) = column-1;
    end
    %[column, row] = ind2sub(size(a'), linear_index)  
end
WEPLs(1:num_bins_to_search,:)
num_members

for i = 1:num_bins_to_search
    WEPLs(i, num_members(i))
    if( num_members(i) < dimensions(2) )
        WEPLs(i, num_members(i)+1)
    else
        'good'
    end
end
%isnan
% for angular_bin = 1 : length(angles)
%     % Plot WEPL distribution (before cuts) for each v bin as subplots in single figure
%     WEPLs = importdata([filename_base_pre,num2str(angles(angular_bin), format),'.txt'])
%     dimensions = size(WEPLs);
%     columns = floor(sqrt(dimensions(1)));
%     if( columns ~= sqrt(dimensions(1)))
%         rows = columns + 1;
%     else 
%         rows = columns;
%     end
%     set( figure, 'Name', ['Angle ', num2str(angles(angular_bin)),': Before Cuts'], 'units', 'normalized', 'outerposition', [0 0 1 1] );
%     suptitle(['Angle ', num2str(angles(angular_bin)),': Before Cuts']);
%     for i = 1 : dimensions(1)
%         subplot(rows,columns,i)
%         h = plot(WEPLs(i,:));
%         set(h, 'Color','g','linewidth', 1.2);
%         xlim( [xmin xmax] );
%         ylim( [ymin ymax] );
%         set( gca, 'XTick', xmin : xtick : xmax, 'FontSize', 8, 'XColor','r', 'FontName', 'Times New Roman',...
%             'YTick', ymin : ytick : ymax, 'Box', 'on', 'YColor', 'r', 'LineWidth', 1.5 );
%         title( ['$v_{bin} = $', num2str(i)], 'Interpreter', 'Latex', 'Color', 'k', 'FontSize', 20, 'FontName',...
%             'Times New Roman' );
%         xlabel( 't bin', 'Interpreter', 'Latex', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 12 );
%         ylabel( 'Mean WEPL', 'Interpreter','Latex', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 12 );
%     end
%     
%     % Plot WEPL distribution (after cuts) for each v bin as subplots in single figure
%     WEPLs = importdata([filename_base_post,num2str(angles(angular_bin), format),'.txt']);
%     set( figure, 'Name', ['Angle ', num2str(angles(angular_bin)),': After Cuts'], 'units', 'normalized', 'outerposition', [0 0 1 1] );
%     suptitle(['Angle ', num2str(angles(angular_bin)),': After Cuts']);
%     for i = 1 : dimensions(1)
%         subplot(rows,columns,i)
%         h = plot(WEPLs(i,:));
%         set(h, 'Color','g','linewidth', 1.2);
%         xlim( [xmin xmax] );
%         ylim( [ymin ymax] );
%         set( gca, 'XTick', xmin : xtick : xmax, 'FontSize', 8, 'XColor','r', 'FontName', 'Times New Roman',...
%             'YTick', ymin : ytick : ymax, 'Box', 'on', 'YColor', 'r', 'LineWidth', 1.5 );
%         title( ['$v_{bin} = $', num2str(i)], 'Interpreter', 'Latex', 'Color', 'k', 'FontSize', 20, 'FontName',...
%             'Times New Roman' );
%         xlabel( 't bin', 'Interpreter', 'Latex', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 12 );
%         ylabel( 'Mean WEPL', 'Interpreter','Latex', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 12 );
%     end
%     
%     % Plot WEPL distributions (before cuts) for each v bin on top of each other
%     WEPLs = importdata([filename_base_pre, num2str(angles(angular_bin), format),'.txt']);
%     set( figure, 'Name', ['Angle ', num2str(angles(angular_bin)),': Before Cuts'], 'units', 'normalized', 'outerposition', [0 0 1 1] );
%     suptitle(['Angle ', num2str(angles(angular_bin)),': Before Cuts']);
%     for i = 1 : dimensions(1)
%         h = plot(WEPLs(i,:)); hold on;
%         set(h, 'Color','g','linewidth', 1.2);
%         xlim( [xmin xmax] );
%         ylim( [ymin ymax] );
%         set( gca, 'XTick', xmin : xtick : xmax, 'FontSize', 8, 'XColor','r', 'FontName', 'Times New Roman',...
%             'YTick', ymin : ytick : ymax, 'Box', 'on', 'YColor', 'r', 'LineWidth', 1.5 );
%         title( ['$v_{bin} = $', num2str(i)], 'Interpreter', 'Latex', 'Color', 'k', 'FontSize', 20, 'FontName',...
%             'Times New Roman' );
%         xlabel( 't bin', 'Interpreter', 'Latex', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 12 );
%         ylabel( 'Mean WEPL', 'Interpreter','Latex', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 12 );
%     end
%     
%     % Plot WEPL distributions (after cuts) for each v bin on top of each other
%     WEPLs = importdata([filename_base_post,num2str(angles(angular_bin), format),'.txt']);
%     set( figure, 'Name', ['Angle ', num2str(angles(angular_bin)),': After Cuts'],'units', 'normalized', 'outerposition', [0 0 1 1] );
%     suptitle(['Angle ', num2str(angles(angular_bin)),': After Cuts']);
%     for i = 1 : dimensions(1)
%         h = plot(WEPLs(i,:)); hold on;
%         set(h, 'Color','g','linewidth', 1.2);
%         xlim( [xmin xmax] );
%         ylim( [ymin ymax] );
%         set( gca, 'XTick', xmin : xtick : xmax, 'FontSize', 8, 'XColor','r', 'FontName', 'Times New Roman',...
%             'YTick', ymin : ytick : ymax, 'Box', 'on', 'YColor', 'r', 'LineWidth', 1.5 );
%         title( ['$v_{bin} = $', num2str(i)], 'Interpreter', 'Latex', 'Color', 'k', 'FontSize', 20, 'FontName',...
%             'Times New Roman' );
%         xlabel( 't bin', 'Interpreter', 'Latex', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 12 );
%         ylabel( 'Mean WEPL', 'Interpreter','Latex', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 12 );
%      end
% end