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
output_directory = 'C:\Users\Blake\Documents\Visual Studio 2010\Projects\pCT_Reconstruction\Output\';

% output_folder = 'output_ESFPhant\';
% start_iteration = 1;
% iterations = 20;
% output_folder = 'input_CTP404\';
output_folder = 'beam-Sep2014\bin-Sensitom\';
% Define basename of text file filenames to be read (before 'xxx')
filename_base_pre = 'WEPL_dist_pre_test2_';
%filename_base_post = 'WEPL_dist_post_test2_';
format = '%03d';

% Specify which plots to generate
pre_cut_separate = true;
post_cut_separate = false;
pre_cut_overlay = false;
post_cut_overlay = false;

% Define plot bounds and appearance parameters
xmin = 90;
xmax = 270;
xtick = 20;
ymin = 0.0;
ymax = 20.0;
ytick = 5.0;


for angular_bin = 1 : length(angles)
    % Plot WEPL distribution (before cuts) for each v bin as subplots in single figure
    if( pre_cut_separate )        
        WEPLs = importdata([output_directory, output_folder, filename_base_pre,num2str(angles(angular_bin), format),'.txt'])
        dimensions = size(WEPLs);
        columns = floor(sqrt(dimensions(1)));
        if( columns ~= sqrt(dimensions(1)))
            rows = columns + 1;
        else 
            rows = columns;
        end
        set( figure, 'Name', ['Angle ', num2str(angles(angular_bin)),': Before Cuts'], 'units', 'normalized', 'outerposition', [0 0 1 1] );
        suptitle(['Angle ', num2str(angles(angular_bin)),': Before Cuts']);
        for i = 1 : dimensions(1)
            subplot(rows,columns,i)
            h = plot(WEPLs(i,:));
            set(h, 'Color','g','linewidth', 1.2);
            xlim( [xmin xmax] );
            ylim( [ymin ymax] );
            set( gca, 'XTick', xmin : xtick : xmax, 'FontSize', 8, 'XColor','r', 'FontName', 'Times New Roman',...
                'YTick', ymin : ytick : ymax, 'Box', 'on', 'YColor', 'r', 'LineWidth', 1.5 );
            title( ['$v_{bin} = $', num2str(i)], 'Interpreter', 'Latex', 'Color', 'k', 'FontSize', 20, 'FontName',...
                'Times New Roman' );
            xlabel( 't bin', 'Interpreter', 'Latex', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 12 );
            ylabel( 'Mean WEPL', 'Interpreter','Latex', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 12 );
        end
    end
    
    % Plot WEPL distribution (after cuts) for each v bin as subplots in single figure
    if( post_cut_separate )       
        WEPLs = importdata([filename_base_post,num2str(angles(angular_bin), format),'.txt']);
        set( figure, 'Name', ['Angle ', num2str(angles(angular_bin)),': After Cuts'], 'units', 'normalized', 'outerposition', [0 0 1 1] );
        suptitle(['Angle ', num2str(angles(angular_bin)),': After Cuts']);
        for i = 1 : dimensions(1)
            subplot(rows,columns,i)
            h = plot(WEPLs(i,:));
            set(h, 'Color','g','linewidth', 1.2);
            xlim( [xmin xmax] );
            ylim( [ymin ymax] );
            set( gca, 'XTick', xmin : xtick : xmax, 'FontSize', 8, 'XColor','r', 'FontName', 'Times New Roman',...
                'YTick', ymin : ytick : ymax, 'Box', 'on', 'YColor', 'r', 'LineWidth', 1.5 );
            title( ['$v_{bin} = $', num2str(i)], 'Interpreter', 'Latex', 'Color', 'k', 'FontSize', 20, 'FontName',...
                'Times New Roman' );
            xlabel( 't bin', 'Interpreter', 'Latex', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 12 );
            ylabel( 'Mean WEPL', 'Interpreter','Latex', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 12 );
        end
    end
    if( pre_cut_overlay )
        % Plot WEPL distributions (before cuts) for each v bin on top of each other
        WEPLs = importdata([filename_base_pre, num2str(angles(angular_bin), format),'.txt']);
        set( figure, 'Name', ['Angle ', num2str(angles(angular_bin)),': Before Cuts'], 'units', 'normalized', 'outerposition', [0 0 1 1] );
        suptitle(['Angle ', num2str(angles(angular_bin)),': Before Cuts']);
        for i = 1 : dimensions(1)
            h = plot(WEPLs(i,:)); hold on;
            set(h, 'Color','g','linewidth', 1.2);
            xlim( [xmin xmax] );
            ylim( [ymin ymax] );
            set( gca, 'XTick', xmin : xtick : xmax, 'FontSize', 8, 'XColor','r', 'FontName', 'Times New Roman',...
                'YTick', ymin : ytick : ymax, 'Box', 'on', 'YColor', 'r', 'LineWidth', 1.5 );
            title( ['$v_{bin} = $', num2str(i)], 'Interpreter', 'Latex', 'Color', 'k', 'FontSize', 20, 'FontName',...
                'Times New Roman' );
            xlabel( 't bin', 'Interpreter', 'Latex', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 12 );
            ylabel( 'Mean WEPL', 'Interpreter','Latex', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 12 );
        end
    end
    if( post_cut_overlay )
        % Plot WEPL distributions (after cuts) for each v bin on top of each other
        WEPLs = importdata([filename_base_post,num2str(angles(angular_bin), format),'.txt']);
        set( figure, 'Name', ['Angle ', num2str(angles(angular_bin)),': After Cuts'],'units', 'normalized', 'outerposition', [0 0 1 1] );
        suptitle(['Angle ', num2str(angles(angular_bin)),': After Cuts']);
        for i = 1 : dimensions(1)
            h = plot(WEPLs(i,:)); hold on;
            set(h, 'Color','g','linewidth', 1.2);
            xlim( [xmin xmax] );
            ylim( [ymin ymax] );
            set( gca, 'XTick', xmin : xtick : xmax, 'FontSize', 8, 'XColor','r', 'FontName', 'Times New Roman',...
                'YTick', ymin : ytick : ymax, 'Box', 'on', 'YColor', 'r', 'LineWidth', 1.5 );
            title( ['$v_{bin} = $', num2str(i)], 'Interpreter', 'Latex', 'Color', 'k', 'FontSize', 20, 'FontName',...
                'Times New Roman' );
            xlabel( 't bin', 'Interpreter', 'Latex', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 12 );
            ylabel( 'Mean WEPL', 'Interpreter','Latex', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 12 );
        end
    end
end

%% Original .m from 7/20/2014 5:03 PM
% close all;

% % Define angles to plot
% all_angles = false;
% angular_interval = 4;
% num_angles = 360 / angular_interval;
% if( all_angles == true )
    % angles = 0 : angular_interval : 360 - angular_interval;
% else
    % single_angular_bin = floor( num_angles/2);
    % angles = single_angular_bin * angular_interval;
% end

% % Define basename of text file filenames to be read (before 'xxx')
% filename_base_pre = 'WEPL_dist_pre_test2_';
% filename_base_post = 'WEPL_dist_post_test2_';
% format = '%03d';

% % Specify which plots to generate
% pre_cut_separate = true;
% post_cut_separate = true;
% pre_cut_overlay = true;
% post_cut_overlay = true;

% % Define plot bounds and appearance parameters
% xmin = 90;
% xmax = 270;
% xtick = 20;
% ymin = 0.0;
% ymax = 20.0;
% ytick = 5.0;


% for angular_bin = 1 : length(angles)
    % % Plot WEPL distribution (before cuts) for each v bin as subplots in single figure
    % if( pre_cut_separate )        
        % WEPLs = importdata([filename_base_pre,num2str(angles(angular_bin), format),'.txt'])
        % dimensions = size(WEPLs);
        % columns = floor(sqrt(dimensions(1)));
        % if( columns ~= sqrt(dimensions(1)))
            % rows = columns + 1;
        % else 
            % rows = columns;
        % end
        % set( figure, 'Name', ['Angle ', num2str(angles(angular_bin)),': Before Cuts'], 'units', 'normalized', 'outerposition', [0 0 1 1] );
        % suptitle(['Angle ', num2str(angles(angular_bin)),': Before Cuts']);
        % for i = 1 : dimensions(1)
            % subplot(rows,columns,i)
            % h = plot(WEPLs(i,:));
            % set(h, 'Color','g','linewidth', 1.2);
            % xlim( [xmin xmax] );
            % ylim( [ymin ymax] );
            % set( gca, 'XTick', xmin : xtick : xmax, 'FontSize', 8, 'XColor','r', 'FontName', 'Times New Roman',...
                % 'YTick', ymin : ytick : ymax, 'Box', 'on', 'YColor', 'r', 'LineWidth', 1.5 );
            % title( ['$v_{bin} = $', num2str(i)], 'Interpreter', 'Latex', 'Color', 'k', 'FontSize', 20, 'FontName',...
                % 'Times New Roman' );
            % xlabel( 't bin', 'Interpreter', 'Latex', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 12 );
            % ylabel( 'Mean WEPL', 'Interpreter','Latex', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 12 );
        % end
    % end
    
    % % Plot WEPL distribution (after cuts) for each v bin as subplots in single figure
    % if( post_cut_separate )       
        % WEPLs = importdata([filename_base_post,num2str(angles(angular_bin), format),'.txt']);
        % set( figure, 'Name', ['Angle ', num2str(angles(angular_bin)),': After Cuts'], 'units', 'normalized', 'outerposition', [0 0 1 1] );
        % suptitle(['Angle ', num2str(angles(angular_bin)),': After Cuts']);
        % for i = 1 : dimensions(1)
            % subplot(rows,columns,i)
            % h = plot(WEPLs(i,:));
            % set(h, 'Color','g','linewidth', 1.2);
            % xlim( [xmin xmax] );
            % ylim( [ymin ymax] );
            % set( gca, 'XTick', xmin : xtick : xmax, 'FontSize', 8, 'XColor','r', 'FontName', 'Times New Roman',...
                % 'YTick', ymin : ytick : ymax, 'Box', 'on', 'YColor', 'r', 'LineWidth', 1.5 );
            % title( ['$v_{bin} = $', num2str(i)], 'Interpreter', 'Latex', 'Color', 'k', 'FontSize', 20, 'FontName',...
                % 'Times New Roman' );
            % xlabel( 't bin', 'Interpreter', 'Latex', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 12 );
            % ylabel( 'Mean WEPL', 'Interpreter','Latex', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 12 );
        % end
    % end
    % if( pre_cut_overlay )
        % % Plot WEPL distributions (before cuts) for each v bin on top of each other
        % WEPLs = importdata([filename_base_pre, num2str(angles(angular_bin), format),'.txt']);
        % set( figure, 'Name', ['Angle ', num2str(angles(angular_bin)),': Before Cuts'], 'units', 'normalized', 'outerposition', [0 0 1 1] );
        % suptitle(['Angle ', num2str(angles(angular_bin)),': Before Cuts']);
        % for i = 1 : dimensions(1)
            % h = plot(WEPLs(i,:)); hold on;
            % set(h, 'Color','g','linewidth', 1.2);
            % xlim( [xmin xmax] );
            % ylim( [ymin ymax] );
            % set( gca, 'XTick', xmin : xtick : xmax, 'FontSize', 8, 'XColor','r', 'FontName', 'Times New Roman',...
                % 'YTick', ymin : ytick : ymax, 'Box', 'on', 'YColor', 'r', 'LineWidth', 1.5 );
            % title( ['$v_{bin} = $', num2str(i)], 'Interpreter', 'Latex', 'Color', 'k', 'FontSize', 20, 'FontName',...
                % 'Times New Roman' );
            % xlabel( 't bin', 'Interpreter', 'Latex', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 12 );
            % ylabel( 'Mean WEPL', 'Interpreter','Latex', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 12 );
        % end
    % end
    % if( post_cut_overlay )
        % % Plot WEPL distributions (after cuts) for each v bin on top of each other
        % WEPLs = importdata([filename_base_post,num2str(angles(angular_bin), format),'.txt']);
        % set( figure, 'Name', ['Angle ', num2str(angles(angular_bin)),': After Cuts'],'units', 'normalized', 'outerposition', [0 0 1 1] );
        % suptitle(['Angle ', num2str(angles(angular_bin)),': After Cuts']);
        % for i = 1 : dimensions(1)
            % h = plot(WEPLs(i,:)); hold on;
            % set(h, 'Color','g','linewidth', 1.2);
            % xlim( [xmin xmax] );
            % ylim( [ymin ymax] );
            % set( gca, 'XTick', xmin : xtick : xmax, 'FontSize', 8, 'XColor','r', 'FontName', 'Times New Roman',...
                % 'YTick', ymin : ytick : ymax, 'Box', 'on', 'YColor', 'r', 'LineWidth', 1.5 );
            % title( ['$v_{bin} = $', num2str(i)], 'Interpreter', 'Latex', 'Color', 'k', 'FontSize', 20, 'FontName',...
                % 'Times New Roman' );
            % xlabel( 't bin', 'Interpreter', 'Latex', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 12 );
            % ylabel( 'Mean WEPL', 'Interpreter','Latex', 'Color', 'b', 'FontName', 'Times New Roman', 'FontSize', 12 );
        % end
    % end
% end