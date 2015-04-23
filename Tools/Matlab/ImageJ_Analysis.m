close all;
clear all;
clc;

directory = 'C:\Users\Blake\Documents\pCT_Data\object_name\Experimental\MMDDYYYY\run_number\Output\MMDDYYYY\Reconstruction\MMDDYYYY\';

% Line profile analysis along diameter of inserts
profile_plot = false;
filename_profile = 'Line_Profiles_FBP.csv';
indices = 0:12;
row_offset = 1;
column_offset = 1;
profiles = csvread(filename_profile,row_offset,column_offset);

if(profile_plot)
    fig = figure;
    set( fig, 'units', 'normalized', 'outerposition', [0 0 1 1] );
    plot(indices, profiles, 'LineWidth', 2.5), grid on;
    title('Line Profiles', 'interpreter', 'latex');
    xlabel('$x$', 'interpreter', 'latex');
    ylabel('Observed RSP');
    legend('Air (bottom)', 'Air (top)', 'PMP', 'LDPE', 'Polystyrene', 'Acrylic', 'Delrin', 'Teflon');
end
print(fig, '-dpng', 'Profile_plot_FBP');

% Mean/standard deviation analysis of each insert
filename_regions = 'Region_Statistics_FBP.csv';
region_plot = false;
%true_RSPs = [0.0014, 1.838, 1.354, 1.155, 0.0014, 1.0386, 0.9973, 0.869];
true_RSPs = [0.0014, 0.0014, 0.869, 0.9973, 1.0386, 1.155, 1.354, 1.838];
true_RSPs_sorted = sort(true_RSPs);

row_offset = 1;
column_offset = 0;
column_headers = ['#', 'Area', 'Mean', 'Std Dev', 'Min', 'Max'];
region_stats = csvread(filename_regions,row_offset,column_offset);
means = region_stats(:,3);
means_sorted = sort(means);
if(region_plot)
    fig = figure;
    set( fig, 'units', 'normalized', 'outerposition', [0 0 1 1] );
        axes('GridLineStyle', ':');
    plot(true_RSPs_sorted, means_sorted, 'g', 'LineWidth', 2.5), hold on;
    plot(true_RSPs_sorted, true_RSPs_sorted, 'k--', 'LineWidth', 1.5), grid on;
    title('Observed vs. Predicted   Mean (green) with $y = x$ reference (black)', 'interpreter', 'latex');
    xlabel('Predicted Mean RSP');
    ylabel('Observed Mean RSP');
end
print(fig, '-dpng', 'RSP_plot_FBP');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 0 : 12
% Line profile analysis along diameter of inserts

filename_profile = ['Line_Profiles_x_',num2str(i),'.csv'];
row_offset = 1;
column_offset = 1;
profiles = csvread(filename_profile,row_offset,column_offset);

FX = gradient(profiles);
    fig = figure;
    set( fig, 'units', 'normalized', 'outerposition', [0 0 1 1] );
    plot(indices, FX, 'LineWidth', 2.5), grid on;
    
if(profile_plot)
    fig = figure;
    set( fig, 'units', 'normalized', 'outerposition', [0 0 1 1] );
    plot(indices, profiles, 'LineWidth', 2.5), grid on;
    title('Line Profiles', 'interpreter', 'latex');
    xlabel('$x$', 'interpreter', 'latex');
    ylabel('Observed RSP');
    legend('Air', 'Air', 'PMP', 'LDPE', 'Polystyrene', 'Acrylic', 'Delrin', 'Teflon');
    print(fig, '-dpng', ['Profile_plot_x_',num2str(i)]);
end


% Mean/standard deviation analysis of each insert
filename_regions = ['Region_Statistics_x_',num2str(i),'.csv'];
region_plot = true;
true_RSPs = [0.0014, 1.838, 1.354, 1.155, 0.0014, 1.0386, 0.9973, 0.869];
true_RSPs_sorted = sort(true_RSPs);

row_offset = 1;
column_offset = 0;
column_headers = ['#', 'Area', 'Mean', 'Std Dev', 'Min', 'Max'];
region_stats = csvread(filename_regions,row_offset,column_offset);
means = region_stats(:,3);
means_sorted = sort(means);
if(region_plot)
    fig = figure;
    set( fig, 'units', 'normalized', 'outerposition', [0 0 1 1] );
        axes('GridLineStyle', ':');
    plot(true_RSPs_sorted, means_sorted, 'g', 'LineWidth', 2.5), hold on;
    plot(true_RSPs_sorted, true_RSPs_sorted, 'k--', 'LineWidth', 1.5), grid on;
    title('Observed vs. Predicted   Mean (green) with $y = x$ reference (black)', 'interpreter', 'latex');
    xlabel('Predicted Mean RSP');
    ylabel('Observed Mean RSP');
    print(fig, '-dpng', ['RSP_plot_x_',num2str(i)]);
end

end