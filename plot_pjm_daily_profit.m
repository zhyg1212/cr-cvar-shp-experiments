clear; clc;

resultFile = fullfile(pwd, 'results', 'pjm_realdata_results.csv');
figDir = fullfile(pwd, 'figures');
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

T = readtable(resultFile);
T.testDate = datetime(T.testDate, 'InputFormat', 'yyyy-MM-dd');

models = {'Nominal', 'CR-SHP', 'CVaR-CR-SHP', 'Mixed'};
lineStyles = {'-', '-', '-', '--'};
markers = {'o', 's', '^', 'd'};

figure('Color', 'w', 'Position', [100, 100, 980, 460]);
hold on;

for i = 1:numel(models)
    idx = strcmp(T.model, models{i});
    Ti = sortrows(T(idx,:), 'testDate');
    plot(Ti.testDate, Ti.profit, ...
        'LineWidth', 1.6, ...
        'LineStyle', lineStyles{i}, ...
        'Marker', markers{i}, ...
        'MarkerSize', 4);
end

grid on;
box on;
xlabel('Test day');
ylabel('Realized profit');
legend(models, 'Location', 'northwest');
title('PJM rolling daily realized profit');
set(gca, 'FontSize', 11);
xtickformat('MM-dd');

outPng = fullfile(figDir, 'pjm_daily_profit.png');
outPdf = fullfile(figDir, 'pjm_daily_profit.pdf');
exportgraphics(gcf, outPng, 'Resolution', 300);
exportgraphics(gcf, outPdf, 'ContentType', 'vector');

fprintf('Saved figure:\n  %s\n  %s\n', outPng, outPdf);
