clear; clc;

% Scalability experiment for the chain-supported reformulation.
% This experiment reports support sizes and runtimes for constructing P*,
% constructing the lower-tail distribution, and running Algorithm 2.

outDir = fullfile(pwd, 'results');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

seeds = 1:5;
nList = [50, 100, 200, 500, 1000, 2000];
gamma = 0.5;
alpha = 0.10;

rows = struct([]);
rowId = 0;

for n = nList
    for seed = seeds
        params = struct();
        params.seed = seed;
        params.n = n;
        params.k = max(1, floor(gamma * n));
        params.aMode = 'beta_mu';
        params.beta = 0.95;
        params.epsilonLow = -8;
        params.epsilonHigh = 8;
        params.muLow = 50;
        params.muHigh = 100;
        params.lowerGapLow = 20;
        params.lowerGapHigh = 70;
        params.upperGapLow = 20;
        params.upperGapHigh = 70;
        params.dLowerRatioLow = 0.45;
        params.dLowerRatioHigh = 0.85;
        params.deltaRatioLow = 0.85;
        params.deltaRatioHigh = 0.98;
        params.nominalMADMode = 'middle';
        params.generateNominalScenarios = false;
        params.generateTestScenarios = false;

        tData = tic;
        data = generate_synthetic_shp_data(params);
        dataTime = toc(tData);

        tBuild = tic;
        crDist = build_chain_worst_distribution(data.supports, data.pWorst);
        buildTime = toc(tBuild);

        tTail = tic;
        tailDist = tail_truncate_distribution(crDist, alpha);
        tailTime = toc(tTail);

        tCR = tic;
        solCR = solve_shp_approximation(data.a, data.k, crDist); %#ok<NASGU>
        crSolveTime = toc(tCR);

        tCVaR = tic;
        solCVaR = solve_shp_approximation(data.a, data.k, tailDist); %#ok<NASGU>
        cvarSolveTime = toc(tCVaR);

        rowId = rowId + 1;
        rows(rowId).n = n; %#ok<SAGROW>
        rows(rowId).k = data.k;
        rows(rowId).seed = seed;
        rows(rowId).fullSupportLog10 = n * log10(3);
        rows(rowId).chainSupport = crDist.supportSize;
        rows(rowId).tailSupport = tailDist.supportSize;
        rows(rowId).dataTime = dataTime;
        rows(rowId).buildTime = buildTime;
        rows(rowId).tailTime = tailTime;
        rows(rowId).crSolveTime = crSolveTime;
        rows(rowId).cvarSolveTime = cvarSolveTime;

        fprintf('n=%d seed=%d chain=%d tail=%d build=%.4fs CR=%.4fs CVaR=%.4fs\n', ...
            n, seed, crDist.supportSize, tailDist.supportSize, ...
            buildTime, crSolveTime, cvarSolveTime);
    end
end

T = struct2table(rows);
save(fullfile(outDir, 'scalability_results.mat'), 'T');
writetable(T, fullfile(outDir, 'scalability_results.csv'));

summary = groupsummary(T, 'n', 'mean', ...
    {'k','fullSupportLog10','chainSupport','tailSupport','dataTime', ...
    'buildTime','tailTime','crSolveTime','cvarSolveTime'});
writetable(summary, fullfile(outDir, 'scalability_summary.csv'));

disp(summary(:, {'n','mean_k','mean_fullSupportLog10','mean_chainSupport', ...
    'mean_tailSupport','mean_buildTime','mean_crSolveTime','mean_cvarSolveTime'}));
