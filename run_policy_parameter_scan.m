clear; clc;

% Lightweight scan for synthetic policy-comparison parameters.
% Goal: avoid degenerate CVaR-CR-SHP policies that sell exactly k items in
% the first stage for every instance, while preserving tail improvement.

outDir = fullfile(pwd, 'results');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

seeds = 1:3;
betaList = [0.60, 0.65, 0.70];
alphaList = [0.30, 0.35, 0.40, 0.45, 0.50];
n = 50;
gamma = 0.5;
lambda = 0.75;
Tnom = 200;
Ttest = 800;
testMode = 't_copula06';
modelNames = {'Nominal', 'CR-SHP', 'CVaR-CR-SHP', 'Mixed'};

rows = struct([]);
rowId = 0;

for beta = betaList
    for alpha = alphaList
        for seed = seeds
            rng(seed);

            params = base_params(seed, n, gamma, beta);
            data = generate_synthetic_shp_data(params);

            Xnom = generate_scenarios_from_marginals(data.supports, data.pNominal, ...
                Tnom, 'independent');
            nominalDist = make_empirical_dist_scan(Xnom);
            crDist = build_chain_worst_distribution(data.supports, data.pWorst);
            tailDist = tail_truncate_distribution(crDist, alpha);
            mixedDist = mix_distributions(nominalDist, 1-lambda, tailDist, lambda);

            sols = cell(4,1);
            sols{1} = solve_shp_approximation(data.a, data.k, nominalDist);
            sols{2} = solve_shp_approximation(data.a, data.k, crDist);
            sols{3} = solve_shp_approximation(data.a, data.k, tailDist);
            sols{4} = solve_shp_approximation(data.a, data.k, mixedDist);

            Xtest = build_test_scenarios_scan(data, Ttest, testMode);

            for mi = 1:4
                met = evaluate_shp_policy(data.a, data.k, sols{mi}.S, Xtest);
                rowId = rowId + 1;
                rows(rowId).beta = beta; %#ok<SAGROW>
                rows(rowId).alpha = alpha;
                rows(rowId).seed = seed;
                rows(rowId).model = modelNames{mi};
                rows(rowId).selectedCount = numel(sols{mi}.S);
                rows(rowId).tailSupport = tailDist.supportSize;
                rows(rowId).testMean = met.mean;
                rows(rowId).cvar10 = met.cvar10;
                rows(rowId).q05 = met.q05;
                rows(rowId).worst = met.worst;
            end
        end
    end
end

T = struct2table(rows);
writetable(T, fullfile(outDir, 'policy_parameter_scan.csv'));

summary = groupsummary(T, {'beta','alpha','model'}, 'mean', ...
    {'selectedCount','tailSupport','testMean','cvar10','q05','worst'});
writetable(summary, fullfile(outDir, 'policy_parameter_scan_summary.csv'));

disp(summary(:, {'beta','alpha','model','mean_selectedCount','mean_tailSupport','mean_testMean','mean_cvar10','mean_q05'}));

function params = base_params(seed, n, gamma, beta)
    params = struct();
    params.seed = seed;
    params.n = n;
    params.k = max(1, floor(gamma * n));
    params.aMode = 'beta_mu';
    params.beta = beta;
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
end

function dist = make_empirical_dist_scan(X)
    dist = struct();
    dist.support = X;
    dist.prob = ones(size(X,1),1) / size(X,1);
    dist.supportSize = size(X,1);
end

function X = build_test_scenarios_scan(data, Ttest, mode)
    switch mode
        case 't_copula06'
            X = generate_scenarios_from_marginals(data.supports, data.pNominal, ...
                Ttest, 't_copula', 'rho', 0.6, 'df', 4);
        otherwise
            error('Unsupported test mode.');
    end
end
