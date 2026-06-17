clear; clc;

% Candidate synthetic policy-comparison experiment after parameter tuning.
% Chosen to avoid the degenerate CVaR policy that sells exactly k items in
% every instance under very small alpha.

outDir = fullfile(pwd, 'results');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

seeds = 1:10;
n = 60;
gamma = 0.5;
alpha = 0.45;
lambda = 0.75;
Tnom = 400;
Ttest = 2000;

testModes = {'independent', 'gaussian06', 'comonotone', 't_copula06', 'adverse'};
modelNames = {'Nominal', 'CR-SHP', 'CVaR-CR-SHP', 'Mixed'};

rows = struct([]);
rowId = 0;

for seed = seeds
    rng(seed);

    params = struct();
    params.seed = seed;
    params.n = n;
    params.k = max(1, floor(gamma * n));
    params.aMode = 'beta_mu';
    params.beta = 0.70;
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

    data = generate_synthetic_shp_data(params);

    Xnom = generate_scenarios_from_marginals(data.supports, data.pNominal, ...
        Tnom, 'independent');
    nominalDist = make_empirical_dist_candidate(Xnom);
    crDist = build_chain_worst_distribution(data.supports, data.pWorst);
    tailDist = tail_truncate_distribution(crDist, alpha);
    mixedDist = mix_distributions(nominalDist, 1-lambda, tailDist, lambda);

    lsOpts = struct('maxIter', 8, 'numRandomStarts', 1);
    sols = cell(4,1);
    sols{1} = solve_shp_local_search(data.a, data.k, nominalDist, lsOpts);
    sols{2} = solve_shp_local_search(data.a, data.k, crDist, lsOpts);
    sols{3} = solve_shp_local_search(data.a, data.k, tailDist, lsOpts);
    sols{4} = solve_shp_local_search(data.a, data.k, mixedDist, lsOpts);

    for tm = 1:numel(testModes)
        mode = testModes{tm};
        if strcmp(mode, 'adverse')
            testDist = crDist;
            useDiscrete = true;
        else
            Xtest = build_test_scenarios_candidate(data, Ttest, mode);
            useDiscrete = false;
        end

        meanVals = zeros(4,1);
        metrics = cell(4,1);
        for mi = 1:4
            if useDiscrete
                metrics{mi} = evaluate_shp_policy_dist(data.a, data.k, sols{mi}.S, testDist);
            else
                metrics{mi} = evaluate_shp_policy(data.a, data.k, sols{mi}.S, Xtest);
            end
            meanVals(mi) = metrics{mi}.mean;
        end
        oracleMean = max(meanVals);

        for mi = 1:4
            rowId = rowId + 1;
            rows(rowId).n = n; %#ok<SAGROW>
            rows(rowId).gamma = gamma;
            rows(rowId).k = data.k;
            rows(rowId).seed = seed;
            rows(rowId).alpha = alpha;
            rows(rowId).beta = params.beta;
            rows(rowId).testMode = mode;
            rows(rowId).model = modelNames{mi};
            rows(rowId).trainObj = sols{mi}.objective;
            rows(rowId).testMean = metrics{mi}.mean;
            rows(rowId).cvar10 = metrics{mi}.cvar10;
            rows(rowId).q05 = metrics{mi}.q05;
            rows(rowId).worst = metrics{mi}.worst;
            rows(rowId).regretToBest = oracleMean - metrics{mi}.mean;
            rows(rowId).supportCR = crDist.supportSize;
            rows(rowId).supportTail = tailDist.supportSize;
            rows(rowId).selectedCount = numel(sols{mi}.S);
        end
    end

    fprintf('seed=%d selected counts: Nom=%d CR=%d CVaR=%d Mix=%d tailSupport=%d\n', ...
        seed, numel(sols{1}.S), numel(sols{2}.S), numel(sols{3}.S), ...
        numel(sols{4}.S), tailDist.supportSize);
end

T = struct2table(rows);
save(fullfile(outDir, 'policy_comparison_candidate_results.mat'), 'T');
writetable(T, fullfile(outDir, 'policy_comparison_candidate_results.csv'));

summary = groupsummary(T, {'testMode','model'}, 'mean', ...
    {'testMean','cvar10','q05','worst','regretToBest','selectedCount','supportTail'});
writetable(summary, fullfile(outDir, 'policy_comparison_candidate_summary.csv'));

disp(summary(:, {'testMode','model','mean_selectedCount','mean_testMean','mean_cvar10','mean_q05','mean_regretToBest'}));

function dist = make_empirical_dist_candidate(X)
    dist = struct();
    dist.support = X;
    dist.prob = ones(size(X,1),1) / size(X,1);
    dist.supportSize = size(X,1);
end

function X = build_test_scenarios_candidate(data, Ttest, mode)
    switch mode
        case 'independent'
            X = generate_scenarios_from_marginals(data.supports, data.pNominal, ...
                Ttest, 'independent');
        case 'gaussian06'
            X = generate_scenarios_from_marginals(data.supports, data.pNominal, ...
                Ttest, 'gaussian', 'rho', 0.6);
        case 'comonotone'
            X = generate_scenarios_from_marginals(data.supports, data.pNominal, ...
                Ttest, 'comonotone');
        case 't_copula06'
            X = generate_scenarios_from_marginals(data.supports, data.pNominal, ...
                Ttest, 't_copula', 'rho', 0.6, 'df', 4);
        otherwise
            error('Unsupported test mode: %s.', mode);
    end
end
