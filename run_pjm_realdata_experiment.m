clear; clc;

% PJM real-data experiment for CR-SHP / CVaR-CR-SHP.
%
% Required input:
%   data_pjm/pjm_da_lmp_raw.csv
%   data_pjm/pjm_rt_lmp_raw.csv
%
% These files can be produced by fetch_pjm_lmp_data.m if a PJM Data Miner
% subscription key is available.

baseDir = pwd;
dataDir = fullfile(baseDir, 'data_pjm');
outDir = fullfile(baseDir, 'results');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% To keep the binary decision dimension moderate, we use a small set of
% representative hubs/zones and peak hours. This can be expanded later.
cfg = struct();
cfg.pnodeNames = {'AEP GEN HUB', 'AEP-DAYTON HUB', 'ATSI GEN HUB', ...
    'CHICAGO GEN HUB', 'CHICAGO HUB', 'DOMINION HUB', 'EASTERN HUB', ...
    'N ILLINOIS HUB', 'NEW JERSEY HUB', 'OHIO HUB', 'WEST INT HUB', ...
    'WESTERN HUB'};
cfg.hours = 8:19;
cfg.trainDays = 60;
cfg.alpha = 0.10;
cfg.lambda = 0.50;
cfg.gamma = 0.50;
cfg.testStart = datetime(2025, 7, 1);
cfg.testEnd = datetime(2025, 7, 31);
cfg.solver = 'approx'; % 'approx' or 'local'
cfg.localMaxIter = 6;
cfg.localRandomStarts = 1;

daFile = fullfile(dataDir, 'pjm_da_lmp_raw.csv');
rtFile = fullfile(dataDir, 'pjm_rt_lmp_raw.csv');
if ~exist(daFile, 'file')
    daFile = fullfile(baseDir, 'da_hrl_lmps.csv');
end
if ~exist(rtFile, 'file')
    rtFile = fullfile(baseDir, 'rt_hrl_lmps.csv');
end

if ~exist(daFile, 'file') || ~exist(rtFile, 'file')
    fprintf('Missing PJM raw CSV files.\n');
    fprintf('Expected:\n  %s\n  %s\n\n', daFile, rtFile);
    fprintf('If you have a PJM API key, run for example:\n');
    fprintf('fetch_pjm_lmp_data(''%s'', datetime(2025,5,1), datetime(2025,8,1), cfg.pnodeNames, getenv(''PJM_API_KEY''));\n', dataDir);
    return;
end

TdaRaw = read_pjm_csv(daFile);
TrtRaw = read_pjm_csv(rtFile);

Tda = normalize_pjm_lmp_table(TdaRaw, 'da');
Trt = normalize_pjm_lmp_table(TrtRaw, 'rt');

testDays = (cfg.testStart:caldays(1):cfg.testEnd)';
modelNames = {'Nominal', 'CR-SHP', 'CVaR-CR-SHP', 'Mixed'};

rows = struct([]);
rowId = 0;

for d = 1:numel(testDays)
    testDay = testDays(d);
    trainStart = testDay - caldays(cfg.trainDays);
    trainEnd = testDay - caldays(1);

    try
        [Xtrain, itemInfo] = build_pjm_matrix(Trt, cfg.pnodeNames, cfg.hours, ...
            trainStart, trainEnd);
        [XrtTest, ~] = build_pjm_matrix(Trt, cfg.pnodeNames, cfg.hours, ...
            testDay, testDay);
        [XdaTest, ~] = build_pjm_matrix(Tda, cfg.pnodeNames, cfg.hours, ...
            testDay, testDay);
    catch ME
        warning('Skipping %s: %s', datestr(testDay), ME.message);
        continue;
    end

    if size(XrtTest, 1) ~= 1 || size(XdaTest, 1) ~= 1
        warning('Skipping %s: test day has incomplete data.', datestr(testDay));
        continue;
    end

    a = XdaTest(:);
    bTest = XrtTest;
    n = numel(a);
    k = max(1, floor(cfg.gamma * n));

    nominalDist = make_empirical_dist_pjm(Xtrain);
    [supports, probs] = empirical_marginals_pjm(Xtrain);
    crDist = build_chain_worst_distribution(supports, probs);
    tailDist = tail_truncate_distribution(crDist, cfg.alpha);
    mixedDist = mix_distributions(nominalDist, 1-cfg.lambda, tailDist, cfg.lambda);

    dists = {nominalDist, crDist, tailDist, mixedDist};
    sols = cell(4,1);
    for mi = 1:4
        sols{mi} = solve_policy_pjm(a, k, dists{mi}, cfg);
    end

    profits = zeros(4,1);
    for mi = 1:4
        met = evaluate_shp_policy(a, k, sols{mi}.S, bTest);
        profits(mi) = met.mean;
    end
    bestProfit = max(profits);

    for mi = 1:4
        rowId = rowId + 1;
        rows(rowId).testDate = datestr(testDay, 'yyyy-mm-dd'); %#ok<SAGROW>
        rows(rowId).model = modelNames{mi};
        rows(rowId).profit = profits(mi);
        rows(rowId).regretToBest = bestProfit - profits(mi);
        rows(rowId).n = n;
        rows(rowId).k = k;
        rows(rowId).chainSupport = crDist.supportSize;
        rows(rowId).tailSupport = tailDist.supportSize;
        rows(rowId).selectedCount = numel(sols{mi}.S);
    end

    fprintf('%s n=%d k=%d best=%.2f nominal=%.2f cr=%.2f cvar=%.2f mixed=%.2f\n', ...
        datestr(testDay, 'yyyy-mm-dd'), n, k, bestProfit, ...
        profits(1), profits(2), profits(3), profits(4));
end

if isempty(rows)
    warning('No PJM experiment rows were produced.');
    return;
end

T = struct2table(rows);
writetable(T, fullfile(outDir, 'pjm_realdata_results.csv'));
save(fullfile(outDir, 'pjm_realdata_results.mat'), 'T', 'cfg');

summary = summarize_pjm_results(T);
writetable(summary, fullfile(outDir, 'pjm_realdata_summary.csv'));
disp(summary);

function sol = solve_policy_pjm(a, k, dist, cfg)
    switch lower(cfg.solver)
        case 'approx'
            sol = solve_shp_approximation(a, k, dist);
        case 'local'
            opts = struct('maxIter', cfg.localMaxIter, ...
                'numRandomStarts', cfg.localRandomStarts);
            sol = solve_shp_local_search(a, k, dist, opts);
        otherwise
            error('Unknown solver: %s.', cfg.solver);
    end
end

function T = read_pjm_csv(fileName)
    opts = detectImportOptions(fileName, 'Delimiter', ',');
    T = readtable(fileName, opts);
end

function summary = summarize_pjm_results(T)
    models = unique(T.model, 'stable');
    rows = struct([]);
    for i = 1:numel(models)
        idx = strcmp(T.model, models{i});
        x = T.profit(idx);
        r = T.regretToBest(idx);
        rows(i).model = models{i}; %#ok<AGROW>
        rows(i).meanProfit = mean(x);
        rows(i).cvar10 = lower_tail_cvar_pjm(x, 0.10);
        rows(i).q05 = quantile_pjm(x, 0.05);
        rows(i).worst = min(x);
        rows(i).meanRegret = mean(r);
    end
    summary = struct2table(rows);
end

function v = lower_tail_cvar_pjm(x, alpha)
    x = sort(x(:), 'ascend');
    h = alpha * numel(x);
    fullCount = floor(h);
    frac = h - fullCount;
    if fullCount == 0
        v = x(1);
    elseif fullCount >= numel(x)
        v = mean(x);
    else
        v = (sum(x(1:fullCount)) + frac * x(fullCount + 1)) / h;
    end
end

function q = quantile_pjm(x, tau)
    x = sort(x(:), 'ascend');
    q = x(max(1, ceil(tau * numel(x))));
end

function dist = make_empirical_dist_pjm(X)
    dist = struct();
    dist.support = X;
    dist.prob = ones(size(X,1),1) / size(X,1);
    dist.supportSize = size(X,1);
end

function [supports, probs] = empirical_marginals_pjm(X)
    n = size(X,2);
    supports = cell(n,1);
    probs = cell(n,1);
    for i = 1:n
        vals = X(:,i);
        vals = vals(~isnan(vals));
        [u, ~, ic] = unique(vals);
        counts = accumarray(ic, 1);
        supports{i} = u(:)';
        probs{i} = (counts(:)' / sum(counts));
    end
end

function [X, itemInfo] = build_pjm_matrix(T, pnodeNames, hours, startDay, endDay)
    days = (startDay:caldays(1):endDay)';
    nItems = numel(pnodeNames) * numel(hours);
    X = nan(numel(days), nItems);
    itemInfo = cell(nItems, 2);

    item = 0;
    for p = 1:numel(pnodeNames)
        for h = 1:numel(hours)
            item = item + 1;
            itemInfo{item,1} = pnodeNames{p};
            itemInfo{item,2} = hours(h);
            for d = 1:numel(days)
                idx = strcmpi(T.pnodeName, pnodeNames{p}) & ...
                    T.marketDate == days(d) & T.hour == hours(h);
                if nnz(idx) ~= 1
                    error('Missing or duplicated observation for %s hour %d on %s.', ...
                        pnodeNames{p}, hours(h), datestr(days(d)));
                end
                X(d,item) = T.lmp(idx);
            end
        end
    end
end

function T = normalize_pjm_lmp_table(Traw, market)
    names = lower(Traw.Properties.VariableNames);

    timeVar = find_first_var(names, {'datetime_beginning_ept','datetimebeginningept','datetime_beginning_utc','datetimebeginningutc'});
    pnodeVar = find_first_var(names, {'pnode_name','pnodename'});

    if strcmpi(market, 'da')
        lmpVar = find_first_var(names, {'total_lmp_da','totallmpda','lmp','total_lmp'});
    else
        lmpVar = find_first_var(names, {'total_lmp_rt','totallmprt','lmp','total_lmp'});
    end

    dt = Traw.(Traw.Properties.VariableNames{timeVar});
    if iscell(dt)
        dt = parse_pjm_datetime(dt);
    elseif ischar(dt) || isstring(dt)
        dt = parse_pjm_datetime(dt);
    end
    if ~isdatetime(dt)
        dt = datetime(dt);
    end

    pnode = Traw.(Traw.Properties.VariableNames{pnodeVar});
    if iscell(pnode)
        pnode = string(pnode);
    end

    lmp = Traw.(Traw.Properties.VariableNames{lmpVar});

    T = table();
    T.datetime = dt;
    T.marketDate = dateshift(dt, 'start', 'day');
    T.hour = hour(dt);
    T.pnodeName = cellstr(upper(strtrim(string(pnode))));
    T.lmp = double(lmp);
end

function dt = parse_pjm_datetime(x)
    if iscell(x)
        x = string(x);
    end
    try
        dt = datetime(x, 'InputFormat', 'M/d/yyyy h:mm:ss a', 'Locale', 'en_US');
    catch
        try
            dt = datetime(x, 'InputFormat', 'MM/dd/yyyy hh:mm:ss a', 'Locale', 'en_US');
        catch
            try
                dt = datetime(x, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss');
            catch
                dt = datetime(x);
            end
        end
    end
end

function idx = find_first_var(names, candidates)
    idx = [];
    normalizedNames = regexprep(names, '[^a-z0-9]', '');
    for c = 1:numel(candidates)
        candidate = regexprep(lower(candidates{c}), '[^a-z0-9]', '');
        hit = find(strcmp(normalizedNames, candidate), 1, 'first');
        if ~isempty(hit)
            idx = hit;
            return;
        end
    end
    error('Cannot find required variable. Candidates: %s', strjoin(candidates, ', '));
end
