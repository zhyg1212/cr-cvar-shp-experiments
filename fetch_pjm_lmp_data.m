function fetch_pjm_lmp_data(outDir, startDate, endDate, pnodeNames, apiKey)
%FETCH_PJM_LMP_DATA Download PJM day-ahead and real-time hourly LMP data.
%
% This function uses PJM Data Miner 2 feeds:
%   da_hrl_lmps : day-ahead hourly LMPs
%   rt_hrl_lmps : real-time hourly LMPs
%
% PJM Data Miner requires a subscription key. Pass it explicitly or set the
% environment variable PJM_API_KEY before running this function.

    if nargin < 5 || isempty(apiKey)
        apiKey = getenv('PJM_API_KEY');
    end
    if isempty(apiKey)
        error(['PJM API key is required. Set environment variable PJM_API_KEY ', ...
            'or pass apiKey to fetch_pjm_lmp_data.']);
    end
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    feeds = {'da_hrl_lmps', 'rt_hrl_lmps'};
    outFiles = {'pjm_da_lmp_raw.csv', 'pjm_rt_lmp_raw.csv'};

    for f = 1:numel(feeds)
        fprintf('Downloading %s...\n', feeds{f});
        T = download_one_feed(feeds{f}, startDate, endDate, pnodeNames, apiKey);
        writetable(T, fullfile(outDir, outFiles{f}));
        fprintf('Saved %d rows to %s\n', height(T), outFiles{f});
    end
end

function T = download_one_feed(feed, startDate, endDate, pnodeNames, apiKey)
    baseUrl = ['https://api.pjm.com/api/v1/', feed];
    rowCount = 50000;
    startRow = 1;
    allTables = {};

    dateFilter = [datestr(startDate, 'yyyy-mm-dd HH:MM'), ' to ', ...
        datestr(endDate, 'yyyy-mm-dd HH:MM')];

    opts = weboptions('HeaderFields', {'Ocp-Apim-Subscription-Key', apiKey}, ...
        'Timeout', 120);

    while true
        params = {
            'rowCount', num2str(rowCount), ...
            'startRow', num2str(startRow), ...
            'datetime_beginning_ept', dateFilter ...
        };

        if ~isempty(pnodeNames)
            params = [params, {'pnode_name', strjoin(pnodeNames, ',')}]; %#ok<AGROW>
        end

        S = webread(baseUrl, params{:}, opts);
        if ~isfield(S, 'items') || isempty(S.items)
            break;
        end

        chunk = struct2table(S.items);
        allTables{end+1} = chunk; %#ok<AGROW>

        if height(chunk) < rowCount
            break;
        end
        startRow = startRow + rowCount;
    end

    if isempty(allTables)
        T = table();
    else
        T = vertcat(allTables{:});
    end
end
