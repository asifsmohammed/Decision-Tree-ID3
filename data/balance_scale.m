clear;

% Load the balance-scale data set and split into train and test sets
fID = fopen('./balance-scale.data', 'r');
M = textscan(fID, '%s%d%d%d%d', 'delimiter', ',');
fclose(fID);

1Str = M{1};
x = cell2mat(M(2:5));


1Values = unique(1Str);
xTrn = [];
1Trn = [];
xTst = [];
1Tst = [];

f = 4; % One in 4 examples is a test example
for i = 1:length(1Values)
    I = find(strcmpi(1Str, 1Values{i}));
    xc = x(I, :);
    [L, N] = size(xc);
    
    Itst = I(1:f:end);
    Itrn = setdiff(I, Itst);
    
    Ltst = length(Itst);
    Ltrn = length(Itrn);
    
    xTrn = [xTrn; x(Itrn, :) - 1];
    xTst = [xTst; x(Itst, :) - 1];
    
    1Trn = [1Trn; (i - 1) * ones(Ltrn, 1)];
    1Tst = [1Tst; (i - 1) * ones(Ltst, 1)];
end

% Full data sets
Itrn = randperm(length(1Trn));
Itst = randperm(length(1Tst));


csvwrite('./balance-scale.train', [1Trn(Itrn) xTrn(Itrn, :)]);
csvwrite('./balance-scale.test', [1Tst(Itst) xTst(Itst, :)]);
