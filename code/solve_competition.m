%% solve_competition.m
% End-to-end Project 1 solver for the ERT competition.
%
% Method (two stages):
%   1. TEMPLATE MATCH: compute the exact ERT forward misfit of every MNIST
%      training image against the measured data y_truth, and pick the best.
%      This identifies the digit and gives a clean starting image in the
%      correct "basin".
%   2. REFINE: run free gradient descent from that template to drive the
%      data misfit toward zero and recover the exact digit.
%
% Run this from the code/ folder:  >> solve_competition
% Produces: competition_answer.mat and competition_answer.png

clc; clear; close all;

%% ---- setup ----
domainDimX = 10; domainDimY = 5; sigmaBackground = 1;
ERTParams = paramPackGenerator([28 28], sigmaBackground, domainDimX, domainDimY);

% Measured data (the professor's competition vector)
yd = load('y_truth_measurement.mat');
y_obs = yd.y_truth(:);
fprintf('Loaded measured data: %d values.\n', numel(y_obs));

% MNIST training images (the prior: the answer is ~one of these)
mn = load(fullfile('MNIST Data','mnist.mat'));
labels = mn.training.labels;
M = double(reshape(mn.training.images, [28*28, size(mn.training.images,3)]));
if max(M(:)) > 1.5, M = M/255; end
N = size(M, 2);

%% ---- Stage 1: template matching (exact forward misfit over all 60k) ----
fprintf('Stage 1: template matching over %d MNIST images ...\n', N);
misfit = zeros(N,1);
t0 = tic;
for i = 1:N
    sig = reshape(sigmaBackground + M(:,i), [28 28]);
    [yi,~] = ERT2D(sig, 1, ERTParams);
    misfit(i) = 0.5*sum((yi - y_obs).^2);
    if mod(i,10000)==0, fprintf('  %d/%d (%.1f min)\n', i, N, toc(t0)/60); end
end
[~, order] = sort(misfit);
best_idx   = order(1);
digit      = labels(best_idx);
x_template = M(:, best_idx);
fprintf('  best match: image #%d, digit = %d, misfit = %.3e\n', ...
    best_idx, digit, misfit(best_idx));
fprintf('  top-8 digits: %s\n', mat2str(labels(order(1:8))'));

%% ---- Stage 2: refine (free gradient descent from the template) ----
fprintf('Stage 2: refining from the template ...\n');
eta = 5; gamma = 0.95; MaxIt = 1500;
x = x_template; theta = 0*x;
for it = 1:MaxIt
    sig = reshape(sigmaBackground + x, [28 28]);
    [y,J] = ERT2D(sig, 2, ERTParams);
    r = y - y_obs;
    theta = gamma*theta + eta*(J'*r);
    x = x - theta;
    x = max(x, -sigmaBackground + 1e-6);
end
sigma_answer = reshape(sigmaBackground + x, [28 28]);
[yfin,~] = ERT2D(sigma_answer, 1, ERTParams);
final_misfit = 0.5*sum((yfin - y_obs).^2);
fprintf('  refined misfit = %.3e (digit %d)\n', final_misfit, digit);

%% ---- save + plot ----
save('competition_answer.mat', 'sigma_answer', 'digit', 'final_misfit', ...
    'x_template', 'best_idx');

fig = figure('Position',[150 150 1000 450],'Color','w','Visible','off');
subplot(1,2,1); imagesc(reshape(sigmaBackground+x_template,[28 28]));
axis image; colorbar; caxis([1 2]);
title(sprintf('Stage 1: best MNIST template\n(digit %d, misfit %.2e)', ...
    digit, misfit(best_idx)));
subplot(1,2,2); imagesc(sigma_answer); axis image; colorbar; caxis([1 2]);
title(sprintf('Stage 2: refined answer\n(digit %d, misfit %.2e)', ...
    digit, final_misfit));
saveas(fig, 'competition_answer.png');
fprintf('Saved competition_answer.mat and competition_answer.png\n');
