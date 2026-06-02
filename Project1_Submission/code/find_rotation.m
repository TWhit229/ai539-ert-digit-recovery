%% find_rotation.m
% Search for the rotation angle of the hidden digit and re-solve the
% competition under the rotation. The professor said the hidden image is
% rotated, which means upright MNIST templates cannot match it exactly --
% the upright-only result (digit 5, misfit 1.76e-6) is likely a "false
% friend." This script:
%   Phase 0 : quick rotation scan over our current best upright 5.
%   Phase 1 : balanced 1,000-template x 36-angle coarse scan.
%   Phase 2 : refine the angle to 0.5° around the Phase-1 winner.
%   Phase 3 : full 60k Stage 1 with all templates rotated by the final angle.
%   Phase 4 : Stage 2 pixel-level refinement from that rotated winner.

clc; clear; close all;
rng(0);

%% setup (identical to solve_competition)
sigmaBackground = 1;
ERTParams = paramPackGenerator([28 28], sigmaBackground, 10, 5);

yd = load('y_truth_measurement.mat'); y_obs = yd.y_truth(:);
mn = load(fullfile('MNIST Data','mnist.mat'));
Mtr = double(reshape(mn.training.images, [784, size(mn.training.images,3)]));
Ltr = mn.training.labels;
if max(Mtr(:)) > 1.5, Mtr = Mtr/255; end
N = size(Mtr, 2);

UPRIGHT_REFINED_MISFIT = 1.76e-6;
fprintf('Upright baseline: refined misfit = %.3e (digit 5)\n', UPRIGHT_REFINED_MISFIT);

%% Phase 0: rotate our current best (m_{51138}) and look for a dip
fprintf('\n--- Phase 0: rotating our current best upright 5 ---\n');
img_best = reshape(Mtr(:, 51138), [28 28]);
anglesP0 = 0:5:355;
misfitP0 = zeros(length(anglesP0), 1);
for a = 1:length(anglesP0)
    theta = anglesP0(a);
    img_rot = max(0, min(1, myrotate(img_best, theta)));
    sig = sigmaBackground + img_rot;
    [yi, ~] = ERT2D(sig, 1, ERTParams);
    misfitP0(a) = 0.5 * sum((yi - y_obs).^2);
end
[bP0, iP0] = min(misfitP0);
fprintf('  best rotation of m_{51138}: %d°, misfit %.3e\n', anglesP0(iP0), bP0);

%% Phase 1: balanced subset x coarse angle scan
fprintf('\n--- Phase 1: 1000 templates x 36 angles ---\n');
NperClass = 100;
subset = [];
for d = 0:9
    idx = find(Ltr == d);
    idx = idx(randperm(length(idx), NperClass));
    subset = [subset; idx];
end
Msub = Mtr(:, subset); Lsub = Ltr(subset);
Nsub = length(subset);

anglesCoarse = 0:10:350;
NA = length(anglesCoarse);
misfit = inf(Nsub, NA);
t0 = tic;
for a = 1:NA
    theta = anglesCoarse(a);
    for i = 1:Nsub
        img28 = reshape(Msub(:,i), [28 28]);
        img_rot = max(0, min(1, myrotate(img28, theta)));
        sig = sigmaBackground + img_rot;
        [yi, ~] = ERT2D(sig, 1, ERTParams);
        misfit(i,a) = 0.5 * sum((yi - y_obs).^2);
    end
    if mod(a, 6) == 0
        fprintf('  angle %d°, best so far %.3e (%.1f min)\n', theta, min(misfit(:)), toc(t0)/60);
    end
end

[bP1, idx] = min(misfit(:));
[bi, ba] = ind2sub(size(misfit), idx);
best_angle_coarse = anglesCoarse(ba);
best_class_coarse = Lsub(bi);
fprintf('Phase 1 winner: class %d at %d°, misfit %.3e\n', ...
        best_class_coarse, best_angle_coarse, bP1);

% Top-10 for context
[~, order] = sort(misfit(:));
fprintf('Top-10 (class @ angle, misfit):\n');
for k = 1:10
    [bk, ak] = ind2sub(size(misfit), order(k));
    fprintf('  %2d. class %d @ %3d°   misfit %.3e\n', k, Lsub(bk), anglesCoarse(ak), misfit(order(k)));
end

%% Phase 2: refine angle around the Phase-1 winner
fprintf('\n--- Phase 2: refine angle to 0.5° ---\n');
anglesFine = (best_angle_coarse - 9):0.5:(best_angle_coarse + 9);
[mb, ~] = min(misfit, [], 2);
topT = find(mb < 3*bP1);     % templates whose best coarse misfit is within 3x of overall best
fprintf('  refining on %d candidate templates, %d angles\n', length(topT), length(anglesFine));
misfitFine = inf(length(topT), length(anglesFine));
for a = 1:length(anglesFine)
    theta = anglesFine(a);
    for ii = 1:length(topT)
        i = topT(ii);
        img28 = reshape(Msub(:,i), [28 28]);
        img_rot = max(0, min(1, myrotate(img28, theta)));
        sig = sigmaBackground + img_rot;
        [yi, ~] = ERT2D(sig, 1, ERTParams);
        misfitFine(ii,a) = 0.5 * sum((yi - y_obs).^2);
    end
end
[bP2, idxF] = min(misfitFine(:));
[biF, baF] = ind2sub(size(misfitFine), idxF);
theta_final = anglesFine(baF);
best_class_fine = Lsub(topT(biF));
fprintf('Phase 2 winner: class %d at %.1f°, misfit %.3e\n', best_class_fine, theta_final, bP2);

%% Phase 3: full 60k Stage 1 with all templates rotated by theta_final
fprintf('\n--- Phase 3: full 60k Stage 1 at %.1f° ---\n', theta_final);
misfit_full = zeros(N, 1);
t0 = tic;
for i = 1:N
    img28 = reshape(Mtr(:,i), [28 28]);
    img_rot = max(0, min(1, myrotate(img28, theta_final)));
    sig = sigmaBackground + img_rot;
    [yi, ~] = ERT2D(sig, 1, ERTParams);
    misfit_full(i) = 0.5 * sum((yi - y_obs).^2);
    if mod(i, 10000) == 0
        fprintf('  %d/%d (%.1f min)\n', i, N, toc(t0)/60);
    end
end
[~, ord] = sort(misfit_full);
best_idx_rot = ord(1);
digit_rot = Ltr(best_idx_rot);
fprintf('Phase 3 winner: template %d, digit %d, misfit %.3e\n', ...
        best_idx_rot, digit_rot, misfit_full(best_idx_rot));
fprintf('Top-8 digits: %s\n', mat2str(Ltr(ord(1:8))'));

%% Phase 4: Stage 2 refinement starting from the rotated winning template
fprintf('\n--- Phase 4: Stage 2 refinement ---\n');
img_win = reshape(Mtr(:, best_idx_rot), [28 28]);
img_rot_win = max(0, min(1, myrotate(img_win, theta_final)));
x_template_rot = img_rot_win(:);

eta = 5; gamma_m = 0.95; MaxIt = 1500;
x = x_template_rot; vel = 0*x;
for it = 1:MaxIt
    sig = reshape(sigmaBackground + x, [28 28]);
    [y, J] = ERT2D(sig, 2, ERTParams);
    r = y - y_obs;
    vel = gamma_m*vel + eta*(J'*r);
    x = x - vel;
    x = max(x, -sigmaBackground + 1e-6);
end
sigma_answer_rot = reshape(sigmaBackground + x, [28 28]);
[yfin, ~] = ERT2D(sigma_answer_rot, 1, ERTParams);
final_misfit_rot = 0.5 * sum((yfin - y_obs).^2);
fprintf('Final refined: digit %d at %.1f°, misfit %.3e (baseline upright: %.3e)\n', ...
        digit_rot, theta_final, final_misfit_rot, UPRIGHT_REFINED_MISFIT);

%% Save and plot
save('rotated_answer.mat', 'sigma_answer_rot', 'digit_rot', 'final_misfit_rot', ...
     'best_idx_rot', 'theta_final', 'x_template_rot', 'misfit_full', ...
     'anglesCoarse', 'misfit');

fig = figure('Position',[60 60 1500 460],'Color','w','Visible','off');
colormap(gray);
subplot(1,3,1);
imagesc(reshape(sigmaBackground + Mtr(:, best_idx_rot), [28 28]));
axis image; caxis([1 2]); set(gca,'XTick',[],'YTick',[]);
title(sprintf('Upright template (digit %d)', digit_rot), 'FontSize', 14);

subplot(1,3,2);
imagesc(reshape(sigmaBackground + x_template_rot, [28 28]));
axis image; caxis([1 2]); set(gca,'XTick',[],'YTick',[]);
title(sprintf('Rotated by %.1f° (Stage 1)', theta_final), 'FontSize', 14);

subplot(1,3,3);
imagesc(sigma_answer_rot);
axis image; caxis([1 2]); set(gca,'XTick',[],'YTick',[]);
title(sprintf('Refined (misfit %.2e)', final_misfit_rot), 'FontSize', 14);

sgtitle(sprintf('Rotated recovery: digit %d at %.1f° (vs upright digit 5 at 0°)', ...
                digit_rot, theta_final), 'FontSize', 15, 'FontWeight','bold');
saveas(fig, fullfile('..','figures','rotated_answer.png'));
fprintf('Saved rotated_answer.mat and figures/rotated_answer.png\n');
