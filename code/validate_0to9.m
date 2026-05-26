%% validate_0to9.m
% Validation of the template-match + refine method on HELD-OUT digits.
%
% The 60k TRAINING images are the dictionary (the prior). We pull one UNSEEN
% image per class 0-9 from the TEST set, simulate its ERT readings with F,
% then run the same two-stage method (template match, then refine) to see
% whether we recover the correct digit. Because the test images are not in
% the dictionary, the method cannot cheat by matching a digit to itself.
%
% Run from the code/ folder:  >> validate_0to9
% Produces: ../figures/validation_0to9.png

clc; clear; close all;
rng(0);

%% setup (identical to solve_competition)
domainDimX = 10; domainDimY = 5; sigmaBackground = 1;
ERTParams = paramPackGenerator([28 28], sigmaBackground, domainDimX, domainDimY);

mn  = load(fullfile('MNIST Data','mnist.mat'));
Mtr = double(reshape(mn.training.images, [784, size(mn.training.images,3)]));
Mte = double(reshape(mn.test.images,     [784, size(mn.test.images,3)]));
sc = 1; if max(Mtr(:)) > 1.5, sc = 255; end       % same normalization for both
Mtr = Mtr/sc; Mte = Mte/sc;
Ltr = mn.training.labels;
Lte = mn.test.labels;

%% build the template dictionary (random training subset for speed)
Npool = 10000;
pool        = randperm(size(Mtr,2), Npool);
poolLabels  = Ltr(pool);

% Precompute the forward readings of every template ONCE (reused for all 10
% test digits, since they do not depend on the test image).
sig1 = reshape(sigmaBackground + Mtr(:,pool(1)), [28 28]);
[y1,~] = ERT2D(sig1, 1, ERTParams);  m = numel(y1);
Ytmpl = zeros(m, Npool); Ytmpl(:,1) = y1;
fprintf('Precomputing forward readings for %d templates ...\n', Npool);
t0 = tic;
for i = 2:Npool
    sig = reshape(sigmaBackground + Mtr(:,pool(i)), [28 28]);
    [yi,~] = ERT2D(sig, 1, ERTParams);
    Ytmpl(:,i) = yi;
    if mod(i,2000)==0, fprintf('  %d/%d (%.1f min)\n', i, Npool, toc(t0)/60); end
end

%% one test image per class 0-9
testIdx = zeros(10,1);
for d = 0:9, testIdx(d+1) = find(Lte==d, 1, 'first'); end

trueImgs = zeros(784,10); stage1Imgs = zeros(784,10); refinedImgs = zeros(784,10);
recClass = zeros(10,1);  s1misfit = zeros(10,1); finmisfit = zeros(10,1);

eta = 5; gamma = 0.95; MaxIt = 400;     % same refine settings, fewer iters

for k = 1:10
    d = k-1;
    xtrue = Mte(:, testIdx(k));  trueImgs(:,k) = xtrue;
    sigtrue = reshape(sigmaBackground + xtrue, [28 28]);
    [yobs,~] = ERT2D(sigtrue, 1, ERTParams);     % simulate this digit's readings

    % ---- Stage 1: nearest template by data misfit (no forward solves needed)
    D = Ytmpl - yobs;
    errs = 0.5*sum(D.^2, 1);
    [s1misfit(k), bi] = min(errs);
    recClass(k) = poolLabels(bi);
    xt = Mtr(:, pool(bi));  stage1Imgs(:,k) = xt;

    % ---- Stage 2: refine from the template
    x = xt; theta = 0*x;
    for it = 1:MaxIt
        sig = reshape(sigmaBackground + x, [28 28]);
        [y,J] = ERT2D(sig, 2, ERTParams);
        r = y - yobs;
        theta = gamma*theta + eta*(J'*r);
        x = x - theta;
        x = max(x, -sigmaBackground + 1e-6);
    end
    refinedImgs(:,k) = x;
    sigr = reshape(sigmaBackground + x, [28 28]);
    [yf,~] = ERT2D(sigr, 1, ERTParams);
    finmisfit(k) = 0.5*sum((yf - yobs).^2);
    fprintf('true %d  ->  recovered %d   (stage1 %.2e, refined %.2e)\n', ...
        d, recClass(k), s1misfit(k), finmisfit(k));
end

ncorrect = sum(recClass == (0:9)');
fprintf('\nClass accuracy: %d/10 (%.0f%%)\n', ncorrect, 100*ncorrect/10);

%% figure: 3 rows (true / Stage 1 / refined) x 10 cols (digits 0-9)
fig = figure('Position',[40 40 1500 520],'Color','w','Visible','off');
colormap(gray);
rows     = {trueImgs, stage1Imgs, refinedImgs};
rowNames = {'True (unseen)','Stage 1 match','Refined'};
for r = 1:3
    for k = 1:10
        subplot(3,10,(r-1)*10+k);
        imagesc(reshape(sigmaBackground + rows{r}(:,k), [28 28]));
        axis image; caxis([1 2]); set(gca,'XTick',[],'YTick',[]);
        if r == 1
            title(sprintf('%d', k-1), 'FontSize', 11);
        elseif r == 2
            correct = recClass(k) == (k-1);
            c = [0 0.6 0]; if ~correct, c = [0.85 0 0]; end
            title(sprintf('%d', recClass(k)), 'Color', c, 'FontWeight','bold','FontSize',11);
        end
        if k == 1, ylabel(rowNames{r}, 'FontSize', 10); end
    end
end
sgtitle(sprintf('Template-match + refine on 10 unseen digits:  %d/10 correct', ncorrect), 'FontSize', 13);
saveas(fig, fullfile('..','figures','validation_0to9.png'));
fprintf('Saved figures/validation_0to9.png\n');

%% slide-friendly figure: ONE row, each cell = unseen digit image with
%% a big bold "true -> recovered" label above (green = correct, red = wrong).
%% Tight tiledlayout, no inner title (the slide title carries the headline).
figS = figure('Position',[40 40 1700 620],'Color','w','Visible','off');
colormap(gray);
tl = tiledlayout(1,10,'TileSpacing','compact','Padding','tight');
for k = 1:10
    nexttile;
    imagesc(reshape(sigmaBackground + trueImgs(:,k), [28 28]));
    axis image; caxis([1 2]); set(gca,'XTick',[],'YTick',[]);
    correct = recClass(k) == (k-1);
    c = [0 0.55 0]; if ~correct, c = [0.85 0 0]; end
    title(sprintf('%d \\rightarrow %d', k-1, recClass(k)), ...
          'FontSize', 32, 'FontWeight','bold', 'Color', c, 'Interpreter','tex');
end
exportgraphics(figS, fullfile('..','figures','validation_slide.png'), 'Resolution', 200);
fprintf('Saved figures/validation_slide.png\n');
