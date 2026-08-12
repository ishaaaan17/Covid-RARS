function build_figures
%BUILD_FIGURES Generate the four main JBHI figures from repository tables.

scriptDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(fileparts(fileparts(scriptDir)));
tablesDir = fullfile(rootDir, 'reports', 'tables');

set(groot, 'defaultFigureColor', 'w');
set(groot, 'defaultAxesFontName', 'Arial');
set(groot, 'defaultTextFontName', 'Arial');
set(groot, 'defaultAxesFontSize', 7.6);
set(groot, 'defaultTextColor', [0.12 0.15 0.17]);
set(groot, 'defaultTextInterpreter', 'none');
set(groot, 'defaultLegendInterpreter', 'none');
set(groot, 'defaultAxesTickLabelInterpreter', 'none');

figure1StudyDesign(scriptDir);
figure2ExternalTransport(tablesDir, scriptDir);
figure3EndpointReliability(tablesDir, scriptDir);
figure4ContextAndStability(tablesDir, scriptDir);
end


function figure1StudyDesign(outDir)
c = palette();
fig = newFigure(7.16, 3.05);

addSection(fig, 0.915, 'SOURCE DEVELOPMENT', c.blue);
source = [
    0.020 0.675 0.165 0.150
    0.220 0.675 0.135 0.150
    0.392 0.675 0.175 0.150
    0.604 0.675 0.165 0.150
    0.806 0.675 0.176 0.150
];
sourceText = {
    sprintf('Coswara source\n2,114 participants\ncough | breath | speech')
    sprintf('16 kHz mono\nquality control\nevent extraction')
    sprintf('Acoustic summaries\nComParE 2016 | IS10\n10,140 numeric candidates')
    sprintf('Source-only ranking\ntop 800 frozen\nper modality')
    sprintf('Four classifiers\nparticipant aggregation\nvalidation-derived fusion')
};
for i = 1:size(source, 1)
    edge = c.border;
    if i == 1 || i == size(source, 1)
        edge = c.blue;
    end
    addNode(fig, source(i, :), sourceText{i}, c.sourceFill, edge, 7.0);
end
for i = 1:size(source, 1)-1
    addArrow(fig, rightMid(source(i, :)), leftMid(source(i+1, :)), c.grey);
end

addSection(fig, 0.585, 'SOURCE EVALUATION', c.purple);
checks = [
    0.235 0.390 0.205 0.115
    0.482 0.390 0.205 0.115
    0.729 0.390 0.235 0.115
];
checkText = {
    sprintf('Participant-disjoint\nsource test')
    sprintf('Time-stratified\nparticipant split')
    sprintf('Retrospective early-to-late\ncalendar stress test')
};
for i = 1:size(checks, 1)
    addNode(fig, checks(i, :), checkText{i}, c.robustnessFill, c.purple, 7.0);
end
branchX = source(5, 1) + source(5, 3) / 2;
branchY = 0.610;
annotation(fig, 'arrow', [branchX branchX], [source(5, 2) branchY], ...
    'Color', c.purple, 'LineWidth', 0.9, 'HeadLength', 5, 'HeadWidth', 5);
centres = checks(:, 1) + checks(:, 3) / 2;
annotation(fig, 'line', [centres(1) branchX], [branchY branchY], ...
    'Color', c.purple, 'LineWidth', 0.9);
for i = 1:numel(centres)
    annotation(fig, 'arrow', [centres(i) centres(i)], ...
        [branchY checks(i, 2) + checks(i, 4)], 'Color', c.purple, ...
        'LineWidth', 0.9, 'HeadLength', 5, 'HeadWidth', 5);
end

addSection(fig, 0.302, 'INDEPENDENT EXTERNAL TRANSFER', c.orange);
target = [
    0.020 0.105 0.185 0.115
    0.247 0.105 0.195 0.115
    0.486 0.105 0.215 0.115
    0.745 0.105 0.237 0.115
];
targetText = {
    sprintf('COUGHVID target\n8,331 recording UUIDs\ncough only')
    sprintf('Same deterministic\naudio preprocessing')
    sprintf('Frozen cough features,\nmodels, and thresholds\nno target fitting/reselection')
    sprintf('External label agreement\nplus calibration and\noperating-point analyses')
};
for i = 1:size(target, 1)
    edge = c.border;
    if i ~= 2
        edge = c.orange;
    end
    addNode(fig, target(i, :), targetText{i}, c.targetFill, edge, 7.0);
end
for i = 1:size(target, 1)-1
    addArrow(fig, rightMid(target(i, :)), leftMid(target(i+1, :)), c.orange);
end

annotation(fig, 'textbox', [0.020 0.018 0.962 0.047], ...
    'String', 'Audit outputs: bootstrap uncertainty | metadata controls | feature stability | source-target separation', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'FontSize', 6.7, 'Color', c.black, 'BackgroundColor', c.auditFill, ...
    'EdgeColor', [0.60 0.68 0.62], 'LineWidth', 0.7, 'Margin', 1);

writeFigure(fig, outDir, 'fig1_study_design');
end


function figure2ExternalTransport(tablesDir, outDir)
c = palette();
transfer = readtable(fullfile(tablesDir, 'reviewer_external_model_family_transfer_summary.csv'), ...
    'TextType', 'string', 'VariableNamingRule', 'preserve');
delta = readDeltaTable(fullfile(tablesDir, 'final_validation_delta_bootstrap_ci.csv'));
transfer.label = shortModels(transfer.family_model);
[~, order] = sort(transfer.internal_auroc, 'descend');
transfer = transfer(order, :);

fig = newFigure(7.16, 2.78);
axA = axes(fig, 'Position', [0.095 0.205 0.235 0.610]);
axB = axes(fig, 'Position', [0.405 0.205 0.225 0.610]);
axC = axes(fig, 'Position', [0.715 0.205 0.235 0.610]);

y = (1:height(transfer))';
hold(axA, 'on');
hChance = xline(axA, 0.5, '--', 'Color', c.grey, 'LineWidth', 0.8);
for i = 1:height(transfer)
    plot(axA, [transfer.external_auroc(i) transfer.internal_auroc(i)], [y(i) y(i)], ...
        '-', 'Color', [0.55 0.59 0.63], 'LineWidth', 1.3);
end
hSource = scatter(axA, transfer.internal_auroc, y, 28, c.blue, 'o', 'filled');
hTarget = scatter(axA, transfer.external_auroc, y, 28, c.orange, 's', 'filled');
uistack(hChance, 'bottom');
formatMetricAxis(axA);
set(axA, 'YTick', y, 'YTickLabel', cellstr(transfer.label), 'YDir', 'reverse');
xlim(axA, [0.45 0.90]);
ylim(axA, [0.5 height(transfer)+0.5]);
xlabel(axA, 'AUROC');
panelTitle(axA, 'a', 'Source-to-target AUROC');
leg = legend(axA, [hSource hTarget], {'Coswara source', 'COUGHVID external'}, ...
    'Orientation', 'horizontal', 'Box', 'off', 'FontSize', 6.8);
set(leg, 'Units', 'normalized', 'Position', [0.095 0.045 0.235 0.050]);
set(leg, 'TextColor', c.black);
set(axA, 'Position', [0.095 0.205 0.235 0.610]);

ids = [
    "existing_cough_catboost_smote_f80_minus_coughvid_external"
    "existing_cough_lightgbm_smote_f80_minus_coughvid_external"
    "existing_cough_svc_rbf_f60_minus_coughvid_external"
    "existing_cough_xgboost_smote_f80_minus_coughvid_external"
];
ciLabels = ["CatBoost"; "LightGBM"; "RBF-SVC"; "XGBoost"];
ci = delta(delta.metric == "auroc", :);
rowIndex = zeros(numel(ids), 1);
for i = 1:numel(ids)
    rowIndex(i) = find(ci.comparison_id == ids(i), 1);
end
ci = ci(rowIndex, :);
yCi = (1:height(ci))';
hold(axB, 'on');
neg = ci.delta - ci.ci_low;
pos = ci.ci_high - ci.delta;
errorbar(axB, ci.delta, yCi, neg, pos, 'horizontal', 'o', ...
    'Color', c.red, 'MarkerFaceColor', c.red, 'MarkerSize', 4.2, ...
    'CapSize', 5, 'LineWidth', 1.0);
xline(axB, 0, '-', 'Color', c.grey, 'LineWidth', 0.8);
for i = 1:height(ci)
    text(axB, ci.delta(i), yCi(i) - 0.18, sprintf('%.3f', ci.delta(i)), ...
        'FontSize', 6.8, 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', 'Color', c.black);
end
formatMetricAxis(axB);
set(axB, 'YTick', yCi, 'YTickLabel', cellstr(ciLabels), 'YDir', 'reverse');
xlim(axB, [0.235 0.415]);
ylim(axB, [0.5 height(ci)+0.5]);
xlabel(axB, sprintf('Internal - external AUROC\n(95%% bootstrap CI)'));
panelTitle(axB, 'b', 'Matched AUROC decline');

[~, externalOrder] = sort(transfer.external_auprc, 'descend');
external = transfer(externalOrder, :);
yExt = (1:height(external))';
prevalence = 285 / 8331;
hold(axC, 'on');
hPrevalence = xline(axC, prevalence, '--', 'Color', c.orange, 'LineWidth', 1.0);
for i = 1:height(external)
    plot(axC, [prevalence external.external_auprc(i)], [yExt(i) yExt(i)], ...
        '-', 'Color', [0.60 0.64 0.68], 'LineWidth', 1.1);
end
for i = 1:height(external)
    color = c.purple;
    if external.model_family(i) == "compare_is10_handcrafted"
        color = c.blue;
    end
    scatter(axC, external.external_auprc(i), yExt(i), 28, color, 'o', 'filled');
    if external.external_auprc(i) < prevalence
        labelX = external.external_auprc(i) - 0.0018;
        labelAlignment = 'right';
    else
        labelX = external.external_auprc(i) + 0.0018;
        labelAlignment = 'left';
    end
    text(axC, labelX, yExt(i), sprintf('%.3f', external.external_auprc(i)), ...
        'FontSize', 6.7, 'HorizontalAlignment', labelAlignment, ...
        'VerticalAlignment', 'middle', 'Color', c.black);
end
uistack(hPrevalence, 'bottom');
formatMetricAxis(axC);
set(axC, 'YTick', yExt, 'YTickLabel', cellstr(external.label), 'YDir', 'reverse');
xlim(axC, [0 0.052]);
xticks(axC, [0 0.02 0.04]);
ylim(axC, [0.5 height(external)+0.5]);
xlabel(axC, 'External AUPRC');
panelTitle(axC, 'c', 'External AUPRC');
text(axC, prevalence, 0.65, 'prevalence', 'Color', c.orange, 'FontSize', 6.5, ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');

writeFigure(fig, outDir, 'fig2_external_transport');
end


function figure3EndpointReliability(tablesDir, outDir)
c = palette();
operating = readtable(fullfile(tablesDir, 'final_validation_fixed_sensitivity_operating_points.csv'), ...
    'TextType', 'string', 'VariableNamingRule', 'preserve');
recal = readtable(fullfile(tablesDir, 'coughvid_partial_recalibration_metrics.csv'), ...
    'TextType', 'string', 'VariableNamingRule', 'preserve');
operating = operating(operating.evaluation_protocol == "coswara_to_coughvid_compare_is10_external" ...
    & operating.target_sensitivity == 0.9, :);
modelIds = ["catboost_smote_f80"; "lightgbm_smote_f80"; "svc_rbf_f60"; "xgboost_smote_f80"];
labels = ["CatBoost"; "LightGBM"; "RBF-SVC"; "XGBoost"];
idx = zeros(4, 1);
for i = 1:4
    idx(i) = find(operating.model_name == modelIds(i), 1);
end
operating = operating(idx, :);

fig = newFigure(7.16, 2.82);
axA = axes(fig, 'Position', [0.095 0.205 0.235 0.625]);
axB = axes(fig, 'Position', [0.405 0.205 0.225 0.625]);
axC = axes(fig, 'Position', [0.710 0.205 0.250 0.625]);
y = (1:4)';

hold(axA, 'on');
for i = 1:4
    plot(axA, [0 operating.specificity(i)], [y(i) y(i)], '-', ...
        'Color', [0.55 0.59 0.63], 'LineWidth', 1.3);
end
scatter(axA, operating.specificity, y, 30, c.blue, 'o', 'filled');
for i = 1:4
    text(axA, operating.specificity(i) + 0.009, y(i), sprintf('%.3f', operating.specificity(i)), ...
        'FontSize', 6.8, 'VerticalAlignment', 'middle');
end
formatMetricAxis(axA);
set(axA, 'YTick', y, 'YTickLabel', cellstr(labels), 'YDir', 'reverse');
xlim(axA, [0 0.20]);
ylim(axA, [0.5 4.5]);
xlabel(axA, 'Specificity vs. pseudo-label');
panelTitle(axA, 'a', 'Specificity at sensitivity >= 0.90');

prevalence = 285 / 8331;
hold(axB, 'on');
hPrevalence = xline(axB, prevalence, '--', 'Color', c.orange, 'LineWidth', 1.0);
for i = 1:4
    plot(axB, [0 operating.precision(i)], [y(i) y(i)], '-', ...
        'Color', [0.55 0.59 0.63], 'LineWidth', 1.3);
end
scatter(axB, operating.precision, y, 30, c.green, 'o', 'filled');
uistack(hPrevalence, 'bottom');
for i = 1:4
    text(axB, operating.precision(i) + 0.0024, y(i), sprintf('%.3f', operating.precision(i)), ...
        'FontSize', 6.8, 'VerticalAlignment', 'middle');
end
formatMetricAxis(axB);
set(axB, 'YTick', y, 'YTickLabel', cellstr(labels), 'YDir', 'reverse');
xlim(axB, [0 0.054]);
ylim(axB, [0.25 4.5]);
xlabel(axB, 'Precision vs. pseudo-label');
panelTitle(axB, 'b', 'Precision at sensitivity >= 0.90');
text(axB, prevalence, 0.34, sprintf('prevalence %.3f', prevalence), 'Color', c.orange, ...
    'FontSize', 6.5, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');

lightgbm = recal(recal.model_name == "lightgbm_smote_f80", :);
methodIds = ["original"; "platt"; "isotonic"];
methodLabels = ["Original"; "Platt"; "Isotonic"];
methodColors = [c.grey; c.blue; c.green];
metricNames = ["AUROC"; "Brier score"; "ECE"];
metricValues = zeros(3, 3);
for i = 1:3
    row = lightgbm(lightgbm.recalibration_method == methodIds(i), :);
    metricValues(i, :) = [row.auroc(1), row.brier(1), row.ece(1)];
end
offset = [-0.17 0 0.17];
markers = {'o', 's', '^'};
hold(axC, 'on');
h = gobjects(3, 1);
for i = 1:3
    h(i) = scatter(axC, metricValues(i, :), (1:3) + offset(i), 30, ...
        methodColors(i, :), markers{i}, 'filled');
    for j = 1:3
        value = metricValues(i, j);
        if value < 0.001
            valueLabel = '<0.001';
        else
            valueLabel = sprintf('%.3f', value);
        end
        text(axC, value + 0.024, j + offset(i), valueLabel, 'FontSize', 6.4, ...
            'Color', methodColors(i, :), 'VerticalAlignment', 'middle');
    end
end
formatMetricAxis(axC);
set(axC, 'YTick', 1:3, 'YTickLabel', cellstr(metricNames), 'YDir', 'reverse');
xlim(axC, [0 0.64]);
ylim(axC, [0.5 3.5]);
xlabel(axC, 'Metric value');
panelTitle(axC, 'c', 'Held-out recalibration');
leg = legend(axC, h, cellstr(methodLabels), 'Location', 'southeast', ...
    'Box', 'off', 'FontSize', 6.6);
set(leg, 'TextColor', c.black);

writeFigure(fig, outDir, 'fig3_endpoint_reliability');
end


function figure4ContextAndStability(tablesDir, outDir)
c = palette();
shuffle = readtable(fullfile(tablesDir, 'metadata_confounding_shuffle_retrain_sanity.csv'), ...
    'TextType', 'string', 'VariableNamingRule', 'preserve');
importance = readtable(fullfile(tablesDir, 'metadata_confounding_permutation_group_summary.csv'), ...
    'TextType', 'string', 'VariableNamingRule', 'preserve');
stability = readtable(fullfile(tablesDir, 'reviewer_feature_selection_stability.csv'), ...
    'TextType', 'string', 'VariableNamingRule', 'preserve');
temporal = readtable(fullfile(tablesDir, 'compare_is10_final_validation_split_summary.csv'), ...
    'TextType', 'string', 'VariableNamingRule', 'preserve');

fig = newFigure(7.16, 4.05);
axA = axes(fig, 'Position', [0.165 0.610 0.300 0.275]);
axB = axes(fig, 'Position', [0.655 0.610 0.285 0.275]);
axC = axes(fig, 'Position', [0.165 0.145 0.300 0.275]);
axD = axes(fig, 'Position', [0.655 0.145 0.285 0.275]);

auditIds = ["full_safe_metadata"; "symptoms_only"; "demographic_protocol_only"];
auditLabels = ["Symptoms + context"; "Symptoms only"; "Demographic + protocol"];
idx = zeros(3, 1);
for i = 1:3
    idx(i) = find(shuffle.audit_model == auditIds(i), 1);
end
shuffle = shuffle(idx, :);
y = (1:3)';
hold(axA, 'on');
hChance = xline(axA, 0.5, '--', 'Color', c.grey, 'LineWidth', 0.8);
neg = shuffle.shuffled_auroc_mean - shuffle.shuffled_auroc_ci_low;
pos = shuffle.shuffled_auroc_ci_high - shuffle.shuffled_auroc_mean;
hShuffle = errorbar(axA, shuffle.shuffled_auroc_mean, y, neg, pos, 'horizontal', 'd', ...
    'Color', c.grey, 'MarkerFaceColor', 'w', 'MarkerSize', 4.0, ...
    'CapSize', 5, 'LineWidth', 0.9);
hObserved = scatter(axA, shuffle.observed_auroc, y, 32, c.blue, 'o', 'filled');
for i = 1:3
    text(axA, shuffle.observed_auroc(i) + 0.018, y(i), sprintf('%.3f', shuffle.observed_auroc(i)), ...
        'Color', c.blue, 'FontSize', 6.8, 'VerticalAlignment', 'middle');
end
uistack(hChance, 'bottom');
formatMetricAxis(axA);
set(axA, 'YTick', y, 'YTickLabel', cellstr(auditLabels), 'YDir', 'reverse');
xlim(axA, [0.44 1.04]);
ylim(axA, [0.5 3.5]);
xlabel(axA, 'AUROC');
panelTitle(axA, 'a', 'Metadata control');
leg = legend(axA, [hObserved hShuffle], {'Observed', 'Shuffled (95% CI)'}, ...
    'Orientation', 'horizontal', 'Box', 'off', 'FontSize', 6.5);
set(leg, 'Units', 'normalized', 'Position', [0.155 0.485 0.320 0.045]);
set(leg, 'TextColor', c.black);
set(axA, 'Position', [0.165 0.610 0.300 0.275]);

full = importance(importance.audit_model == "full_safe_metadata", :);
[~, order] = sort(full.importance_share, 'descend');
full = full(order, :);
groupLabels = strings(height(full), 1);
for i = 1:height(full)
    switch full.feature_group(i)
        case "recording_protocol"
            groupLabels(i) = "Recording protocol";
        case "symptom_or_label_proxy"
            groupLabels(i) = "Symptoms";
        case "comorbidity_proxy"
            groupLabels(i) = "Comorbidity";
        otherwise
            groupLabels(i) = "Demographic";
    end
end
groupColors = [c.orange; c.blue; c.green; c.purple];
yB = (1:height(full))';
hold(axB, 'on');
for i = 1:height(full)
    barh(axB, yB(i), full.importance_share(i), 0.56, 'FaceColor', groupColors(i, :), 'EdgeColor', 'none');
    text(axB, full.importance_share(i) + 0.012, yB(i), sprintf('%.1f%%', 100*full.importance_share(i)), ...
        'FontSize', 6.8, 'FontWeight', 'bold', 'VerticalAlignment', 'middle');
end
formatMetricAxis(axB);
set(axB, 'YTick', yB, 'YTickLabel', cellstr(groupLabels), 'YDir', 'reverse');
xlim(axB, [0 0.80]);
ylim(axB, [0.5 height(full)+0.5]);
xticks(axB, [0 0.2 0.4 0.6 0.8]);
xticklabels(axB, {'0%', '20%', '40%', '60%', '80%'});
xlabel(axB, 'Permutation-importance share');
panelTitle(axB, 'b', 'Metadata attribution');

parts = [stability.early_only_count(1); stability.overlap_count(1); stability.late_only_count(1)];
featureLabels = ["Early only"; "Shared"; "Late only"];
featureColors = [c.blue; c.green; c.orange];
yC = (1:3)';
hold(axC, 'on');
for i = 1:3
    barh(axC, yC(i), parts(i), 0.54, 'FaceColor', featureColors(i, :), 'EdgeColor', 'none');
    text(axC, parts(i) + 14, yC(i), sprintf('%d', parts(i)), 'FontSize', 6.8, ...
        'FontWeight', 'bold', 'VerticalAlignment', 'middle');
end
formatMetricAxis(axC);
set(axC, 'YTick', yC, 'YTickLabel', cellstr(featureLabels), 'YDir', 'reverse');
xlim(axC, [0 790]);
ylim(axC, [0.5 3.5]);
xlabel(axC, sprintf('Features (Jaccard = %.3f; union = %d)', ...
    stability.jaccard_overlap(1), sum(parts)));
panelTitle(axC, 'c', 'Feature-set overlap');

temporal = temporal(temporal.evaluation_protocol == "compare_is10_temporal_early_to_late", :);
splitIds = ["train"; "validation"; "test"];
splitLabels = ["Training"; "Validation"; "Test"];
idx = zeros(3, 1);
for i = 1:3
    idx(i) = find(temporal.temporal_split == splitIds(i), 1);
end
values = temporal.positive_prevalence(idx);
splitColors = [c.blue; c.purple; c.orange];
yD = (1:3)';
hold(axD, 'on');
for i = 1:3
    barh(axD, yD(i), values(i), 0.54, 'FaceColor', splitColors(i, :), 'EdgeColor', 'none');
    text(axD, values(i) + 0.025, yD(i), sprintf('%.1f%%', 100*values(i)), ...
        'FontSize', 6.8, 'FontWeight', 'bold', 'VerticalAlignment', 'middle');
end
formatMetricAxis(axD);
set(axD, 'YTick', yD, 'YTickLabel', cellstr(splitLabels), 'YDir', 'reverse');
xlim(axD, [0 1]);
ylim(axD, [0.5 3.5]);
xticks(axD, [0 0.2 0.4 0.6 0.8 1.0]);
xticklabels(axD, {'0%', '20%', '40%', '60%', '80%', '100%'});
xlabel(axD, 'Positive-label prevalence');
panelTitle(axD, 'd', 'Temporal cohort prevalence');

writeFigure(fig, outDir, 'fig4_context_and_stability');
end


function fig = newFigure(widthInches, heightInches)
fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', ...
    'Position', [0.5 0.5 widthInches heightInches], ...
    'PaperPositionMode', 'auto', 'InvertHardcopy', 'off');
end


function formatMetricAxis(ax)
set(ax, 'FontName', 'Arial', 'FontSize', 7.2, 'LineWidth', 0.7, ...
    'Box', 'off', 'TickDir', 'out', 'Layer', 'bottom', ...
    'Color', 'white', 'XColor', [0.12 0.15 0.17], 'YColor', [0.12 0.15 0.17], ...
    'XGrid', 'on', 'YGrid', 'off', 'GridColor', [0.78 0.81 0.83], ...
    'GridAlpha', 0.75, 'Clipping', 'on');
end


function panelTitle(ax, letter, titleText)
text(ax, 0, 1.075, sprintf('(%s) %s', letter, titleText), ...
    'Units', 'normalized', 'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'bottom', 'FontSize', 7.6, ...
    'FontWeight', 'bold', 'Color', [0.12 0.15 0.17], 'Clipping', 'off');
end


function writeFigure(fig, outDir, stem)
pdfPath = fullfile(outDir, [stem '.pdf']);
pngPath = fullfile(outDir, [stem '.png']);
exportgraphics(fig, pdfPath, 'ContentType', 'vector', 'BackgroundColor', 'white');
exportgraphics(fig, pngPath, 'Resolution', 300, 'BackgroundColor', 'white');
close(fig);
end


function labels = shortModels(names)
labels = strings(size(names));
for i = 1:numel(names)
    switch names(i)
        case "compare_is10_catboost_smote_f80"
            labels(i) = "CatBoost";
        case "compare_is10_lightgbm_smote_f80"
            labels(i) = "LightGBM";
        case "compare_is10_svc_rbf_f60"
            labels(i) = "RBF-SVC";
        case "compare_is10_xgboost_smote_f80"
            labels(i) = "XGBoost";
        case "wavlm_base_plus_pooled_cough"
            labels(i) = "WavLM";
        case "cnn_bigru"
            labels(i) = "CNN-BiGRU";
        otherwise
            labels(i) = names(i);
    end
end
end


function output = readDeltaTable(path)
fid = fopen(path, 'r');
if fid < 0
    error('Could not open delta-bootstrap table: %s', path);
end
cleanup = onCleanup(@() fclose(fid));
format = ['%s%f%f%f%f%f%f%s%f%s%s%s%s%f%f%s%s%s%f%f' ...
    '%q%q%f'];
columns = textscan(fid, format, 'Delimiter', ',', 'HeaderLines', 1, ...
    'TextType', 'string', 'ReturnOnError', false);
output = table(columns{1}, columns{4}, columns{6}, columns{7}, columns{16}, ...
    'VariableNames', {'metric', 'delta', 'ci_low', 'ci_high', 'comparison_id'});
clear cleanup
end


function addSection(fig, y, label, color)
annotation(fig, 'textbox', [0.020 y-0.018 0.40 0.035], 'String', label, ...
    'EdgeColor', 'none', 'Color', color, 'FontSize', 7.4, ...
    'FontWeight', 'bold', 'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'middle', 'Margin', 0);
annotation(fig, 'line', [0.020 0.982], [y-0.033 y-0.033], ...
    'Color', [0.72 0.75 0.78], 'LineWidth', 0.7);
end


function addNode(fig, position, label, faceColor, edgeColor, fontSize)
annotation(fig, 'textbox', position, 'String', label, ...
    'BackgroundColor', faceColor, 'EdgeColor', edgeColor, 'LineWidth', 0.8, ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'FontSize', fontSize, 'Color', [0.12 0.15 0.17], 'Margin', 2);
end


function addArrow(fig, startPoint, endPoint, color)
annotation(fig, 'arrow', [startPoint(1) endPoint(1)], [startPoint(2) endPoint(2)], ...
    'Color', color, 'LineWidth', 0.9, 'HeadLength', 5, 'HeadWidth', 5);
end


function point = rightMid(position)
point = [position(1) + position(3), position(2) + position(4)/2];
end


function point = leftMid(position)
point = [position(1), position(2) + position(4)/2];
end


function c = palette()
c.blue = [0.000 0.447 0.698];
c.orange = [0.835 0.369 0.000];
c.green = [0.000 0.541 0.408];
c.red = [0.698 0.227 0.227];
c.purple = [0.420 0.333 0.639];
c.grey = [0.400 0.443 0.486];
c.black = [0.125 0.145 0.165];
c.border = [0.500 0.537 0.569];
c.sourceFill = [0.918 0.953 0.973];
c.robustnessFill = [0.945 0.933 0.973];
c.targetFill = [1.000 0.945 0.910];
c.auditFill = [0.929 0.961 0.945];
end
