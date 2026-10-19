#!/usr/bin/env python3
"""Build publication-quality Milestone 1 PDF Submission Report from Kaggle Version 7 outputs."""
import json
import io
import base64
import shutil
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable,
)
from xml.sax.saxutils import escape

ROOT = Path('/home/youssef/Documents/Programming/f1-podium-prediction')
NB_PATH = Path('/home/youssef/Downloads/acl-ms1 (8).ipynb')
FIG_DIR = ROOT / 'outputs' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load notebook outputs from Kaggle Version 7
nb = json.load(NB_PATH.open())

def extract_png(cell_i, out_i, name):
    data = nb['cells'][cell_i]['outputs'][out_i]['data']['image/png']
    (FIG_DIR / name).write_bytes(base64.b64decode(data))

for c, o, n in [
    (6, 0, 'eda_position_order_raw.png'),
    (6, 1, 'eda_qualifying_coverage_by_year.png'),
    (10, 0, 'eda_podium_rate_by_year.png'),
    (12, 7, 'de_q1_front_row_podium.png'),
    (12, 11, 'de_q2_home_advantage.png'),
    (12, 15, 'de_q3_mechanical_retirement.png'),
    (16, 2, 'feature_target_relationships.png'),
    (24, 3, 'model_pr_auc_comparison.png'),
    (26, 1, 'feature_ablation_test_pr_auc.png'),
    (28, 1, 'xai_permutation_importance.png'),
    (28, 3, 'xai_shap_beeswarm.png'),
    (28, 4, 'xai_lime_row0.png'),
]:
    extract_png(c, o, n)

# 2. Setup typography and styles
styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name='DocTitle',
    fontName='Helvetica-Bold',
    fontSize=22,
    leading=26,
    textColor=colors.HexColor('#0F2537'),
    spaceAfter=4,
))
styles.add(ParagraphStyle(
    name='DocSubTitle',
    fontName='Helvetica',
    fontSize=11,
    leading=14,
    textColor=colors.HexColor('#3D5A70'),
    spaceAfter=8,
))
styles.add(ParagraphStyle(
    name='MetaBlock',
    fontName='Helvetica',
    fontSize=8.5,
    leading=12,
    textColor=colors.HexColor('#2C3E50'),
    spaceAfter=8,
))
styles.add(ParagraphStyle(
    name='SecH1',
    fontName='Helvetica-Bold',
    fontSize=13.5,
    leading=17,
    textColor=colors.HexColor('#133B5C'),
    spaceBefore=8,
    spaceAfter=4,
    keepWithNext=True,
))
styles.add(ParagraphStyle(
    name='SecH2',
    fontName='Helvetica-Bold',
    fontSize=10.5,
    leading=14,
    textColor=colors.HexColor('#0D6680'),
    spaceBefore=6,
    spaceAfter=3,
    keepWithNext=True,
))
styles.add(ParagraphStyle(
    name='BodyRegular',
    fontName='Helvetica',
    fontSize=8.8,
    leading=12.2,
    textColor=colors.HexColor('#22313F'),
    spaceAfter=5,
))
styles.add(ParagraphStyle(
    name='BodyCallout',
    fontName='Helvetica-Oblique',
    fontSize=8.4,
    leading=11.5,
    textColor=colors.HexColor('#1C4E80'),
    spaceAfter=5,
))
styles.add(ParagraphStyle(
    name='Caption',
    fontName='Helvetica',
    fontSize=7.4,
    leading=9.5,
    textColor=colors.HexColor('#4A6274'),
    alignment=TA_CENTER,
    spaceAfter=6,
))
styles.add(ParagraphStyle(
    name='CellText',
    fontName='Helvetica',
    fontSize=7.4,
    leading=9.5,
    textColor=colors.HexColor('#1E2B37'),
))
styles.add(ParagraphStyle(
    name='CellHeader',
    fontName='Helvetica-Bold',
    fontSize=7.4,
    leading=9.5,
    textColor=colors.white,
))

def P(txt, style='BodyRegular'):
    return Paragraph(txt, styles[style])

def fmt_val(v):
    if pd.isna(v):
        return ''
    if isinstance(v, float):
        return f'{v:.4f}'
    return str(v)

def build_table(df, widths=None, max_rows=None, col_names=None):
    d = df.copy()
    if max_rows is not None:
        d = d.head(max_rows)
    cols = col_names if col_names is not None else list(d.columns)
    vals = [[P(escape(str(c)), 'CellHeader') for c in cols]]
    for _, r in d.iterrows():
        vals.append([P(escape(fmt_val(v)), 'CellText') for v in r.tolist()])
    t = Table(vals, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0D6680')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F4F7F9')]),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#CAD7E0')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    return t

def make_chart(name, caption, width=6.6*inch, max_height=2.3*inch):
    path = FIG_DIR / name
    im = Image(str(path))
    ratio = min(width / im.imageWidth, max_height / im.imageHeight)
    im.drawWidth = im.imageWidth * ratio
    im.drawHeight = im.imageHeight * ratio
    return KeepTogether([
        Spacer(1, 2),
        im,
        Spacer(1, 2),
        P(caption, 'Caption'),
        Spacer(1, 4),
    ])

# 3. Assemble document content
story = []

# --- PAGE 1: Header, Executive Summary, Cutoff, Cleaning & Audit ---
story.append(P('Milestone 1 Submission Report', 'DocTitle'))
story.append(P('Formula 1 Podium Prediction | Introduction to Artificial Intelligence', 'DocSubTitle'))
story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#0D6680'), spaceAfter=6, spaceBefore=0))

meta_text = (
    '<b>Run Provenance:</b> Kaggle notebook <code>ACL_MS1</code>, Version 7 of 7 (successful 113.1s run; 110 output files). '
    'Source revision <code>3f5da97</code>; Formula 1 World Championship dataset (1950–2024). All figures and metrics extracted directly from Version 7.<br/>'
    '<b>Team Representative:</b> Youssef Ahmed<br/>'
    '<b>Team Members:</b> Youssef Ahmed (58-4250, T-27); Omar Eldafrawy (58-5371, T-11); Ismail Almoatazbellah (58-4720, T-16); Mohamed Elayadi (43-2918, T-22)<br/>'
    '<b>Kaggle Notebook:</b> <font color="#0D6680">https://www.kaggle.com/code/crimewizard/acl-ms1</font><br/>'
    '<b>GitHub Repository:</b> <font color="#0D6680">https://github.com/CrimeWizard/f1-podium-prediction</font>'
)
story.append(P(meta_text, 'MetaBlock'))

story.append(P('Executive Summary', 'SecH1'))
exec_text = (
    'This project builds an end-to-end, leakage-safe machine learning pipeline to predict whether a Formula 1 driver '
    'will finish on the podium (top-3 finish) based strictly on information available after Saturday qualifying and before Sunday race start. '
    'Using 14 relational tables from the Kaggle Ergast dataset (1950–2024), we conduct extensive data auditing, address historical shared-drive duplicates, '
    'and engineer pre-race championship standings and rolling form features. '
    'A strictly chronological train/validation/test split is enforced (train &le; 2019, validation 2020–2021, held-out test 2022–2024). '
    'Given severe class imbalance (~13–15% podiums), PR-AUC is our primary evaluation metric. '
    'Among four model variants, regularized Random Forest attained the highest unseen test PR-AUC (0.6965), while a shallow Neural Network achieved '
    'the highest validation PR-AUC (0.7433). Both decisively outperformed a defaulted, overfitted tree baseline and linear logistic regression.'
)
story.append(P(exec_text, 'BodyRegular'))

story.append(P('1. Task, Unit of Analysis, and Prediction Cutoff', 'SecH1'))
cutoff_text = (
    'The prediction unit is one driver-race entry uniquely identified by <code>(raceId, driverId)</code>. '
    'The binary prediction target is 1 if the official finishing order satisfies <code>positionOrder &le; 3</code>, and 0 otherwise. '
    '<b>Prediction Cutoff:</b> Information is locked after Saturday qualifying and prior to race start. '
    'Allowed features comprise penalty-adjusted grid slots, qualifying lap times, shifted championship standings, and prior-race historical form. '
    'Current-race results, finishing status, points scored, lap charts, pit stops, and updated standings are strictly excluded to eliminate target leakage.'
)
story.append(P(cutoff_text, 'BodyRegular'))

story.append(P('2. Data Auditing, Cleaning, and Duplicate Handling', 'SecH1'))
cleaning_text = (
    'The relational schema (14 CSV tables) was ingested with immutable raw source integrity. '
    'Placeholder sentinels (literal <code>\\N</code>) were parsed to nulls; dates and qualifying string durations were parsed to semantic datetimes and numeric seconds. '
    '<b>Key Integrity Audit:</b> Primary key audits across all 14 tables revealed zero duplicate and zero null keys. '
    'Foreign key reconciliations across 23 relational links showed zero orphan child records and zero broken references.<br/>'
    '<b>Shared-Drive Resolution:</b> Early Formula 1 regulations (principally 1950s) permitted drivers to swap cars mid-race, producing multiple records per driver-race. '
    'In the complete historical dataset, 176 duplicate rows were identified (131 when excluding the non-championship Indianapolis 500 rounds). '
    'Our policy retains the single record with the best (lowest) <code>positionOrder</code>, capturing the driver’s best verified finishing stint. '
    'The cleaned primary modeling table contains 26,287 unique driver-race records.'
)
story.append(P(cleaning_text, 'BodyRegular'))

# --- PAGE 2: Exploratory Data Analysis (Before & After Cleaning) ---
story.append(PageBreak())
story.append(P('Exploratory Data Analysis (Before & After Cleaning)', 'SecH1'))
eda_text = (
    'EDA was conducted both prior to and following data cleaning to quantify historical coverage drift and verify target stability. '
    'Figure 1 illustrates the raw finishing position distribution across all entries; Figure 2 demonstrates that recorded qualifying session rows '
    'per race were virtually non-existent prior to the 1990s, requiring robust imputation and missingness indicators during modeling. '
    'Figure 3 confirms that after cleaning duplicate rows and enforcing one row per driver-race, the annual podium rate remains tightly bounded around ~15%.'
)
story.append(P(eda_text, 'BodyRegular'))
story.append(make_chart('eda_position_order_raw.png', 'Figure 1: Distribution of raw finishing positions across historical Formula 1 records.', max_height=2.1*inch))
story.append(make_chart('eda_qualifying_coverage_by_year.png', 'Figure 2: Historical qualifying data coverage drift: mean qualifying rows per Grand Prix by season.', max_height=2.1*inch))
story.append(make_chart('eda_podium_rate_by_year.png', 'Figure 3: Annual podium rate post-cleaning: stable ~15% positive prevalence across modern championship eras.', max_height=2.1*inch))

# --- PAGE 3: Data-Engineering Questions (Q1 & Q2) ---
story.append(PageBreak())
story.append(P('3. Data-Engineering Questions (Multi-Hop Joins)', 'SecH1'))

story.append(P('Q1. Front-Row Starts and Podium Conversion (Pre- vs Post-2014 Hybrid Era)', 'SecH2'))
q1_text = (
    'We distinguish official starting grid front rows (grid 1–2, post-penalties) from qualifying session front rows (quali 1–2). '
    'The multi-hop join combines <code>results</code>, <code>races</code>, <code>circuits</code>, and <code>qualifying</code>. '
    'Denominators exclude 1,601 non-starters and account for 14,231 legacy starters lacking electronic qualifying records. '
    'Circuits require &ge; 20 starts per era/basis combination. '
    '<b>Leading conversions:</b> Magny-Cours converts actual grid front rows highest pre-2014 (80.6%; 29/36); Yas Marina converts actual grid highest from 2014 (86.4%; 19/22). '
    'Suzuka converts qualifying front rows highest pre-2014 (76.9%; 20/26); Monaco converts qualifying highest from 2014 (85.0%; 17/20). '
    'Paired Spearman rank correlation across 7 common circuits is weakly negative: &minus;0.180 for actual grid and &minus;0.291 for qualifying, '
    'confirming that hybrid-era turbo-hybrid power units and DRS altered track-specific conversion dynamics.'
)
story.append(P(q1_text, 'BodyRegular'))
story.append(make_chart('de_q1_front_row_podium.png', 'Figure 4: Front-row podium conversion rates by circuit, comparing pre-2014 vs from-2014 across actual grid and qualifying.', max_height=2.2*inch))

story.append(P('Q2. Home-Race Performance Controlling for Starting Grid Position', 'SecH2'))
q2_text = (
    'Home races are determined by matching normalized driver nationality to circuit host country (e.g., British &rarr; UK). '
    'To decouple genuine home advantage from qualifying superiority, drivers are stratified into starting grid bins (1–5, 6–10, 11–15, 16+). '
    'Exclusions: 0 unknown-country records, 130 zero/missing grid starts. '
    '<b>Findings:</b> In Grid 1–5, home drivers achieved a 48.7% podium rate (181/372) vs 45.2% away (2,340/5,180; +3.5%). '
    'In Grid 6–10, home was 8.7% (36/416) vs 11.4% away (582/5,126; &minus;2.7%). '
    'In Grid 11–15, home was 2.7% (11/414) vs 3.0% away (154/5,117). '
    'In Grid 16+, home was 0.3% (2/729) vs 0.7% away (53/7,202). '
    'Because the effect is inconsistent across bins and negligible in the midfield and rear, apparent home advantage is primarily driven by qualifying position rather than race-day psychological effects.'
)
story.append(P(q2_text, 'BodyRegular'))

# Q2 Table exactly matching Version 7 output
q2_data = pd.DataFrame([
    ['1–5', 'Away', 5180, 2340, 0.4517],
    ['1–5', 'Home', 372, 181, 0.4866],
    ['6–10', 'Away', 5126, 582, 0.1135],
    ['6–10', 'Home', 416, 36, 0.0865],
    ['11–15', 'Away', 5117, 154, 0.0301],
    ['11–15', 'Home', 414, 11, 0.0266],
    ['16+', 'Away', 7202, 53, 0.0074],
    ['16+', 'Home', 729, 2, 0.0027],
], columns=['Grid Bin', 'Status', 'Starts', 'Podiums', 'Podium Rate'])
story.append(build_table(q2_data, widths=[1.1*inch, 1.1*inch, 1.2*inch, 1.2*inch, 1.4*inch]))
story.append(Spacer(1, 4))
story.append(make_chart('de_q2_home_advantage.png', 'Figure 5: Home versus away podium conversion stratified by starting grid position bin.', max_height=1.9*inch))

# --- PAGE 4: Q3, Features & Pre-Race Cutoff ---
story.append(PageBreak())
story.append(P('Q3. Mechanical-Retirement Rates by Constructor (2014–2021 vs 2022–2024)', 'SecH2'))
q3_text = (
    'We categorize official status descriptions using semantic keyword matching, separating purely mechanical failures '
    '(power unit, turbo, electrical, hydraulic, gearbox, suspension) from crashes, driver errors, and regulatory disqualifications. '
    'Denominators include all race starters (1,601 non-starters excluded). '
    '<b>Findings:</b> In 2014–2021 (turbo-hybrid era), Caterham exhibited the highest observed failure rate at 32.4% (11/34 starts). '
    'In 2022–2024 (ground-effect era), RB F1 Team had the highest observed rate at 10.4% (5/48 starts). '
    'Across all constructors, mechanical retirement frequencies dropped substantially in 2022–2024, reflecting mature engine freeze regulations.'
)
story.append(P(q3_text, 'BodyRegular'))
story.append(make_chart('de_q3_mechanical_retirement.png', 'Figure 6: Mechanical retirement rates by constructor, comparing 2014–2021 vs 2022–2024 eras.', max_height=2.2*inch))

story.append(P('4. Leakage-Safe Feature Selection and Target Evidence', 'SecH1'))
feat_text = (
    'All features are strictly verified against the pre-race prediction cutoff. '
    'Championship standings are shifted to reflect the driver and constructor standings <i>entering</i> the race weekend, '
    'resetting to zero at each season opener. Prior podium rate is computed over past career starts only.'
)
story.append(P(feat_text, 'BodyRegular'))

feat_desc = pd.DataFrame([
    ['grid', 'Official starting grid slot after steward penalties; strongly predictive of race outcome.'],
    ['quali_position, q1_sec, q2_sec, q3_sec', 'Saturday qualifying positions and lap times in seconds (available post-1994).'],
    ['points_before, position_before', 'Driver championship points and rank entering the race (shifted, reset each season).'],
    ['constructor_points_before, constructor_position_before', 'Constructor championship points and rank entering the race (shifted, reset each season).'],
    ['podium_rate_prev', 'Cumulative rolling driver podium rate strictly over prior career races.'],
], columns=['Feature Group', 'Pre-Race Validity and Description'])
story.append(build_table(feat_desc, widths=[2.2*inch, 4.4*inch]))
story.append(Spacer(1, 4))
story.append(make_chart('feature_target_relationships.png', 'Figure 7: Empirical relationships between pre-race features (grid slot and prior form) and podium probability.', max_height=1.9*inch))

# --- PAGE 5: Splitting, Preprocessing Experiments, & Predictive Modeling ---
story.append(PageBreak())
story.append(P('5. Temporal Dataset Splitting and Preprocessing Experiments', 'SecH1'))
split_text = (
    'To mirror actual deployment and prevent look-ahead bias, data is split chronologically: '
    '<b>Training:</b> 1950–2019 (24,148 rows; 12.58% podium prevalence); '
    '<b>Validation:</b> 2020–2021 (780 rows; 15.00% podium prevalence); '
    '<b>Held-out Test:</b> 2022–2024 (1,359 rows; 15.01% podium prevalence). '
    'All model hyperparameters and decision thresholds are selected strictly on training and validation splits.'
)
story.append(P(split_text, 'BodyRegular'))

prep_comp = pd.DataFrame([
    ['Impute then scale', 0.7009],
    ['Scale then impute', 0.7003],
    ['Median imputation', 0.7009],
    ['Mean imputation', 0.6906],
], columns=['Preprocessing Experiment', 'Validation PR-AUC'])

indy_comp = pd.DataFrame([
    ['exclude_indy_500 (Primary Policy)', 26287, 0.7009, 0.6610],
    ['include_indy_500 (Sensitivity Check)', 26668, 0.7009, 0.6620],
], columns=['Indianapolis 500 Policy', 'Rows', 'Val PR-AUC', 'Test PR-AUC'])

story.append(build_table(prep_comp, widths=[3.3*inch, 3.3*inch]))
story.append(Spacer(1, 4))
story.append(P('<b>Sensitivity Analysis:</b> Excluding Indianapolis 500 rounds (which ran under different technical regulations from 1950–1960) produces identical validation PR-AUC and negligible change on test (<0.001 PR-AUC), proving pipeline stability.', 'BodyRegular'))
story.append(build_table(indy_comp, widths=[2.6*inch, 1.2*inch, 1.4*inch, 1.4*inch]))

story.append(P('6. Predictive Modeling and Evaluation', 'SecH1'))
model_desc_text = (
    'We evaluate four model configurations representing diverse mathematical approaches: '
    '(1) <b>Logistic Regression</b>: Linear log-odds baseline with L2 penalty, providing well-calibrated probabilities under linear assumptions. '
    '(2) <b>Default Random Forest</b>: Unconstrained bagged decision trees showing heavy overfitting. '
    '(3) <b>Regularized Random Forest</b>: Max-depth capped with min-samples-leaf constraints, preventing single-sample leaf memorization. '
    '(4) <b>Shallow Neural Network (FFNN)</b>: Feed-forward network with 1–2 dense hidden layers, ReLU activations, dropout regularization, and sigmoid output. '
    '<b>Primary Metric:</b> PR-AUC is chosen because podiums represent ~13–15% of samples; ROC-AUC is over-optimistic on rare classes. '
    'Thresholds were tuned on validation data (optimal threshold = 0.7584) and frozen before test evaluation.'
)
story.append(P(model_desc_text, 'BodyRegular'))

# Complete metrics table from Version 7
metric_rows = [
    ['Logistic regression', 'Train', 0.5370, 0.8914, 0.5595, 0.8797, 0.5189, 0.6070],
    ['Logistic regression', 'Validation', 0.7009, 0.9181, 0.6877, 0.8987, 0.6397, 0.7436],
    ['Logistic regression', 'Test', 0.6610, 0.9246, 0.6667, 0.8837, 0.5852, 0.7745],
    ['Random forest (default)', 'Train', 0.9754, 0.9976, 0.9535, 0.9878, 0.9155, 0.9947],
    ['Random forest (default)', 'Validation', 0.6859, 0.9128, 0.6900, 0.9090, 0.7054, 0.6752],
    ['Random forest (default)', 'Test', 0.6480, 0.9250, 0.6410, 0.8904, 0.6303, 0.6520],
    ['Random forest (regularized)', 'Train', 0.6277, 0.9242, 0.5945, 0.8967, 0.5874, 0.6017],
    ['Random forest (regularized)', 'Validation', 0.7072, 0.9237, 0.7043, 0.9128, 0.7168, 0.6923],
    ['Random forest (regularized)', 'Test', 0.6965, 0.9335, 0.6420, 0.8933, 0.6468, 0.6373],
    ['Shallow FFNN', 'Train', 0.5799, 0.9044, 0.5725, 0.8879, 0.5501, 0.5968],
    ['Shallow FFNN', 'Validation', 0.7433, 0.9093, 0.7155, 0.9154, 0.7217, 0.7094],
    ['Shallow FFNN', 'Test', 0.6624, 0.9264, 0.6682, 0.8926, 0.6229, 0.7206],
]
df_metrics = pd.DataFrame(metric_rows, columns=['Model', 'Split', 'PR-AUC', 'ROC-AUC', 'F1', 'Accuracy', 'Precision', 'Recall'])
story.append(build_table(df_metrics, widths=[1.6*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.65*inch, 0.7*inch, 0.7*inch, 0.65*inch]))
story.append(Spacer(1, 4))
story.append(make_chart('model_pr_auc_comparison.png', 'Figure 8: Comparative PR-AUC across models and chronological data splits.', max_height=1.85*inch))

# --- PAGE 6: Feature Ablation & Explainability (XAI) ---
story.append(PageBreak())
story.append(P('7. Feature-Group Ablation Experiment', 'SecH1'))
abl_text = (
    'We systematically ablate one feature group at a time from the baseline model to quantify each component’s marginal contribution. '
    'Removing <b>grid</b> triggers the steepest drop on unseen test data (&Delta; &minus;0.0249 PR-AUC), confirming starting slot as the paramount predictor. '
    'Qualifying times (&Delta; &minus;0.0101) and historical form (&Delta; &minus;0.0056) provide necessary secondary refinements.'
)
story.append(P(abl_text, 'BodyRegular'))

abl_data = pd.DataFrame([
    ['All Features (Baseline)', 0.7009, 0.6610],
    ['Drop Starting Grid', 0.6749, 0.6361],
    ['Drop Qualifying Session Times/Rank', 0.6909, 0.6509],
    ['Drop Championship Standings', 0.7012, 0.6565],
    ['Drop Rolling Driver Form', 0.7071, 0.6554],
], columns=['Feature Group Removed', 'Val PR-AUC', 'Held-Out Test PR-AUC'])
story.append(build_table(abl_data, widths=[3.0*inch, 1.8*inch, 1.8*inch]))
story.append(Spacer(1, 4))
story.append(make_chart('feature_ablation_test_pr_auc.png', 'Figure 9: Impact of feature group ablation on held-out test PR-AUC.', max_height=1.8*inch))

story.append(P('8. Model Explainability (XAI) and Detailed Interpretation', 'SecH1'))
xai_intro = (
    'To inspect model decision rules globally and locally, we compute permutation feature importance, SHAP values, and LIME explanations. '
    '<b>Important Causal Disclaimer:</b> These explanations represent statistical predictive associations within the model, <i>not causal effects</i>. '
    'A high feature importance confirms model reliance, not that changing a feature will directly alter race outcomes.<br/>'
    '<b>Global Permutation Importance (Figure 10):</b> Permuting starting grid degrades validation PR-AUC by ~0.15, confirming it as the primary anchor. '
    'Driver rolling podium form is second (~0.11), followed by constructor standings rank (~0.05).<br/>'
    '<b>Global SHAP Beeswarm (Figure 11):</b> Lower numerical grid values (front-of-grid starts) produce large positive SHAP contributions to podium log-odds, '
    'whereas starts outside the top 10 yield heavy negative penalties. High prior podium rate similarly shifts probabilities upward.<br/>'
    '<b>Local LIME Explanation (Figure 12):</b> For an exemplary validation prediction (Lewis Hamilton, predicted podium probability 85.5%), '
    'the local explanation demonstrates that starting near the front (<code>num__grid &le; &minus;0.85</code>) added +0.36 to podium probability, '
    'and high historical podium rate (<code>num__podium_rate_prev > 0.33</code>) added +0.16, comfortably surpassing the 0.7584 classification threshold.'
)
story.append(P(xai_intro, 'BodyRegular'))
story.append(make_chart('xai_permutation_importance.png', 'Figure 10: Permutation feature importance scored by PR-AUC drop on validation data.', max_height=1.8*inch))

# --- PAGE 7: SHAP, LIME, Inference Demo, Limitations & References ---
story.append(PageBreak())
story.append(P('Model Explainability Figures (Continued)', 'SecH1'))
story.append(make_chart('xai_shap_beeswarm.png', 'Figure 11: SHAP summary beeswarm plot illustrating feature value distributions and impact on model output.', max_height=2.3*inch))
story.append(make_chart('xai_lime_row0.png', 'Figure 12: Local LIME explanation for an individual high-confidence validation prediction (predicted probability: 85.5%).', max_height=2.3*inch))

story.append(P('9. Inference Function Demonstration', 'SecH1'))
inf_text = (
    'The production inference function (<code>src/inference.py</code>) processes single driver-race dictionary inputs, '
    'applying imputation and scaling pipelines consistently without data leakage. '
    'Tested on a complete validation record (Hamilton, grid 2), it predicted a <b>85.46%</b> podium probability (Class 1 at threshold 0.7584). '
    'Tested on a partially missing record (midfield entry lacking session qualifying lap times), the pipeline handled missing values via median imputation '
    'and returned a <b>40.55%</b> podium probability (Class 0), demonstrating production stability.'
)
story.append(P(inf_text, 'BodyRegular'))

story.append(P('10. Limitations, Technical Constraints, and References', 'SecH1'))
limit_text = (
    '<b>Key Limitations:</b> (1) <i>Qualifying coverage drift:</i> Official lap times are absent prior to the mid-1990s, requiring imputation for early seasons. '
    '(2) <i>Constructor identity:</i> Frequent team acquisitions/rebrandings complicate historical constructor lineage. '
    '(3) <i>Sprint race formats:</i> Modern sprint weekends (introduced 2021) alter weekend tire allocation and starting grids, which are not separately modeled. '
    '(4) <i>Starter flag ambiguity:</i> Ergast lacks an independent starter flag; mechanical retirements before lights out are inferred from status labels.<br/>'
    '<b>References:</b> Course Milestone 1 Companion Guide; Formula 1 for Newcomers Guide; Kaggle Formula 1 World Championship Dataset (Ergast); project repository and public notebook.'
)
story.append(P(limit_text, 'BodyRegular'))

# 4. Build document with running header and footer
def add_decorations(canvas, doc):
    canvas.saveState()
    w, h = A4
    # Running header
    canvas.setFont('Helvetica', 7.5)
    canvas.setFillColor(colors.HexColor('#6B8294'))
    canvas.drawString(0.55*inch, h - 0.4*inch, 'Formula 1 Podium Prediction — Milestone 1 Submission Report')
    canvas.drawRightString(w - 0.55*inch, h - 0.4*inch, 'Introduction to AI')
    canvas.setStrokeColor(colors.HexColor('#D3DDE5'))
    canvas.setLineWidth(0.5)
    canvas.line(0.55*inch, h - 0.45*inch, w - 0.55*inch, h - 0.45*inch)
    
    # Running footer
    canvas.line(0.55*inch, 0.48*inch, w - 0.55*inch, 0.48*inch)
    canvas.drawString(0.55*inch, 0.34*inch, 'Kaggle Notebook: crimewizard/acl-ms1 | Version 7 Run')
    canvas.drawRightString(w - 0.55*inch, 0.34*inch, f'Page {doc.page}')
    canvas.restoreState()

pdf_target = ROOT / 'docs' / 'MILESTONE1_SUBMISSION_REPORT.pdf'
doc = SimpleDocTemplate(
    str(pdf_target),
    pagesize=A4,
    leftMargin=0.55*inch,
    rightMargin=0.55*inch,
    topMargin=0.58*inch,
    bottomMargin=0.58*inch,
    title='Milestone 1 Submission Report - F1 Podium Prediction',
    author='Youssef Ahmed and Team',
)

doc.build(story, onFirstPage=add_decorations, onLaterPages=add_decorations)
print(f"Built PDF with {doc.page} pages at {pdf_target}")

# Synchronize submission_package copy
sub_target = ROOT / 'submission_package' / 'MILESTONE1_SUBMISSION_REPORT.pdf'
shutil.copy2(pdf_target, sub_target)
print(f"Copied to {sub_target}")
