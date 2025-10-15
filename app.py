# app2.py
"""
Telco Customer Churn Dashboard (Final)
Author signature displayed on page: 王三出品

Requirements (example):
pip install dash dash-bootstrap-components pandas numpy scikit-learn plotly joblib
"""

import os
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
import joblib

# -------------------------
# Config & Data path
# -------------------------
DATA_PATH = os.path.join("WA_Fn-UseC_-Telco-Customer-Churn.csv")
# If you store CSV in project root, uncomment:
# DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# -------------------------
# Load & clean data
# -------------------------
df_raw = pd.read_csv(DATA_PATH)
df_raw.columns = df_raw.columns.str.strip()

# Convert TotalCharges to numeric safely (avoid chained assignment warning)
df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')
df_raw['TotalCharges'] = df_raw['TotalCharges'].fillna(df_raw['TotalCharges'].median())

# 【修改1】数据清洗阶段统一处理Churn为数值型，避免后续重复映射
df_raw['Churn'] = df_raw['Churn'].map({'Yes': 1, 'No': 0})

# Keep a copy for display & scoring
df_display = df_raw.copy()

# -------------------------
# Prepare X, y and preprocessing pipeline
# -------------------------
# Drop identifier if present
if 'customerID' in df_raw.columns:
    X_all = df_raw.drop(columns=['customerID', 'Churn'])
else:
    X_all = df_raw.drop(columns=['Churn'])

# 【优化】直接使用已数值化的Churn列，无需重复map
y_all = df_raw['Churn']

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [c for c in X_all.columns if c not in numeric_features]

# Preprocessors
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# -------------------------
# Build and train models (at startup)
# -------------------------
# Split for evaluation (stratify to preserve churn ratio)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.25, random_state=42, stratify=y_all)

# Logistic Regression pipeline
pipe_lr = Pipeline([
    ('pre', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])

# Random Forest pipeline
pipe_rf = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
])

# Fit
pipe_lr.fit(X_train, y_train)
pipe_rf.fit(X_train, y_train)

# Evaluate on test
y_proba_lr = pipe_lr.predict_proba(X_test)[:, 1]
y_proba_rf = pipe_rf.predict_proba(X_test)[:, 1]

auc_lr = roc_auc_score(y_test, y_proba_lr)
auc_rf = roc_auc_score(y_test, y_proba_rf)

# Confusion matrices (using threshold 0.5)
y_pred_lr = (y_proba_lr >= 0.5).astype(int)
y_pred_rf = (y_proba_rf >= 0.5).astype(int)
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_rf = confusion_matrix(y_test, y_pred_rf)

# Feature names after preprocessing (from RF encoder)
cat_encoder = pipe_rf.named_steps['pre'].named_transformers_['cat']
cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
feature_names = numeric_features + cat_feature_names
rf_importances = pipe_rf.named_steps['clf'].feature_importances_
feat_imp_series = pd.Series(rf_importances, index=feature_names).sort_values(ascending=False)

# Logistic coefficients
lr_coefs = pipe_lr.named_steps['clf'].coef_[0]
lr_coef_series = pd.Series(lr_coefs, index=feature_names).sort_values(key=lambda s: np.abs(s), ascending=False)

# Score full dataset with default model (choose the better AUC by default)
default_model_name = 'Logistic Regression' if auc_lr >= auc_rf else 'Random Forest'
default_pipe = pipe_lr if auc_lr >= auc_rf else pipe_rf

# Score full df_display
X_full = X_all.copy()
probs_full = default_pipe.predict_proba(X_full)[:, 1]
df_display = df_display.copy()
df_display['churn_prob'] = probs_full
df_display['predicted_churn'] = np.where(df_display['churn_prob'] >= 0.5, 'High risk', 'Low risk')

# Overall stats
overall_churn_rate = y_all.mean()
contract_churn_rate = df_raw.groupby('Contract')['Churn'].mean().sort_values(ascending=False)
internet_churn_rate = df_raw.groupby('InternetService')['Churn'].mean().sort_values(ascending=False)
payment_churn_rate = df_raw.groupby('PaymentMethod')['Churn'].mean().sort_values(ascending=False)
tech_churn_rate = df_raw.groupby('TechSupport')['Churn'].mean().sort_values(ascending=False)

# Save models (optional)
os.makedirs("models", exist_ok=True)
joblib.dump(pipe_lr, "models/pipe_lr.pkl")
joblib.dump(pipe_rf, "models/pipe_rf.pkl")
joblib.dump(preprocessor, "models/preprocessor.pkl")
joblib.dump(scaler := None, "models/placeholder_scaler.pkl")  # placeholder if needed

# -------------------------
# Plotly figures (initial)
# -------------------------
# Colors
color_keep = {'0': '#3A84FF', '1': '#FF6B6B'}  # 0=No churn,1=Yes churn

# Churn distribution
churn_counts = df_raw['Churn'].value_counts().sort_index()  # 0/1
fig_churn = px.bar(
    x=['No', 'Yes'],
    # 【修改2】用iloc按位置访问，解决FutureWarning（替代get(0,0)）
    y=[churn_counts.iloc[0], churn_counts.iloc[1]],
    title="客户流失分布 (Churn)",
    labels={'x': 'Churn', 'y': '客户数量'},
    color=['No', 'Yes'],
    color_discrete_map={'No': color_keep['0'], 'Yes': color_keep['1']}
)
fig_churn.update_traces(marker_line_width=0)

# Contract vs churn grouped bar
fig_contract = px.histogram(df_raw, x='Contract', color='Churn', barmode='group',
                            title='合同类型与 Churn 关系', color_discrete_map={0: color_keep['0'], 1: color_keep['1']})

# Correlation heatmap (numeric)
# 【修改3】直接用已数值化的Churn列计算，无需临时转换（解决ValueError）
corr = df_raw[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']].corr()
fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title='数值特征相关性热力图')

# Distribution with churn color: tenure
fig_tenure = px.histogram(df_raw, x='tenure', color='Churn', nbins=30, title='tenure 分布（按 Churn 分组）',
                          color_discrete_map={0: color_keep['0'], 1: color_keep['1']}, marginal='box')

# MonthlyCharges distribution
fig_month = px.histogram(df_raw, x='MonthlyCharges', color='Churn', nbins=30, title='MonthlyCharges 分布（按 Churn 分组）',
                         color_discrete_map={0: color_keep['0'], 1: color_keep['1']}, marginal='box')

# Boxplots for tenure and MonthlyCharges
fig_box_tenure = px.box(df_raw, x='Churn', y='tenure', title='tenure vs Churn（箱线图）',
                        color_discrete_map={0: color_keep['0'], 1: color_keep['1']})
fig_box_month = px.box(df_raw, x='Churn', y='MonthlyCharges', title='MonthlyCharges vs Churn（箱线图）',
                       color_discrete_map={0: color_keep['0'], 1: color_keep['1']})

# Feature importance top 15
feat_imp_df = feat_imp_series.reset_index()
feat_imp_df.columns = ['feature', 'importance']
fig_feat_imp = px.bar(feat_imp_df.head(15).sort_values('importance'), x='importance', y='feature', orientation='h',
                      title='Random Forest Top 15 特征重要性')

# ROC curves for both models (on test set)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, mode='lines', name=f'Logistic AUC={auc_lr:.3f}'))
fig_roc.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode='lines', name=f'RandomForest AUC={auc_rf:.3f}'))
fig_roc.add_shape(type='line', x0=0, x1=1, y0=0, y1=1, line=dict(dash='dash', color='gray'))
fig_roc.update_layout(title='模型 ROC 曲线', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')

# Confusion matrices plotted as heatmaps
fig_cm_lr = px.imshow(cm_lr, text_auto=True, color_continuous_scale='Blues', title='Confusion Matrix - Logistic (threshold=0.5)')
fig_cm_lr.update_xaxes(title_text="Predicted")
fig_cm_lr.update_yaxes(title_text="Actual")
fig_cm_rf = px.imshow(cm_rf, text_auto=True, color_continuous_scale='Blues', title='Confusion Matrix - RandomForest (threshold=0.5)')
fig_cm_rf.update_xaxes(title_text="Predicted")
fig_cm_rf.update_yaxes(title_text="Actual")

# -------------------------
# Dash App layout
# -------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server
app.title = "Telco Customer Churn 仪表盘 — 王三出品"

# Controls initial values
initial_model = 'lr' if auc_lr >= auc_rf else 'rf'
initial_threshold = 0.5

app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        # 顶部标题栏，保持不变，但在手机上标题可能占 8 份，右侧信息占 4 份，仍能适配。
        dbc.Col(html.H3("📊 Telco Customer Churn 仪表盘 — 王三出品"), width=8), 
        dbc.Col(html.Div([
            html.Div(f"Default Model: {default_model_name}", style={'fontSize': 12})
        ], style={'textAlign': 'right'}), width=4)
    ], align='center', className='my-2'),

    # top KPI cards (响应式修改: 手机上两列显示)
    dbc.Row([
        # 原 width=3 -> 现 xs=6, md=3
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("总体样本数"),
            html.H4(f"{len(df_raw):,}")
        ])), xs=6, md=3), 
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("总体流失率"),
            html.H4(f"{overall_churn_rate:.2%}")
        ])), xs=6, md=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Logistic AUC"),
            html.H4(f"{auc_lr:.3f}")
        ])), xs=6, md=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("RandomForest AUC"),
            html.H4(f"{auc_rf:.3f}")
        ])), xs=6, md=3)
    ], className='mb-3'),

    # Controls (响应式修改: 手机上堆叠显示)
    dbc.Row([
        # 原 width=6 -> 现 xs=12, md=6
        dbc.Col([
            html.Label("选择用于评分的模型"),
            dcc.RadioItems(id='model-select', options=[{'label': 'Logistic Regression', 'value': 'lr'},
                                                      {'label': 'Random Forest', 'value': 'rf'}],
                           value=initial_model, inline=True),
        ], xs=12, md=6), 
        dbc.Col([
            html.Label("高风险阈值 (churn_prob)"),
            dcc.Slider(id='prob-threshold', min=0.1, max=0.9, step=0.01, value=initial_threshold,
                       marks={0.1: '0.1', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 0.9: '0.9'})
        ], xs=12, md=6)
    ], className='mb-3'),

    # Row: churn distribution & contract (响应式修改: 手机上堆叠显示)
    dbc.Row([
        # 原 width=6 -> 现 xs=12, md=6
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("客户流失分布"),
            dcc.Graph(figure=fig_churn),
            html.P("说明：展示留存（No）与流失（Yes）客户数量对比，帮助快速判断样本平衡。", style={'fontSize': 12})
        ])), xs=12, md=6), 
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("合同类型 vs 流失"),
            dcc.Graph(figure=fig_contract),
            html.P("说明：观察不同合同类型下的流失分布，通常 Month-to-month（按月）流失率最高。", style={'fontSize': 12})
        ])), xs=12, md=6)
    ]),

    # Row: heatmap & distribution (响应式修改: 手机上堆叠显示)
    dbc.Row([
        # 原 width=6 -> 现 xs=12, md=6
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("数值特征相关性热力图"),
            dcc.Graph(figure=fig_corr),
            html.P("说明：tenure 与 Churn 呈显著负相关，MonthlyCharges 与 Churn 有弱正相关。", style={'fontSize': 12})
        ])), xs=12, md=6), 

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("数值分布（tenure / MonthlyCharges）"),
            dcc.Tabs([
                dcc.Tab(label='tenure 分布', children=[dcc.Graph(figure=fig_tenure)]),
                dcc.Tab(label='MonthlyCharges 分布', children=[dcc.Graph(figure=fig_month)])
            ]),
            html.P("说明：流失客户通常集中在任期短（新客户）和月费用较高的分布区间。", style={'fontSize': 12})
        ])), xs=12, md=6)
    ]),

    # Row: boxplots & feature importance (响应式修改: 手机上堆叠显示)
    dbc.Row([
        # 原 width=6 -> 现 xs=12, md=6
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("箱线图（tenure / MonthlyCharges）"),
            dcc.Graph(figure=fig_box_tenure),
            dcc.Graph(figure=fig_box_month),
            html.P("说明：箱线图显示了流失与非流失在数值特征上的分布差异及异常值。", style={'fontSize': 12})
        ])), xs=12, md=6), 

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("特征重要性（Random Forest）"),
            dcc.Graph(figure=fig_feat_imp),
            html.P("说明：模型认为 tenure、Contract 及 MonthlyCharges 等是最重要的预测因子。", style={'fontSize': 12})
        ])), xs=12, md=6)
    ]),

    # Row: model performance (响应式修改: 手机上堆叠显示)
    dbc.Row([
        # 原 width=6 -> 现 xs=12, md=6
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("模型性能 - ROC 曲线"),
            dcc.Graph(figure=fig_roc),
            html.P("说明：ROC 曲线展示模型的区分能力，AUC 值越大表示模型越好（接近1）。", style={'fontSize': 12})
        ])), xs=12, md=6), 

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("混淆矩阵对比（阈值=0.5）"),
            dcc.Tabs([
                dcc.Tab(label='Logistic', children=[dcc.Graph(figure=fig_cm_lr)]),
                dcc.Tab(label='RandomForest', children=[dcc.Graph(figure=fig_cm_rf)])
            ]),
            html.P("说明：混淆矩阵用于查看真阳性/假阳性等分类结果详细分布。", style={'fontSize': 12})
        ])), xs=12, md=6)
    ]),

    # High-risk table and export (保持 width=12)
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("高风险客户（基于当前模型与阈值）"),
            html.Div(id='highrisk-stats', style={'marginBottom': 8}),
            dcc.Loading(dash_table.DataTable(
                id='highrisk-table',
                columns=[{"name": c, "id": c} for c in ['customerID', 'gender', 'tenure', 'Contract', 'MonthlyCharges', 'churn_prob']],
                data=[],
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center'},
            )),
            html.Br(),
            dbc.Button("导出高风险客户 CSV", id='btn-export', color='primary'),
            dcc.Download(id='download-highrisk')
        ])), width=12)
    ], className='my-3'),

    # Insights & business suggestions (保持 width=12)
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("核心洞察与业务建议"),
            dcc.Markdown(
                f"""
**核心洞察（自动生成）** - 总体流失率：**{overall_churn_rate:.2%}**。  
- 主要影响因子（模型重要性）：{', '.join(feat_imp_df['feature'].head(5).tolist())}。  
- 模型表现：Logistic AUC={auc_lr:.3f}，RandomForest AUC={auc_rf:.3f}。  

**推荐的商业行动（示例）** 1. 优先对 Month-to-month 用户做优惠/续约激励；  
2. 对高月费用户提供个性化客服或账单优化；  
3. 对没有 TechSupport 的用户做主动关怀；  
4. 按价值排序（LTV × churn_prob）优先挽留高价值高风险客户。  
                """, style={'fontSize': 14}
            )
        ])), width=12)
    ]),

    html.Hr(),
    html.Footer("© 2025 王三出品 | Telco Customer Churn Dashboard", style={'textAlign': 'center', 'color': 'gray', 'padding': '10px'})
], style={'maxWidth': '1200px', 'margin': '0 auto'})

# -------------------------
# Callbacks
# -------------------------
@app.callback(
    Output('highrisk-table', 'data'),
    Output('highrisk-stats', 'children'),
    Input('model-select', 'value'),
    Input('prob-threshold', 'value')
)
def update_highrisk_table(model_select, prob_threshold):
    # choose model pipeline
    sel_pipe = pipe_lr if model_select == 'lr' else pipe_rf
    # score full dataset
    X_full_local = X_all.copy()
    probs = sel_pipe.predict_proba(X_full_local)[:, 1]
    df_full_scored = df_display.copy()
    df_full_scored['churn_prob'] = probs
    # prepare DataTable fields: need customerID existence
    if 'customerID' in df_raw.columns:
        id_col = df_raw['customerID']
        df_full_scored['customerID'] = id_col
    else:
        # create an index id
        df_full_scored['customerID'] = df_full_scored.index.astype(str)

    highrisk = df_full_scored[df_full_scored['churn_prob'] >= prob_threshold].copy()
    highrisk_count = len(highrisk)
    stats = html.Div([
        html.P(f"使用模型：{'Logistic Regression' if model_select=='lr' else 'Random Forest'}； 阈值：{prob_threshold:.2f}"),
        html.P(f"当前高风险客户数：{highrisk_count}（占总用户 {highrisk_count/len(df_full_scored):.2%}）")
    ])

    # select table columns (ensure they exist)
    cols = ['customerID', 'gender', 'tenure', 'Contract', 'MonthlyCharges', 'churn_prob']
    # some columns may be missing in X_all if one-hot used - they exist in original df_raw
    table_df = highrisk[cols].copy() if set(cols).issubset(highrisk.columns) else highrisk.reset_index()[['index', 'churn_prob']].rename(columns={'index':'customerID'})

    # round churn_prob
    if 'churn_prob' in table_df.columns:
        table_df['churn_prob'] = table_df['churn_prob'].round(3)

    return table_df.to_dict('records'), stats


@app.callback(
    Output('download-highrisk', 'data'),
    Input('btn-export', 'n_clicks'),
    State('model-select', 'value'),
    State('prob-threshold', 'value'),
    prevent_initial_call=True
)
def export_highrisk(n_clicks, model_select, prob_threshold):
    sel_pipe = pipe_lr if model_select == 'lr' else pipe_rf
    X_full_local = X_all.copy()
    probs = sel_pipe.predict_proba(X_full_local)[:, 1]
    df_full_scored = df_display.copy()
    df_full_scored['churn_prob'] = probs
    if 'customerID' in df_raw.columns:
        df_full_scored['customerID'] = df_raw['customerID']
    else:
        df_full_scored['customerID'] = df_full_scored.index.astype(str)
    highrisk = df_full_scored[df_full_scored['churn_prob'] >= prob_threshold].copy()
    export_cols = ['customerID', 'gender', 'tenure', 'Contract', 'MonthlyCharges', 'churn_prob']
    export_df = highrisk[export_cols] if set(export_cols).issubset(highrisk.columns) else highrisk
    return dcc.send_data_frame(export_df.sort_values('churn_prob', ascending=False).to_csv,
                               filename=f"highrisk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               index=False)

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    # Dash >=3.0 uses app.run
    app.run(debug=True, host="0.0.0.0", port=8050)

