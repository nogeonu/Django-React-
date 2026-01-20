"""
pCR Prediction Service for LIS
Integrates AI model for breast cancer pCR prediction
"""

import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import xgboost as xgb
import lightgbm as lgb
from scipy.stats.mstats import winsorize
import os
import json
import shap
import warnings
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')


def set_korean_font():
    import matplotlib.font_manager as fm
    k_fonts = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'Dotum', 'Gulim']
    system_fonts = [f.name for f in fm.fontManager.ttflist]
    
    selected_font = 'sans-serif'
    for f in k_fonts:
        if f in system_fonts:
            selected_font = f
            break
            
    plt.rcParams['font.family'] = selected_font
    plt.rcParams['axes.unicode_minus'] = False


set_korean_font()


class PathwayAttention(nn.Module):
    def __init__(self, input_dim):
        super(PathwayAttention, self).__init__()
        self.attention = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Tanh(), nn.Linear(input_dim, 1))
    
    def forward(self, x):
        weights = torch.softmax(self.attention(x), dim=1)
        return x * weights


class HierarchicalModel(nn.Module):
    def __init__(self, pathway_sizes, hidden_dim=32, dropout=0.3):
        super(HierarchicalModel, self).__init__()
        self.pathway_sizes = pathway_sizes
        self.pathway_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(size, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout)) 
            for size in pathway_sizes
        ])
        self.attention = PathwayAttention(hidden_dim)
        self.integration = nn.Sequential(
            nn.Linear(len(pathway_sizes) * hidden_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), 
            nn.Dropout(dropout), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        outputs = []
        start = 0
        for i, size in enumerate(self.pathway_sizes):
            out = self.pathway_layers[i](x[:, start:start+size])
            out = self.attention(out)
            outputs.append(out)
            start += size
        combined = torch.cat(outputs, dim=1)
        return self.integration(combined)


class PCRPredictor:
    def __init__(self):
        self.genes_27 = [
            'CXCL13', 'CD8A', 'CCR7', 'C1QA', 'LY9', 'CXCL10', 'CXCL9', 'STAT1',
            'CCND1', 'MKI67', 'TOP2A', 'BRCA1', 'RAD51', 'PRKDC', 'POLD3', 'POLB', 'LIG1',
            'ERBB2', 'ESR1', 'PGR', 'ARAF', 'PIK3CA', 'AKT1', 'MTOR', 'TP53', 'PTEN', 'MYC'
        ]
        self.pathways = {
            '면역 (Immune)': ['CXCL13', 'CD8A', 'CCR7', 'C1QA', 'LY9', 'CXCL10', 'CXCL9', 'STAT1'],
            '세포증식 (Proliferation)': ['CCND1', 'MKI67', 'TOP2A'],
            'DNA 복구 (DNA Repair)': ['BRCA1', 'RAD51', 'PRKDC', 'POLD3', 'POLB', 'LIG1'],
            'HER2 수용체': ['ERBB2'],
            '호르몬 수용체 (ER/PR)': ['ESR1', 'PGR'],
            '신호전달 (AKT/mTOR)': ['ARAF', 'PIK3CA', 'AKT1', 'MTOR', 'TP53', 'PTEN', 'MYC']
        }
        self.model_dir = os.path.join(os.path.dirname(__file__), 'models', 'saved')
        self.load_models()
    
    def load_models(self):
        self.scaler = joblib.load(os.path.join(self.model_dir, 'final_ensemble_scaler.pkl'))
        
        # XGBoost 모델 로드
        self.xgb_model = xgb.XGBClassifier()
        model_path = os.path.join(self.model_dir, 'final_xgb_model.json')
        self.xgb_model.load_model(model_path)
        
        # base_score 형식 오류 수정 (SHAP 호환성)
        try:
            booster = self.xgb_model.get_booster()
            config = booster.save_config()
            config_dict = json.loads(config)
            
            # base_score 수정
            if 'learner' in config_dict and 'learner_model_param' in config_dict['learner']:
                base_score_str = config_dict['learner']['learner_model_param'].get('base_score', '0.5')
                # '[5E-1]' 형식인 경우 처리
                if isinstance(base_score_str, str):
                    if base_score_str.startswith('[') and base_score_str.endswith(']'):
                        base_score_str = base_score_str.strip('[]')
                    try:
                        base_score = float(base_score_str)
                        config_dict['learner']['learner_model_param']['base_score'] = str(base_score)
                        booster.load_config(json.dumps(config_dict))
                    except (ValueError, TypeError):
                        # 변환 실패 시 기본값 사용
                        config_dict['learner']['learner_model_param']['base_score'] = '0.5'
                        booster.load_config(json.dumps(config_dict))
        except Exception as e:
            warnings.warn(f"XGBoost 모델 설정 수정 실패 (계속 진행): {e}")
        
        self.lgb_model = joblib.load(os.path.join(self.model_dir, 'final_lgb_model.pkl'))
        
        pathway_sizes = [8, 3, 6, 1, 2, 7]
        self.hier_model = HierarchicalModel(pathway_sizes)
        self.hier_model.load_state_dict(torch.load(os.path.join(self.model_dir, 'final_hier_model.pth')))
        self.hier_model.eval()
    
    def preprocess(self, gene_values):
        """gene_values: dict with gene names as keys"""
        X = np.array([[gene_values.get(g, 0) for g in self.genes_27]])
        
        for i in range(X.shape[1]):
            X[:, i] = winsorize(X[:, i], limits=(0.01, 0.01))
        
        skewed = ['TP53', 'POLD3', 'PGR', 'PIK3CA', 'MTOR']
        for gene in skewed:
            idx = self.genes_27.index(gene)
            X[:, idx] = np.log1p(X[:, idx] - X[:, idx].min() + 1)
        
        return self.scaler.transform(X)
    
    def predict(self, gene_values):
        X_scaled = self.preprocess(gene_values)
        xgb_prob = self.xgb_model.predict_proba(X_scaled)[:, 1]
        lgb_prob = self.lgb_model.predict_proba(X_scaled)[:, 1]
        X_tensor = torch.FloatTensor(X_scaled)
        with torch.no_grad():
            hier_prob = self.hier_model(X_tensor).numpy().flatten()
        
        prob = float((xgb_prob[0] + lgb_prob[0] + hier_prob[0]) / 3)
        return prob
    
    def get_shap_values(self, gene_values):
        X_scaled = self.preprocess(gene_values)
        try:
            explainer = shap.TreeExplainer(self.xgb_model)
            return explainer.shap_values(X_scaled)[0]
        except Exception as e:
            # SHAP 오류 시 기본값 반환 (예측은 계속 진행)
            warnings.warn(f"SHAP 값 계산 실패, 기본값 사용: {e}")
            return np.zeros(len(self.genes_27))
    
    def generate_report_image(self, gene_values, patient_info):
        """Generate clinical report as base64 image"""
        prob = self.predict(gene_values)
        shap_vals = self.get_shap_values(gene_values)
        
        fig = plt.figure(figsize=(24, 16), facecolor='#F5F7FA')
        gs = GridSpec(3, 3, figure=fig, width_ratios=[1.1, 1, 0.9], height_ratios=[0.28, 0.36, 0.36], wspace=0.2, hspace=0.32)

        # Header
        ax_header = fig.add_subplot(gs[0, :])
        ax_header.axis('off')
        ax_header.add_patch(patches.Rectangle((0, 0.1), 0.02, 0.8, color='#3498DB', transform=ax_header.transAxes))
        ax_header.text(0.04, 0.80, f"Patient Clinical Report: PATIENT_{patient_info.get('patient_id', 'Unknown')}", 
                      fontsize=30, weight='bold', color='#2C3E50')
        
        info_fs, label_c, val_c = 16, '#7F8C8D', '#2C3E50'
        col1, col2, col3 = 0.04, 0.35, 0.65
        
        ax_header.text(col1, 0.60, f"• 환자 ID:    ", fontsize=info_fs, color=label_c)
        ax_header.text(col1 + 0.08, 0.60, f"{patient_info.get('patient_id', 'N/A')}", fontsize=info_fs, color=val_c, weight='bold')
        ax_header.text(col2, 0.60, f"• 환자 성명:    ", fontsize=info_fs, color=label_c)
        ax_header.text(col2 + 0.09, 0.60, patient_info.get('name', '정보없음 (N/A)'), fontsize=info_fs, color=val_c, weight='bold')
        ax_header.text(col3, 0.60, f"• 나이:    ", fontsize=info_fs, color=label_c)
        ax_header.text(col3 + 0.05, 0.60, f"{patient_info.get('age', 'N/A')}세", fontsize=info_fs, color=val_c, weight='bold')
        
        ax_header.text(col1, 0.42, f"• 성별:    ", fontsize=info_fs, color=label_c)
        ax_header.text(col1 + 0.08, 0.42, patient_info.get('gender', 'N/A'), fontsize=info_fs, color=val_c, weight='bold')
        ax_header.text(col2, 0.42, f"• 아류형 (Subtype):    ", fontsize=info_fs, color=label_c)
        ax_header.text(col2 + 0.15, 0.42, "HER2-Enriched (HR-/HER2+)", fontsize=info_fs, color=val_c, weight='bold')
        ax_header.text(col3, 0.42, f"• 검사일:    ", fontsize=info_fs, color=label_c)
        ax_header.text(col3 + 0.06, 0.42, patient_info.get('test_date', 'N/A'), fontsize=info_fs, color=val_c, weight='bold')
        
        ax_header.text(0.04, 0.20, "AI 기반 정밀 분석 결과 요약 리포트입니다. 치료 방향 결정 시 임상적 소견과 병행하여 검토하십시오.", 
                     fontsize=12, style='italic', color='#95A5A6')

        # Left Panel (Top 10 Genes)
        ax_feat = fig.add_subplot(gs[1:, 0])
        gene_to_path = {g: p.split(' (')[0] for p, genes in self.pathways.items() for g in genes}
        feat_df = pd.DataFrame({'gene': self.genes_27, 'shap': shap_vals})
        feat_df['abs_shap'] = feat_df['shap'].abs()
        top_10 = feat_df.sort_values('abs_shap').tail(10).sort_values('shap', ascending=True)
        labels = [f"{r['gene']} ({gene_to_path.get(r['gene'], '기타')})" for _, r in top_10.iterrows()]
        
        colors = plt.cm.magma(np.linspace(0.4, 0.85, 10))
        bars = ax_feat.barh(labels, top_10['shap'], color=colors, height=0.7, edgecolor='white')
        ax_feat.set_title("1. 핵심 예측 지표 (Top 10 Impacts)", fontsize=18, weight='bold', pad=25)
        ax_feat.axvline(0, color='black', lw=1.2)
        ax_feat.grid(axis='x', linestyle='--', alpha=0.4)
        for b in bars:
            ax_feat.text(b.get_width() if b.get_width() > 0 else 0, b.get_y() + 0.35, f' {b.get_width():.2f}', weight='bold')

        # Center Panel (Radar)
        ax_radar = fig.add_subplot(gs[1:, 1], polar=True)
        X_s = self.preprocess(gene_values)[0]
        path_scores = {p: np.mean(X_s[[self.genes_27.index(g) for g in gs]]) for p, gs in self.pathways.items()}
        l_r, s_r = list(path_scores.keys()), list(path_scores.values())
        s_r += s_r[:1]
        angles = np.linspace(0, 2*np.pi, len(l_r), endpoint=False).tolist()
        angles += angles[:1]
        ax_radar.plot(angles, s_r, 'o-', lw=3, color='#E74C3C')
        ax_radar.fill(angles, s_r, color='#E74C3C', alpha=0.25)
        ax_radar.set_thetagrids(np.degrees(angles[:-1]), l_r, fontsize=12, weight='bold')
        ax_radar.set_title("2. 바이오마커 경로 활성도 (Z-Score)", pad=50, fontsize=17, weight='bold')
        ax_radar.set_ylim(-3, 3)

        # Right Panel (Result)
        ax_res = fig.add_subplot(gs[1, 2])
        ax_res.axis('off')
        ax_res.add_patch(patches.FancyBboxPatch((0, 0.05), 1.0, 0.9, boxstyle='round,pad=0.05', 
                                                ec='#2ECC71', fc='white', lw=3, transform=ax_res.transAxes))
        ax_res.text(0.5, 0.82, "최종 예측 결과", ha='center', fontsize=20, weight='bold', color='#27AE60')
        ax_res.text(0.5, 0.52, f"{prob*100:.1f}%", ha='center', va='center', fontsize=60, weight='bold')
        ax_res.text(0.5, 0.12, "양성 (Positive)" if prob >= 0.342 else "음성 (Negative)", 
                   ha='center', fontsize=24, weight='bold', color='#27AE60' if prob >= 0.342 else '#E74C3C')

        # Right Panel (Recommendations)
        ax_rec = fig.add_subplot(gs[2, 2])
        ax_rec.axis('off')
        ax_rec.add_patch(patches.FancyBboxPatch((0, -0.05), 1.0, 1.1, boxstyle='round,pad=0.05', 
                                               fc='#F4ECF7', transform=ax_rec.transAxes))
        ax_rec.add_patch(patches.Rectangle((0, 0.88), 1.0, 0.12, color='#8E44AD', alpha=0.9, transform=ax_rec.transAxes))
        ax_rec.text(0.5, 0.94, "AI 맞춤 치료 제안", ha='center', va='center', fontsize=22, weight='bold', color='white')
        
        rec = "📋 HER2 양성 특성\n   • Trastuzumab/Pertuzumab 표적치료 권장\n\n📋 높은 면역 활성\n   • 면역관문억제제 병용 고려 가능\n\n📋 빠른 세포 증식\n   • 세포독성 항암제 반응성 우수 예상" if prob >= 0.342 else "📋 관찰 요망\n   • 표준 프로토콜 준수\n   • 정밀 추적 검사 권장"
        ax_rec.text(0.08, 0.80, rec, fontsize=16, va='top', linespacing=1.8)

        # Convert to base64
        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return {
            'probability': prob,
            'prediction': 'Positive' if prob >= 0.342 else 'Negative',
            'image': image_base64,
            'top_genes': top_10[['gene', 'shap']].to_dict('records'),
            'pathway_scores': path_scores
        }
