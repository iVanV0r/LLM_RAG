import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from config import *

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

COLORS = {
    'base': '#FF6B6B',
    'rag': '#4ECDC4',
    'ru': '#FF8C42',
    'en': '#45B7D1',
    'improvement': '#96CEB4'
}


class ResultsVisualizer:
    """Визуализация результатов"""

    def __init__(self, output_dir: Path, language: str):
        self.output_dir = output_dir
        self.language = language
        self.charts_dir = output_dir / "charts"
        self.charts_dir.mkdir(parents=True, exist_ok=True)

    def plot_metrics_comparison(self, metrics_summary: Dict, save: bool = True):
        if not metrics_summary:
            return

        metrics = []
        base_values = []
        rag_values = []
        improvements = []

        for metric, values in metrics_summary.items():
            if metric in ['win_rate']:
                continue
            metrics.append(metric.upper())
            base_values.append(values['base_mean'])
            rag_values.append(values['rag_mean'])
            improvements.append(values['improvement_pct'])

        if not metrics:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax1.bar(x - width/2, base_values, width, label='Base', color=COLORS['base'], alpha=0.8)
        bars2 = ax1.bar(x + width/2, rag_values, width, label='RAG', color=COLORS['rag'], alpha=0.8)

        ax1.set_xlabel('Метрики')
        ax1.set_ylabel('Значение')
        ax1.set_title(f'Сравнение метрик ({self.language.upper()})')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()

        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        colors_imp = ['green' if x > 0 else 'red' for x in improvements]
        bars = ax2.bar(metrics, improvements, color=colors_imp, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Метрики')
        ax2.set_ylabel('Улучшение (%)')
        ax2.set_title('Улучшение RAG над Base (%)')

        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5 if height > 0 else height - 2,
                    f'{imp:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

        plt.tight_layout()

        if save:
            path = self.charts_dir / "metrics_comparison.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  ✓ График сохранен: {path}")

        plt.close()

    def plot_per_article_comparison(self, detailed_results: pd.DataFrame, save: bool = True):
        if detailed_results.empty:
            return

        df = detailed_results.copy()
        df['rouge_diff'] = df.get('rougeL_rag', 0) - df.get('rougeL_base', 0)
        df = df.sort_values('rouge_diff')

        fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.3)))

        labels = []
        for _, row in df.iterrows():
            fname = row.get('filename', row.get('article_filename', ''))
            labels.append(fname[:30] + '...' if len(fname) > 30 else fname)

        y_pos = range(len(df))

        ax.barh(y_pos, df['rougeL_base'], height=0.35, label='Base', color=COLORS['base'], alpha=0.7)
        ax.barh(y_pos, df['rougeL_rag'], height=0.35, label='RAG', color=COLORS['rag'], alpha=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('ROUGE-L Score')
        ax.set_title(f'Постатейное сравнение ({self.language.upper()})')
        ax.legend()

        plt.tight_layout()

        if save:
            path = self.charts_dir / "per_article_comparison.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  ✓ График сохранен: {path}")

        plt.close()

    def plot_significance_heatmap(self, significance_data: Dict, save: bool = True):
        if not significance_data:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        metrics = []
        p_values = []
        significant = []

        for metric, values in significance_data.items():
            if isinstance(values, dict) and 'p_value' in values:
                metrics.append(metric.upper())
                p_values.append(values['p_value'])
                significant.append(values.get('significant', False))

        if not metrics:
            return

        data = np.array([p_values])
        im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.1)

        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics)
        ax.set_yticks([])

        for i, (p_val, sig) in enumerate(zip(p_values, significant)):
            color = 'white' if p_val < 0.01 else 'black'
            text = f'p={p_val:.4f}\n{"✓" if sig else "✗"}'
            ax.text(i, 0, text, ha='center', va='center', color=color, fontweight='bold')

        ax.set_title('Статистическая значимость улучшений')

        plt.colorbar(im, ax=ax, label='p-value')
        plt.tight_layout()

        if save:
            path = self.charts_dir / "significance_heatmap.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  ✓ График сохранен: {path}")

        plt.close()

    def create_summary_dashboard(self, metrics_summary: Dict, judge_results: Dict,
                                detailed_df: pd.DataFrame, save: bool = True):
        fig = plt.figure(figsize=(16, 12))

        fig.suptitle(f'RAG Evaluation Dashboard ({self.language.upper()})',
                    fontsize=16, fontweight='bold', y=0.98)

        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Метрики сравнения
        ax1 = fig.add_subplot(gs[0, :2])
        if metrics_summary:
            metrics = [k.upper() for k in metrics_summary if k != 'win_rate']
            base_vals = [metrics_summary[k]['base_mean'] for k in metrics_summary if k != 'win_rate']
            rag_vals = [metrics_summary[k]['rag_mean'] for k in metrics_summary if k != 'win_rate']

            x = np.arange(len(metrics))
            width = 0.35

            ax1.bar(x - width/2, base_vals, width, label='Base', color=COLORS['base'])
            ax1.bar(x + width/2, rag_vals, width, label='RAG', color=COLORS['rag'])
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics)
            ax1.legend()
            ax1.set_title('Metrics Comparison')

        # 2. Win rate pie
        ax2 = fig.add_subplot(gs[0, 2])
        if metrics_summary and 'win_rate' in metrics_summary:
            wr = metrics_summary['win_rate']
            labels = ['RAG', 'Base', 'Tie']
            sizes = [wr['rag_wins'], wr['base_wins'], wr['ties']]
            colors_pie = [COLORS['rag'], COLORS['base'], '#CCCCCC']
            ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%')
            ax2.set_title('Win Rate')

        # 3. Per-article
        ax4 = fig.add_subplot(gs[1, :])
        if not detailed_df.empty:
            if 'rouge_diff' not in detailed_df.columns:
                detailed_df['rouge_diff'] = detailed_df.get('rougeL_rag', 0) - detailed_df.get('rougeL_base', 0)

            sorted_df = detailed_df.sort_values('rouge_diff')
            top3 = sorted_df.tail(3)
            bottom3 = sorted_df.head(3)
            combined = pd.concat([bottom3, top3])

            labels = []
            for _, row in combined.iterrows():
                fname = row.get('filename', row.get('article_filename', ''))
                labels.append(fname[:20])

            values = combined['rouge_diff'].values
            colors_bar = ['red' if v < 0 else 'green' for v in values]
            ax4.barh(range(len(labels)), values, color=colors_bar, alpha=0.7)
            ax4.set_yticks(range(len(labels)))
            ax4.set_yticklabels(labels, fontsize=8)
            ax4.set_xlabel('ROUGE-L Difference')
            ax4.set_title('Best/Worst Articles')
            ax4.axvline(x=0, color='black', linestyle='-')

        # 4. Significance
        ax5 = fig.add_subplot(gs[2, :2])
        if metrics_summary:
            sig_metrics = [k.upper() for k in metrics_summary if k != 'win_rate']
            sig_p = [metrics_summary[k]['p_value'] for k in metrics_summary if k != 'win_rate']
            sig_colors = ['green' if p < 0.05 else 'red' for p in sig_p]

            ax5.bar(sig_metrics, sig_p, color=sig_colors, alpha=0.7)
            ax5.axhline(y=0.05, color='red', linestyle='--', label='p=0.05')
            ax5.set_ylabel('p-value')
            ax5.set_title('Statistical Significance')
            ax5.legend()

        # 5. Stats text
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')

        lines = ["STATISTICS", "="*20]
        if metrics_summary:
            for k in ['rougeL', 'bertscore']:
                if k in metrics_summary:
                    imp = metrics_summary[k]['improvement_pct']
                    lines.append(f"{k}: {imp:+.1f}%")

        ax6.text(0.1, 0.9, '\n'.join(lines), transform=ax6.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace')

        if save:
            path = self.charts_dir / "summary_dashboard.png"
            plt.savefig(path, dpi=200, bbox_inches='tight')
            print(f"  ✓ Дашборд сохранен: {path}")

        plt.close()


class ComparativeVisualizer:
    """Визуализация сравнения языков"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.charts_dir = output_dir / "charts"
        self.charts_dir.mkdir(parents=True, exist_ok=True)

    def plot_language_heatmap(self, ru_metrics: Dict, en_metrics: Dict, save: bool = True):
        fig, ax = plt.subplots(figsize=(10, 6))

        metrics = ['rouge1', 'rouge2', 'rougeL', 'bertscore']
        metric_labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore']

        data = []
        row_labels = ['RU Base', 'RU RAG', 'EN Base', 'EN RAG']

        for summary, prefix in [(ru_metrics, 'RU'), (en_metrics, 'EN')]:
            for method in ['base', 'rag']:
                row = []
                for m in metrics:
                    if m in summary:
                        row.append(summary[m][f'{method}_mean'])
                    else:
                        row.append(0)
                data.append(row)

        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(range(len(metric_labels)))
        ax.set_xticklabels(metric_labels)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)

        for i in range(len(row_labels)):
            for j in range(len(metric_labels)):
                ax.text(j, i, f'{data[i][j]:.3f}',
                       ha="center", va="center", color="black" if data[i][j] < 0.5 else "white")

        ax.set_title('Cross-Language Performance Heatmap')
        plt.colorbar(im, ax=ax)

        if save:
            path = self.charts_dir / "language_heatmap.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Heatmap сохранен: {path}")

        plt.close()

    def plot_improvement_comparison(self, ru_metrics: Dict, en_metrics: Dict, save: bool = True):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        metrics = ['rouge1', 'rouge2', 'rougeL', 'bertscore']
        labels = ['R-1', 'R-2', 'R-L', 'BERT']

        ru_imp = [ru_metrics.get(m, {}).get('improvement_pct', 0) for m in metrics]
        en_imp = [en_metrics.get(m, {}).get('improvement_pct', 0) for m in metrics]

        x = np.arange(len(labels))
        width = 0.35

        ax1.bar(x - width/2, ru_imp, width, label='RU', color=COLORS['ru'])
        ax1.bar(x + width/2, en_imp, width, label='EN', color=COLORS['en'])
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_ylabel('Improvement %')
        ax1.set_title('RAG Improvement by Language')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.legend()

        ax2.scatter(ru_imp, en_imp, s=100, c=[COLORS['ru'], COLORS['ru'], COLORS['en'], COLORS['en']])
        for i, label in enumerate(labels):
            ax2.annotate(label, (ru_imp[i], en_imp[i]),
                       textcoords="offset points", xytext=(5, 5), fontsize=10)

        max_val = max(max(ru_imp), max(en_imp), 1)
        ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
        ax2.set_xlabel('RU Improvement %')
        ax2.set_ylabel('EN Improvement %')
        ax2.set_title('Correlation of Improvements')

        plt.tight_layout()

        if save:
            path = self.charts_dir / "improvement_comparison.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Сравнение сохранено: {path}")

        plt.close()