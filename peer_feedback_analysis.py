"""
Peer Feedback Analysis - XAI Project Part 2
Generates 3 plots + 1 report from peer feedback data

Usage: python peer_feedback_analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# DATA (Extracted from PDF)
# ============================================================================

feedback_data = {
    'Scenario': [1]*10 + [2]*10 + [3]*10 + [4]*10 + [5]*10,
    'Clarity': [
        5, 5, 5, 3, 2, 2, 2, 1, 1, 1,
        5, 5, 5, 5, 4, 4, 4, 4, 3, 2,
        5, 5, 5, 5, 4, 4, 4, 4, 4, 3,
        5, 5, 5, 5, 5, 4, 4, 4, 4, 4,
        5, 5, 5, 4, 4, 3, 3, 3, 3, 2
    ],
    'Understandability': [
        5, 5, 5, 4, 4, 3, 3, 3, 2, 2,
        5, 5, 5, 5, 5, 4, 4, 4, 3, 3,
        5, 5, 5, 5, 4, 4, 4, 3, 3, 3,
        5, 5, 5, 5, 5, 5, 5, 5, 4, 4,
        5, 5, 4, 4, 4, 4, 3, 3, 3, 2
    ],
    'Appropriateness': [
        5, 5, 4, 2, 2, 2, 2, 2, 1, 1,
        5, 5, 5, 5, 4, 4, 4, 4, 3, 3,
        5, 5, 5, 5, 5, 4, 4, 4, 4, 3,
        5, 5, 5, 5, 5, 5, 4, 4, 4, 3,
        5, 5, 5, 4, 4, 3, 3, 3, 2, 2
    ],
    'BestInterest': [
        5, 5, 5, 3, 3, 2, 1, 1, 1, 1,
        5, 5, 5, 5, 4, 4, 4, 3, 3, 2,
        5, 5, 5, 5, 5, 4, 4, 4, 4, 3,
        5, 5, 5, 5, 5, 5, 4, 4, 4, 4,
        5, 5, 4, 4, 4, 4, 3, 2, 2, 2
    ],
    'Satisfaction': [
        5, 5, 5, 2, 2, 2, 2, 1, 1, 1,
        5, 5, 5, 5, 4, 4, 4, 4, 3, 2,
        5, 5, 4, 4, 4, 4, 4, 4, 3, 3,
        5, 5, 5, 4, 4, 4, 4, 4, 4, 4,
        5, 4, 4, 4, 4, 3, 3, 3, 3, 2
    ],
    'Comments': [
        'Contradicts itself', 'Mistake in prompt', 'Does not address question',
        'Preferences not considered', 'Not clear', 'Vague', 'Should mention prohibition',
        'Could specify goal', 'Understandable but not clear', 'Does not describe decision',
        'Clear, quality not justified', 'Great explanation', 'Accurate and direct',
        'Heavy cognitive load', 'Preference order misleading', 'Best explanation',
        'Logic still flawed', 'Very readable', 'Solid explanation', 'Clearly written',
        'Quality comparison unclear', 'Short and concise', 'Good structure', 'Language awkward',
        'Slightly unnatural', 'Interested in method', 'English flows naturally', 'Clear',
        'Good structure', 'Mentions alternatives', 'Clear reasoning', 'Concise',
        'Accurate explanation', 'Language not everyday', 'Slightly confusing', 'Which rule?',
        'Clear norm following', 'Very concise', 'Shows reasoning', 'Good explanation',
        'Clear description', 'Grammar issues', 'Wrong preference order', 'Irrelevant info',
        'Vague explanations', 'Price unclear', 'Confusing overall', 'Could specify action',
        'Different preference order', 'Very good overall'
    ]
}

df = pd.DataFrame(feedback_data)
criteria = ['Clarity', 'Understandability', 'Appropriateness', 'BestInterest', 'Satisfaction']

# ============================================================================
# PLOT 1: STACKED BAR CHART
# ============================================================================

def plot_stacked_bar():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    rating_counts = {criterion: df[criterion].value_counts().sort_index() for criterion in criteria}
    ratings = [1, 2, 3, 4, 5]
    data = {rating: [rating_counts[c].get(rating, 0) for c in criteria] for rating in ratings}
    
    x = np.arange(len(criteria))
    bottom = np.zeros(len(criteria))
    colors = ['#d62728', '#ff7f0e', '#ffdd57', '#2ca02c', '#1f77b4']
    labels = ['1 (Poor)', '2 (Fair)', '3 (Good)', '4 (Very Good)', '5 (Excellent)']
    
    for rating, color, label in zip(ratings, colors, labels):
        ax.bar(x, data[rating], 0.6, label=label, bottom=bottom, color=color)
        bottom += data[rating]
    
    ax.set_xlabel('Criteria', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Responses', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Ratings Across Criteria', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(criteria, rotation=45, ha='right')
    ax.legend(title='Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feedback_stacked_bar.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: feedback_stacked_bar.png")
    plt.close()

# ============================================================================
# PLOT 2: RADAR CHART
# ============================================================================

def plot_radar():
    means = df[criteria].mean()
    num_vars = len(criteria)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values = means.tolist()
    values += values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=2, label='Mean Score', color='#1f77b4')
    ax.fill(angles, values, alpha=0.25, color='#1f77b4')
    ax.set_ylim(0, 5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria, fontsize=11)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=9)
    ax.grid(True)
    ax.set_title('Mean Scores Across Five Criteria', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig('feedback_radar_chart.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: feedback_radar_chart.png")
    plt.close()

# ============================================================================
# PLOT 3: THEMATIC HEATMAP
# ============================================================================

def plot_heatmap():
    themes = {
        'Logic Errors': ['logic', 'error', 'mistake', 'wrong', 'incorrect', 'contradict'],
        'Natural Phrasing': ['natural', 'phrasing', 'wording', 'language', 'awkward', 'grammar'],
        'Tone': ['tone', 'friendly', 'formal', 'casual'],
        'Clarity': ['clear', 'clarity', 'confusing', 'vague'],
        'Technical': ['technical', 'jargon', 'simple', 'complex'],
        'Completeness': ['missing', 'incomplete', 'context', 'detail'],
        'Positive': ['good', 'great', 'excellent', 'best']
    }
    
    scenarios = sorted(df['Scenario'].unique())
    theme_matrix = []
    
    for scenario in scenarios:
        comments = df[df['Scenario'] == scenario]['Comments'].tolist()
        counts = []
        for theme, keywords in themes.items():
            count = sum(1 for c in comments if any(k in c.lower() for k in keywords))
            counts.append(count)
        theme_matrix.append(counts)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(theme_matrix, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(np.arange(len(themes)))
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_xticklabels(themes.keys(), rotation=45, ha='right')
    ax.set_yticklabels([f'Scenario {s}' for s in scenarios])
    
    for i in range(len(scenarios)):
        for j in range(len(themes)):
            ax.text(j, i, theme_matrix[i][j], ha="center", va="center", color="black", fontsize=10)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Frequency', rotation=270, labelpad=20)
    
    ax.set_title('Thematic Analysis of Qualitative Feedback', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Themes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Scenarios', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('feedback_thematic_heatmap.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: feedback_thematic_heatmap.png")
    plt.close()

# ============================================================================
# ANALYSIS REPORT
# ============================================================================

def generate_report():
    means = df[criteria].mean()
    overall_mean = means.mean()
    
    report = [
        "="*70,
        "PEER FEEDBACK ANALYSIS REPORT",
        "="*70,
        "",
        f"Total Responses: {len(df)}",
        f"Number of Scenarios: {df['Scenario'].nunique()}",
        "",
        "Mean Scores (out of 5.00):",
    ]
    
    for criterion, mean in means.items():
        report.append(f"  {criterion:20s}: {mean:.2f}")
    
    report.extend([
        "",
        f"Overall Mean Score: {overall_mean:.2f}/5.00",
        "",
        f"Strongest: {means.idxmax()} ({means.max():.2f})",
        f"Weakest: {means.idxmin()} ({means.min():.2f})",
        "",
        "="*70
    ])
    
    report_text = "\n".join(report)
    with open('feedback_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("[OK] Saved: feedback_analysis_report.txt")
    print("\n" + report_text)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("PEER FEEDBACK ANALYSIS")
    print("="*70 + "\n")
    
    print("Generating visualizations...\n")
    plot_stacked_bar()
    plot_radar()
    plot_heatmap()
    
    print("\nGenerating report...\n")
    generate_report()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. feedback_stacked_bar.png")
    print("  2. feedback_radar_chart.png")
    print("  3. feedback_thematic_heatmap.png")
    print("  4. feedback_analysis_report.txt")
    print("="*70 + "\n")
