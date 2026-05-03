"""Consistent matplotlib style for all paper figures."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODEL_COLORS = {
    'TGN': '#2176AE',
    'JODIE': '#57B894',
    'TGAT': '#F0803C',
    'EdgeBank': '#8B8B8B',
    'TGN (no mem)': '#2176AE',
    'SOTA (TPNet)': '#D64045',
}
MODEL_ORDER = ['EdgeBank', 'TGAT', 'JODIE', 'TGN', 'SOTA (TPNet)']

def apply_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
