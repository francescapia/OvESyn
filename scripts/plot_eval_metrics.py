#!/usr/bin/env python3
import os
import re
import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def find_files(root, name):
    return glob.glob(os.path.join(root, '**', name), recursive=True)


def parse_ckpt_tag(path):
    # parent folder name is checkpoint tag like unet_clipFT_FT_200_resume or unet_clipFT_FT_50
    tag = os.path.basename(os.path.dirname(path))
    m = re.search(r"_(\d+)", tag)
    num = int(m.group(1)) if m else -1
    return tag, num


def aggregate_eval_metrics(root):
    files = find_files(root, 'summary_metrics.csv')
    records = []
    for f in files:
        tag, num = parse_ckpt_tag(f)
        df = pd.read_csv(f)
        row = df.iloc[0].to_dict()
        run_type = 'classic'
        if '2026-05-22_multigpu' in f:
            run_type = 'multigpu'
        records.append({
            'run': run_type,
            'ckpt_tag': tag,
            'ckpt': num,
            'FID25D_overall': row.get('FID25D_overall'),
            'FID3D': row.get('FID3D'),
            'Precision': row.get('Precision'),
            'Recall': row.get('Recall'),
        })
    return pd.DataFrame(records)


def aggregate_semantic_metrics(root):
    files = find_files(root, 'summary_semantic_metrics.csv')
    records = []
    for f in files:
        tag, num = parse_ckpt_tag(f)
        df = pd.read_csv(f)
        row = df.iloc[0].to_dict()
        run_type = 'classic'
        if '2026-05-22_multigpu' in f:
            run_type = 'multigpu'
        records.append({
            'run': run_type,
            'ckpt_tag': tag,
            'ckpt': num,
            'presence_agreement_mean': row.get('presence_agreement_mean'),
            'dice_mean': row.get('dice_mean'),
            'rel_volume_error_mean': row.get('rel_volume_error_mean'),
            'mentioned_presence_f1_mean': row.get('mentioned_presence_f1_mean'),
            'n_cases_evaluated': row.get('n_cases_evaluated')
        })
    return pd.DataFrame(records)


def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)


def plot_eval(df, outdir):
    ensure_outdir(outdir)
    df = df.dropna(subset=['ckpt']).sort_values(['run','ckpt'])

    for metric, ylabel in [('FID25D_overall','FID25D (overall)'), ('FID3D','FID 3D'), ('Precision','Precision'), ('Recall','Recall')]:
        plt.figure(figsize=(8,4))
        for run, g in df.groupby('run'):
            plt.plot(g['ckpt'], g[metric], marker='o', label=run)
        plt.xlabel('Checkpoint (epoch)')
        plt.ylabel(ylabel)
        plt.title(f'Andamento {ylabel} per checkpoint (multigpu vs classic_run)')
        plt.legend()
        plt.grid(alpha=0.3)
        out = os.path.join(outdir, f'eval_{metric}.png')
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()


def plot_semantic(df, outdir):
    ensure_outdir(outdir)
    df = df.dropna(subset=['ckpt']).sort_values(['run','ckpt'])
    for metric, ylabel in [('presence_agreement_mean','Presence agreement'), ('dice_mean','Dice (mean)'), ('rel_volume_error_mean','Relative volume error (%)'), ('mentioned_presence_f1_mean','Mention presence F1')]:
        plt.figure(figsize=(8,4))
        for run, g in df.groupby('run'):
            plt.plot(g['ckpt'], g[metric], marker='o', label=run)
        plt.xlabel('Checkpoint (epoch)')
        plt.ylabel(ylabel)
        plt.title(f'Andamento {ylabel} per checkpoint (multigpu vs classic_run)')
        plt.legend()
        plt.grid(alpha=0.3)
        out = os.path.join(outdir, f'semantic_{metric}.png')
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()


def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    eval_root = os.path.join(repo_root, 'results', 'eval_test')
    sem_root = os.path.join(repo_root, 'results', 'semantic_eval')
    outdir = os.path.join(repo_root, 'results', 'plots')

    eval_df = aggregate_eval_metrics(eval_root)
    sem_df = aggregate_semantic_metrics(sem_root)

    # save aggregated tables
    ensure_outdir(outdir)
    eval_df.to_csv(os.path.join(outdir, 'agg_eval_metrics.csv'), index=False)
    sem_df.to_csv(os.path.join(outdir, 'agg_semantic_metrics.csv'), index=False)

    plot_eval(eval_df, outdir)
    plot_semantic(sem_df, outdir)

    print('Plots saved to', outdir)


if __name__ == '__main__':
    main()
