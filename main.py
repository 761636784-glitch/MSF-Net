"""MSF-Net revision experiment: strict patient-grouped 5x4 nested CV.

- Outer 5-fold CV: each outer test fold is used only once for evaluation.
- Inner 4-fold CV: selects training epoch and classification thresholds from inner OOF predictions.
- Spectral input: 5x5 regional mean/std descriptors [25,2,256], matching Experiment C.
- Primary patient aggregation: top 20% of patch probabilities; alternative rules are sensitivity analyses.
- Bootstrap confidence intervals resample patients (and keep all patches from a sampled patient together).
- ImageNet-pretrained ResNet34 morphology backbone; stem and layer1 frozen, layer2-layer4 fine-tuned.
"""
from __future__ import annotations
import os
# Configure cuBLAS reproducibility BEFORE importing torch / initializing CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import csv
import hashlib
import json
import random
import time
import platform
import sys
from dataclasses import asdict
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             confusion_matrix, recall_score)
from sklearn.model_selection import StratifiedGroupKFold

from config import CFG
from dataset import load_manifest
from model import MSFNet

PRIMARY_AGG = 'top20'
AGGS = ('mean', 'median', 'max', 'top5', 'top10', 'top20', 'top30')
METRICS = ('auc', 'acc', 'f1', 'sensitivity', 'specificity')


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Warn rather than interrupt on an unsupported deterministic CUDA operator.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def dump_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csvwrite(path: Path, rows, columns):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path):
    hasher = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def experiment_signature(args, manifest_digest):
    # Ensure cached outer folds are never silently reused after modifying code.
    code_hashes = {name: file_sha256(Path(__file__).resolve().parent / name)
                   for name in ('main.py', 'model.py', 'dataset.py', 'config.py')}
    cfg = {name: str(value) if isinstance(value, Path) else value
           for name, value in asdict(CFG).items()}
    params = vars(args).copy()
    # Execution mode does not change model predictions.
    params.pop('dry_run', None)
    params.pop('nested', None)
    # Bootstrap affects only statistical postprocessing, not trained predictions.
    params.pop('bootstrap', None)
    # Checkpoint saving does not alter predictions.
    params.pop('no_save_models', None)
    signature = {'manifest_digest': manifest_digest, 'config': cfg, 'args_for_training': params,
                 'code_hashes': code_hashes, 'pretrained': 'ResNet34_Weights.IMAGENET1K_V1',
                 'primary_aggregation': PRIMARY_AGG, 'training_loss': 'cross_entropy',
                 'regional_mean_std_spectrum': True, 'determinism': 'cublas_workspace4096_8; cudnn deterministic; warn_only'}
    digest = hashlib.sha256(json.dumps(signature, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return signature, digest


class PatchDataset(Dataset):
    def __init__(self, patient_ids, train=False):
        self.train = train
        self.cache = {}
        with (CFG.output_dir / f'patch_index_region25_seed{CFG.seed}.csv').open(newline='', encoding='utf-8-sig') as f:
            self.rows = [r for r in csv.DictReader(f) if r['patient_id'] in patient_ids]
        expected = len(patient_ids) * 195
        if len(self.rows) != expected:
            raise RuntimeError(f'Expected {expected} patches for {len(patient_ids)} patients, got {len(self.rows)}')

    def __len__(self):
        return len(self.rows)

    def _array(self, path):
        if path not in self.cache:
            self.cache[path] = np.load(path, mmap_mode='r')
        return self.cache[path]

    def __getitem__(self, i):
        row = self.rows[i]
        j = int(row['patch_index'])
        rgb = np.array(self._array(row['rgb_file'])[j], dtype=np.float32, copy=True)
        spectrum = np.array(self._array(row['spec_file'])[j], dtype=np.float32, copy=True)
        if spectrum.shape != (25, 2, 256):
            raise RuntimeError(f"Expected regional spectrum (25,2,256), got {spectrum.shape}; rerun dataset.py --prepare")
        if self.train:
            # Apply identical spatial transforms to pseudo-RGB and its aligned 5x5 spectral grid.
            spec_grid = spectrum.reshape(5, 5, 2, 256)
            if random.random() < 0.5:
                rgb = np.flip(rgb, axis=2)
                spec_grid = np.flip(spec_grid, axis=1)
            if random.random() < 0.5:
                rgb = np.flip(rgb, axis=1)
                spec_grid = np.flip(spec_grid, axis=0)
            k = random.randrange(4)
            rgb = np.rot90(rgb, k, axes=(1, 2))
            spec_grid = np.rot90(spec_grid, k, axes=(0, 1))
            spectrum = spec_grid.reshape(25, 2, 256)
        return (torch.from_numpy(np.ascontiguousarray(rgb)),
                torch.from_numpy(np.ascontiguousarray(spectrum)),
                int(row['label']), row['patient_id'], j)


def get_loaders(train_ids, valid_ids, args, device):
    def loader(ids, train):
        return DataLoader(PatchDataset(set(ids), train), batch_size=args.batch_size,
                          shuffle=train, num_workers=args.num_workers,
                          pin_memory=device.type == 'cuda')
    return loader(train_ids, True), loader(valid_ids, False)


def new_model_optimizer(args, device):
    # Match the original manuscript code: ImageNet-pretrained ResNet34.
    # The morphology backbone (including its SE blocks) uses a smaller fine-tuning
    # LR; the spectral branch, fusion layers and classifier use args.lr.
    model = MSFNet(pretrained=True).to(device)
    morphology_params = [p for p in model.morphology.parameters() if p.requires_grad]
    other_params = [p for name, p in model.named_parameters()
                    if p.requires_grad and not name.startswith('morphology.')]
    optimizer = torch.optim.AdamW(
        [
            {'params': morphology_params, 'lr': args.backbone_lr},
            {'params': other_params, 'lr': args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    return model, optimizer, scaler


def train_epoch(model, optimizer, scaler, loader, device, args):
    model.train()
    loss_sum, n = 0.0, 0
    for batch_no, (rgb, spectrum, y, _, _) in enumerate(loader, 1):
        rgb = rgb.to(device, non_blocking=True)
        spectrum = spectrum.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', enabled=device.type == 'cuda'):
            logits = model(rgb, spectrum)
            loss = F.cross_entropy(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.detach().item() * len(y)
        n += len(y)
        if args.max_train_steps and batch_no >= args.max_train_steps:
            break
    return loss_sum / max(1, n)


def predictions(model, loader, device):
    model.eval()
    records, loss_sum, n = [], 0.0, 0
    with torch.inference_mode():
        for rgb, spectrum, y, patient_ids, patch_ids in loader:
            rgb = rgb.to(device, non_blocking=True)
            spectrum = spectrum.to(device, non_blocking=True)
            y = y.to(device)
            with torch.autocast('cuda', enabled=device.type == 'cuda'):
                logits = model(rgb, spectrum)
                loss = F.cross_entropy(logits, y)
            probs = logits.float().softmax(1)[:, 1].cpu().tolist()
            records.extend({'patient_id': pid, 'patch_index': int(j), 'label': int(lbl),
                            'prob_ehcc': float(prob)}
                           for pid, j, lbl, prob in zip(patient_ids, patch_ids.tolist(), y.cpu().tolist(), probs))
            loss_sum += loss.item() * len(y)
            n += len(y)
    return records, loss_sum / max(n, 1)


def aggregate(rows, method):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['patient_id']].append(row)
    result = []
    for pid, data in sorted(grouped.items()):
        probs = np.asarray([v['prob_ehcc'] for v in data], dtype=float)
        labels = {int(v['label']) for v in data}
        if len(labels) != 1:
            raise ValueError(f'Conflicting labels for patient {pid}')
        if method == 'mean':
            score = float(np.mean(probs))
        elif method == 'median':
            score = float(np.median(probs))
        elif method == 'max':
            score = float(np.max(probs))
        elif method in ('top5', 'top10', 'top20', 'top30'):
            proportion = int(method[3:]) / 100.0
            k = max(1, int(np.ceil(len(probs) * proportion)))
            score = float(np.mean(np.sort(probs)[-k:]))
        else:
            raise ValueError(f'Unknown aggregation: {method}')
        result.append({'patient_id': pid, 'label': int(next(iter(labels))),
                       'prob_ehcc': score, 'n_patches': len(probs)})
    return result


def select_threshold(rows):
    """Select accuracy-maximizing threshold on development/inner-OOF predictions only."""
    labels = np.asarray([int(r['label']) for r in rows])
    scores = np.asarray([float(r['prob_ehcc']) for r in rows])
    unique = np.unique(scores)
    mids = (unique[:-1] + unique[1:]) / 2 if len(unique) > 1 else np.asarray([])
    candidates = np.unique(np.r_[0.0, mids, 0.5, 1.0, np.nextafter(1.0, 2.0)])
    accs = np.asarray([np.mean((scores >= t).astype(int) == labels) for t in candidates])
    winners = candidates[np.isclose(accs, accs.max(), atol=1e-12)]
    return float(winners[np.argmin(np.abs(winners - 0.5))])


def metric_summary(rows, threshold_key):
    y = np.asarray([int(r['label']) for r in rows])
    p = np.asarray([float(r['prob_ehcc']) for r in rows])
    thresholds = np.asarray([float(r[threshold_key]) for r in rows])
    pred = (p >= thresholds).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'n': int(len(y)),
            'auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            'acc': float(accuracy_score(y, pred)),
            'f1': float(f1_score(y, pred, zero_division=0)),
            'sensitivity': float(tp / (tp + fn)) if tp + fn else None,
            'specificity': float(tn / (tn + fp)) if tn + fp else None,
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def _weighted_metrics(y, p, pred, weights):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1], sample_weight=weights).ravel()
    return {'auc': float(roc_auc_score(y, p, sample_weight=weights)),
            'acc': float(accuracy_score(y, pred, sample_weight=weights)),
            'f1': float(f1_score(y, pred, sample_weight=weights, zero_division=0)),
            'sensitivity': float(tp / (tp + fn)) if tp + fn else None,
            'specificity': float(tn / (tn + fp)) if tn + fp else None}


def patient_cluster_bootstrap_ci(patch_rows, patient_rows_by_agg, n_boot, seed):
    """Stratified patient bootstrap; all patches from a sampled patient share its multiplicity.

    Thresholds are fixed at the values learned from the corresponding inner OOF data.
    Thus intervals quantify sampling uncertainty conditional on the fitted outer-fold models
    and selected thresholds; they do not include the cost of retraining the whole CV pipeline.
    """
    rng = np.random.default_rng(seed)
    primary_patients = patient_rows_by_agg[PRIMARY_AGG]
    patient_by_id = {r['patient_id']: r for r in primary_patients}
    ids0 = [pid for pid, r in patient_by_id.items() if int(r['label']) == 0]
    ids1 = [pid for pid, r in patient_by_id.items() if int(r['label']) == 1]
    if not ids0 or not ids1:
        raise RuntimeError('Both classes are required for stratified bootstrap')
    patch_by_id = defaultdict(list)
    for row in patch_rows:
        patch_by_id[row['patient_id']].append(row)
    patient_arrays = {}
    for agg, rows in patient_rows_by_agg.items():
        patient_arrays[agg] = {
            'rows': rows,
            'y': np.asarray([int(r['label']) for r in rows]),
            'p': np.asarray([float(r['prob_ehcc']) for r in rows]),
            'pred': np.asarray([int(float(r['prob_ehcc']) >= float(r['threshold'])) for r in rows]),
            'ids': [r['patient_id'] for r in rows],
        }
    py = np.asarray([int(r['label']) for r in patch_rows])
    pp = np.asarray([float(r['prob_ehcc']) for r in patch_rows])
    pth = np.asarray([float(r['threshold_patch']) for r in patch_rows])
    ppred = (pp >= pth).astype(int)
    pids = [r['patient_id'] for r in patch_rows]

    boot_values = {'patient': {agg: {m: [] for m in METRICS} for agg in patient_rows_by_agg},
                   'patch': {m: [] for m in METRICS}}
    for _ in range(n_boot):
        sampled = list(rng.choice(ids0, size=len(ids0), replace=True)) + list(rng.choice(ids1, size=len(ids1), replace=True))
        multiplicity = Counter(sampled)
        for agg, arr in patient_arrays.items():
            weights = np.asarray([multiplicity.get(pid, 0) for pid in arr['ids']], dtype=float)
            metrics = _weighted_metrics(arr['y'], arr['p'], arr['pred'], weights)
            for m in METRICS:
                boot_values['patient'][agg][m].append(metrics[m])
        patch_weights = np.asarray([multiplicity.get(pid, 0) for pid in pids], dtype=float)
        metrics = _weighted_metrics(py, pp, ppred, patch_weights)
        for m in METRICS:
            boot_values['patch'][m].append(metrics[m])

    def intervals(values):
        return {m: [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
                for m, vals in values.items()}
    return {'method': 'stratified patient-level bootstrap; all patches clustered by patient',
            'replicates': int(n_boot),
            'thresholds_fixed_from_inner_oof': True,
            'patient_level_95ci': {agg: intervals(vals) for agg, vals in boot_values['patient'].items()},
            'patch_level_patient_cluster_bootstrap_95ci': intervals(boot_values['patch'])}


def fit(args):
    seed_all(CFG.seed)
    manifest = load_manifest()
    ids = np.asarray([r['patient_id'] for r in manifest])
    y = np.asarray([int(r['label']) for r in manifest])
    if len(ids) != 60 or len(set(ids)) != 60 or int(np.sum(y)) != 30:
        raise RuntimeError('Expected the verified 60-patient cohort: 30 eHCC and 30 HGDN')
    manifest_digest = hashlib.sha256(('\n'.join(sorted(f'{pid},{lbl}' for pid, lbl in zip(ids, y)))).encode()).hexdigest()[:16]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    signature, run_fingerprint = experiment_signature(args, manifest_digest)
    out = CFG.output_dir / f'nested_cv_MSFNet_region25_seed{CFG.seed}' / f'seed{CFG.seed}_ep{args.epochs}_bs{args.batch_size}_{run_fingerprint[:12]}'
    out.mkdir(parents=True, exist_ok=True)
    run_config = {
        'args': vars(args), 'seed': CFG.seed, 'patient_manifest_hash': manifest_digest,
        'run_fingerprint': run_fingerprint, 'experiment_signature': signature,
        'environment': {'python': sys.version, 'platform': platform.platform(),
                        'torch': torch.__version__, 'numpy': np.__version__,
                        'cuda_runtime': torch.version.cuda,
                        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')},
        'outer_folds': CFG.outer_folds, 'inner_folds': CFG.inner_folds,
        'primary_patient_aggregation': PRIMARY_AGG, 'sensitivity_aggregations': list(AGGS),
        'spectral_input': '5x5 regional mean/std descriptors [25,2,256], with per-region mean-spectrum SNV and std scaling',
        'pretrained_weights': 'ImageNet ResNet34 IMAGENET1K_V1',
        'backbone_finetuning': 'stem and layer1 frozen; layer2-layer4 trainable; SE blocks trainable',
        'learning_rates': {'trainable_morphology_branch': args.backbone_lr, 'spectral_fusion_classifier': args.lr},
        'threshold_selection': 'accuracy-maximizing thresholds selected from inner OOF predictions only',
        'outer_test_policy': 'outer test patients used once only; not used for tuning or threshold selection',
        'bootstrap': 'stratified patient-level bootstrap; thresholds fixed from inner OOF predictions'
    }
    dump_json(out / 'run_config.json', run_config)

    outer = StratifiedGroupKFold(n_splits=CFG.outer_folds, shuffle=True, random_state=CFG.seed)
    splits = list(outer.split(ids, y, groups=ids))
    if args.dry_run:
        for fold, (dev, test) in enumerate(splits, 1):
            inner = StratifiedGroupKFold(n_splits=CFG.inner_folds, shuffle=True, random_state=CFG.seed + fold)
            print('OUTER', fold, 'development', len(dev), 'test', len(test),
                  'test class counts', np.bincount(y[test]).tolist())
            for j, (tr, va) in enumerate(inner.split(ids[dev], y[dev], groups=ids[dev]), 1):
                print('  inner', j, 'train', len(tr), 'validation', len(va))
        print('Dry run complete: no model fitting or outer evaluation')
        return

    all_patch_oof = []
    all_patient_oof = {agg: [] for agg in AGGS}
    fold_summaries = []
    for fold, (dev, test) in enumerate(splits, 1):
        fold_dir = out / f'outer_{fold:02d}'
        fold_dir.mkdir(exist_ok=True)
        complete_file = fold_dir / 'COMPLETE.json'
        if complete_file.exists():
            summary = json.loads(complete_file.read_text(encoding='utf-8'))
            if summary.get('manifest_hash') != manifest_digest or summary.get('run_fingerprint') != run_fingerprint:
                raise RuntimeError('Manifest or code/config fingerprint changed since fold completion')
            if not (fold_dir / 'outer_test_patches.csv').exists():
                raise RuntimeError('Fold marked COMPLETE but outer predictions are missing')
            print('REUSE completed outer fold', fold)
        else:
            print(f'\n=== OUTER {fold}/{CFG.outer_folds}: inner selection (outer test untouched) ===', flush=True)
            inner = StratifiedGroupKFold(n_splits=CFG.inner_folds, shuffle=True, random_state=CFG.seed + fold)
            inner_oof, chosen_epochs, inner_stats = [], [], []
            for k, (tr_idx, val_idx) in enumerate(inner.split(ids[dev], y[dev], groups=ids[dev]), 1):
                tr, val = ids[dev][tr_idx], ids[dev][val_idx]
                if set(tr) & set(val) or set(val) & set(ids[test]) or set(tr) & set(ids[test]):
                    raise RuntimeError('Patient leakage detected across inner/outer partitions')
                seed_all(CFG.seed + fold * 100 + k)
                train_loader, val_loader = get_loaders(tr, val, args, device)
                model, optimizer, scaler = new_model_optimizer(args, device)
                best_auc, best_epoch, best_rows, wait = -1.0, 1, [], 0
                history = []
                for epoch in range(1, args.epochs + 1):
                    start = time.time()
                    train_loss = train_epoch(model, optimizer, scaler, train_loader, device, args)
                    rows, val_loss = predictions(model, val_loader, device)
                    auc = float(roc_auc_score([r['label'] for r in rows], [r['prob_ehcc'] for r in rows]))
                    history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                                    'patch_auc': auc, 'seconds': round(time.time() - start, 1)})
                    print(f'OUTER={fold} INNER={k} epoch={epoch} train_loss={train_loss:.4f} val_auc={auc:.4f} val_loss={val_loss:.4f}', flush=True)
                    if auc > best_auc + 1e-7:
                        best_auc, best_epoch, best_rows, wait = auc, epoch, rows, 0
                    else:
                        wait += 1
                    if wait >= args.patience:
                        break
                chosen_epochs.append(best_epoch)
                inner_oof.extend(best_rows)
                inner_stats.append({'inner_fold': k, 'best_epoch': best_epoch,
                                    'best_patch_auc': best_auc, 'history': history})
                dump_json(fold_dir / f'inner_{k}_history.json', inner_stats[-1])
                del model, optimizer, scaler, train_loader, val_loader

            if len({r['patient_id'] for r in inner_oof}) != len(dev):
                raise RuntimeError('Inner OOF predictions do not cover every development patient exactly once')
            csvwrite(fold_dir / 'inner_oof_patches.csv', inner_oof,
                     ['patient_id', 'patch_index', 'label', 'prob_ehcc'])
            thresholds = {'patch': select_threshold(inner_oof)}
            for agg in AGGS:
                thresholds[agg] = select_threshold(aggregate(inner_oof, agg))
            selected_epoch = max(1, int(np.median(chosen_epochs) + 0.5))
            dump_json(fold_dir / 'inner_selection.json', {
                'selected_epoch': selected_epoch, 'inner_best_epochs': chosen_epochs,
                'inner_fold_best_auc': [r['best_patch_auc'] for r in inner_stats],
                'thresholds': thresholds,
                'primary_patient_aggregation': PRIMARY_AGG,
                'thresholds_selected_from': 'inner OOF predictions only'})
            print(f'OUTER={fold} selected epoch={selected_epoch}; thresholds={thresholds}', flush=True)

            seed_all(CFG.seed + fold * 1000)
            final_train_loader, _unused_outer_test_loader = get_loaders(ids[dev], ids[test], args, device)
            model, optimizer, scaler = new_model_optimizer(args, device)
            for epoch in range(1, selected_epoch + 1):
                start = time.time()
                loss = train_epoch(model, optimizer, scaler, final_train_loader, device, args)
                print(f'OUTER={fold} refit epoch={epoch}/{selected_epoch} loss={loss:.4f} seconds={time.time()-start:.1f}', flush=True)
            if not args.no_save_models:
                torch.save({'model': model.state_dict(), 'epoch': selected_epoch}, fold_dir / 'outer_model.pt')
            test_loader = DataLoader(PatchDataset(set(ids[test]), False), batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers,
                                     pin_memory=device.type == 'cuda')
            test_rows, test_loss = predictions(model, test_loader, device)
            if len(test_rows) != len(test) * 195:
                raise RuntimeError('Outer test patch count mismatch')
            for row in test_rows:
                row['threshold_patch'] = float(thresholds['patch'])
                row['outer_fold'] = fold
            csvwrite(fold_dir / 'outer_test_patches.csv', test_rows,
                     ['patient_id', 'patch_index', 'label', 'prob_ehcc', 'threshold_patch', 'outer_fold'])
            summary = {'outer_fold': fold, 'manifest_hash': manifest_digest, 'run_fingerprint': run_fingerprint,
                       'selected_epoch': selected_epoch, 'inner_thresholds': thresholds,
                       'outer_test_patient_ids': sorted(ids[test].tolist()),
                       'outer_test_patch_count': len(test_rows), 'outer_test_loss': float(test_loss)}
            dump_json(complete_file, summary)
            del model, optimizer, scaler, final_train_loader, _unused_outer_test_loader, test_loader

        summary = json.loads(complete_file.read_text(encoding='utf-8'))
        test_rows = list(csv.DictReader((fold_dir / 'outer_test_patches.csv').open(encoding='utf-8-sig')))
        fold_rows = [{'patient_id': r['patient_id'], 'patch_index': int(r['patch_index']),
                      'label': int(r['label']), 'prob_ehcc': float(r['prob_ehcc']),
                      'threshold_patch': float(r['threshold_patch']), 'outer_fold': int(r['outer_fold'])}
                     for r in test_rows]
        if any(r['patient_id'] in set(ids[dev]) for r in fold_rows):
            raise RuntimeError('An outer test patient appeared in the development set')
        all_patch_oof.extend(fold_rows)
        fold_patient_rows = {}
        for agg in AGGS:
            patients = aggregate(fold_rows, agg)
            for r in patients:
                r.update({'outer_fold': fold, 'threshold': float(summary['inner_thresholds'][agg])})
            all_patient_oof[agg].extend(patients)
            fold_patient_rows[agg] = patients
        fold_patch_metrics = metric_summary(fold_rows, 'threshold_patch')
        fold_patch_fixed05 = metric_summary([{**r, 'fixed_threshold': 0.5} for r in fold_rows], 'fixed_threshold')
        fold_summaries.append({
            'outer_fold': fold, 'n_test_patients': len(summary['outer_test_patient_ids']),
            'selected_epoch': summary['selected_epoch'], 'threshold_patch': summary['inner_thresholds']['patch'],
            'patch_metrics': fold_patch_metrics,
            'patch_metrics_fixed_0_5': fold_patch_fixed05,
            'patient_metrics_by_aggregation': {agg: metric_summary(rows, 'threshold')
                                               for agg, rows in fold_patient_rows.items()},
            'patient_metrics_by_aggregation_fixed_0_5': {
                agg: metric_summary([{**r, 'fixed_threshold': 0.5} for r in rows], 'fixed_threshold')
                for agg, rows in fold_patient_rows.items()}
        })
        print(f"OUTER {fold} complete; patch AUC={fold_patch_metrics['auc']:.4f}; "
              f"patient top20 AUC={metric_summary(fold_patient_rows[PRIMARY_AGG], 'threshold')['auc']:.4f}", flush=True)

    if len({r['patient_id'] for r in all_patch_oof}) != 60 or len(all_patch_oof) != 11700:
        raise RuntimeError('Missing/duplicate outer patients or patches in pooled OOF predictions')
    csvwrite(out / 'oof_patches_60patients.csv', all_patch_oof,
             ['patient_id', 'patch_index', 'label', 'prob_ehcc', 'threshold_patch', 'outer_fold'])

    patient_metrics = {}
    for agg, rows in all_patient_oof.items():
        csvwrite(out / f'oof_patients_{agg}.csv', rows,
                 ['patient_id', 'label', 'prob_ehcc', 'n_patches', 'outer_fold', 'threshold'])
        patient_metrics[agg] = metric_summary(rows, 'threshold')
    patch_metrics = metric_summary(all_patch_oof, 'threshold_patch')
    bootstrap_ci = patient_cluster_bootstrap_ci(all_patch_oof, all_patient_oof, args.bootstrap, CFG.seed + 101)
    # Same OOF scores, no retraining or reselection: only the threshold is changed to 0.5.
    fixed_patches = [{**r, 'threshold_patch': 0.5} for r in all_patch_oof]
    fixed_patients = {agg: [{**r, 'threshold': 0.5} for r in rows]
                      for agg, rows in all_patient_oof.items()}
    fixed_metrics = {
        'patch_level': metric_summary(fixed_patches, 'threshold_patch'),
        'patient_level_by_aggregation': {agg: metric_summary(rows, 'threshold')
                                         for agg, rows in fixed_patients.items()}
    }
    fixed_bootstrap_ci = patient_cluster_bootstrap_ci(
        fixed_patches, fixed_patients, args.bootstrap, CFG.seed + 101)
    csvwrite(out / 'oof_patches_fixed_0_5.csv',
             [{**r, 'pred_fixed_0_5': int(r['prob_ehcc'] >= 0.5)} for r in all_patch_oof],
             ['patient_id', 'patch_index', 'label', 'prob_ehcc', 'threshold_patch', 'outer_fold', 'pred_fixed_0_5'])
    for agg, rows in all_patient_oof.items():
        csvwrite(out / f'oof_patients_{agg}_fixed_0_5.csv',
                 [{**r, 'pred_fixed_0_5': int(r['prob_ehcc'] >= 0.5)} for r in rows],
                 ['patient_id', 'label', 'prob_ehcc', 'n_patches', 'outer_fold', 'threshold', 'pred_fixed_0_5'])

    # Descriptive mean +/- SD across the five outer test folds, reported alongside pooled OOF estimates.
    fold_mean_sd = {}
    for level, selector in (
        ('patch', lambda f: f['patch_metrics']),
        ('patient_top20', lambda f: f['patient_metrics_by_aggregation'][PRIMARY_AGG])):
        fold_mean_sd[level] = {}
        for metric in METRICS:
            vals = [selector(f)[metric] for f in fold_summaries if selector(f).get(metric) is not None]
            fold_mean_sd[level][metric] = {'mean': float(np.mean(vals)),
                                           'sd': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                                           'fold_values': [float(v) for v in vals]}

    results = {
        'patient_manifest_hash': manifest_digest,
        'n_patients': 60, 'n_patches': len(all_patch_oof),
        'primary_patient_aggregation': PRIMARY_AGG,
        'folds': fold_summaries,
        'pooled_oof': {'patch_level': patch_metrics, 'patient_level_by_aggregation': patient_metrics},
        'outer_fold_mean_sd': fold_mean_sd,
        'bootstrap_confidence_intervals': bootstrap_ci,
        'fixed_0_5_sensitivity': {'pooled_oof': fixed_metrics,
                                  'bootstrap_confidence_intervals': fixed_bootstrap_ci},
        'run_fingerprint': run_fingerprint,
        'interpretation_note': (
            'Patients are the independent clinical units. Patch-level uncertainty uses a patient-clustered bootstrap. '
            'Thresholds were learned only from inner OOF predictions and held fixed on each outer test fold. '
            'Pooled OOF AUC combines predictions from different outer-fold models; fold-specific AUCs and their '
            'mean/SD are reported alongside pooled estimates because probability calibration can differ across folds. '
            'Bootstrap intervals condition on the fitted outer-fold models and selected thresholds.'
        )
    }
    dump_json(out / 'FINAL_METRICS.json', results)
    dump_json(out / 'FIXED_THRESHOLD_0_5_METRICS.json', results['fixed_0_5_sensitivity'])
    print('DONE nested CV:', out)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Print patient-level nested splits only; no training')
    parser.add_argument('--nested', action='store_true', help='Run nested cross-validation')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--backbone-lr', type=float, default=3e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--bootstrap', type=int, default=1000,
                        help='Number of stratified patient-bootstrap replicates for 95%% CIs')
    parser.add_argument('--max-train-steps', type=int, default=0,
                        help='SMOKE ONLY: truncated training cannot be used for reported performance')
    parser.add_argument('--no-save-models', action='store_true', help='Do not save outer model checkpoints')
    args = parser.parse_args()
    if not args.nested and not args.dry_run:
        parser.error('Supply --nested or --dry-run')
    if args.max_train_steps:
        raise RuntimeError('Truncated training is disabled for nested evaluation; use --dry-run for split checks.')
    if not 1 <= args.epochs <= 100:
        raise ValueError('epochs must be within 1..100')
    if args.bootstrap < 100:
        raise ValueError('Use at least 100 bootstrap replicates; 1000 or more is recommended for final reporting.')
    fit(args)


if __name__ == '__main__':
    main()
