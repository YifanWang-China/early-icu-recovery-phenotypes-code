import ast
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
_workspace = tempfile.TemporaryDirectory()
os.environ['ICU_WORKSPACE'] = _workspace.name

import numpy as np
import pandas as pd
import torch
from dataset1_firstpass_phenotype_discovery import BG_CHANNELS, summary_stats
from dataset1_dataset2_temporal_transport import SHARED_CHANNELS, interpolate_channel
from frozen_discovery_phenotype_transport import normalize_from_train, train_model, predict, set_seed
from revision_20261005_analysis import fit_discovery, teacher_predict, STRICT


class PipelineTests(unittest.TestCase):
    def test_summary_uses_time_order(self):
        result = summary_stats([2, 0, 1], [6, 2, 4])
        self.assertEqual(result['first'], 2)
        self.assertEqual(result['last'], 6)
        self.assertEqual(result['delta'], 4)
        self.assertAlmostEqual(result['slope'], 2)
        self.assertEqual(summary_stats([], [])['count'], 0)

    def test_grid_averages_duplicate_times_without_extrapolation(self):
        values, mask = interpolate_channel([(1, 2), (1, 4), (2, 5)])
        self.assertTrue(np.isnan(values[0]))
        self.assertEqual(values[2], 3)
        self.assertEqual(values[3], 4)
        self.assertEqual(values[4], 5)
        np.testing.assert_array_equal(np.flatnonzero(mask), [2, 4])

    def test_normalization_uses_training_observations_only(self):
        train = np.ones((2, 13, 8), dtype=np.float32)
        masks = np.ones_like(train)
        train[0, 0, 0] = 999
        masks[0, 0, 0] = 0
        external = np.full((1, 13, 8), 100, dtype=np.float32)
        normalized, means, scales = normalize_from_train(train, masks, external)
        np.testing.assert_allclose(means, 1)
        np.testing.assert_allclose(scales, 1)
        np.testing.assert_allclose(normalized, 99)

    def test_discovery_does_not_use_outcomes(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(60, 16 * 9))
        stats = ['count', 'first', 'last', 'min', 'max', 'mean', 'std', 'delta', 'slope']
        df = pd.DataFrame(x, columns=[f'{c}_{s}' for c in BG_CHANNELS for s in stats])
        df['label72'] = np.arange(60) % 2
        labels, _, _, model, _ = fit_discovery(df, 'test', kfix=3)
        altered = df.copy()
        altered['label72'] = 1 - altered.label72
        other, _, _, _, _ = fit_discovery(altered, 'test', kfix=3)
        np.testing.assert_array_equal(labels, other)
        np.testing.assert_array_equal(labels, teacher_predict(model, df))

    def test_assignment_train_predict(self):
        set_seed(42)
        rng = np.random.default_rng(4)
        values = rng.normal(size=(12, 13, 8)).astype(np.float32)
        masks = np.ones_like(values)
        labels = np.arange(12) % 3
        model, history = train_model(values, masks, labels, 12, 1, .001, 2, .2, torch.device('cpu'))
        predicted, probabilities = predict(model, values, masks, values[:3], masks[:3], 3, torch.device('cpu'))
        self.assertEqual(predicted.shape, (3,))
        self.assertEqual(probabilities.shape, (3, 3))
        np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
        self.assertTrue(np.isfinite(history[0]['recon_loss']))
        self.assertEqual(SHARED_CHANNELS[4], 'hb')

    def test_assay_names_and_procedure_groups(self):
        self.assertIn('血红蛋白浓度', STRICT['hb'])
        self.assertNotIn('糖化血红蛋白', STRICT['hb'])
        # Load the classifier without executing the analysis that reads clinical files.
        source = (ROOT / 'scripts/revision_20261005_extended_checks.py').read_text(encoding='utf-8')
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'group')
        namespace = {'re': re}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<procedure>', 'exec'), namespace)
        group = namespace['group']
        self.assertEqual(group('冠状动脉旁路移植术'), 'CABG')
        self.assertEqual(group('心脏移植术'), 'Heart transplantation')
        self.assertEqual(group('冠状动脉旁路移植术及瓣膜置换术'), 'CABG + valve')


if __name__ == '__main__':
    unittest.main()
