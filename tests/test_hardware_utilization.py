#!/usr/bin/env python3
"""Regression tests for the four canonical hardware-utilization metrics."""
from __future__ import annotations

import contextlib
import gzip
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PROFILE = ROOT / 'weaveprof/pipeline/capture'
if str(PROFILE) not in sys.path:
    sys.path.insert(0, str(PROFILE))

import merge_hw_into_perfetto  # noqa: E402
from hardware_utilization import (  # noqa: E402
    CANONICAL_METRIC_IDS, evaluate_hardware_utilization,
    filter_utilization_evidence_to_window)


def counter(name: str, values, unit: str = 'Throughput %', step_ns: int = 1000):
    return {
        'name': name,
        'unit': unit,
        'samples': [[1000 + index * step_ns, value]
                    for index, value in enumerate(values)],
    }


def complete_counters():
    return [
        counter('GPU Throughput', [10, 20, 30]),
        counter('DRAM Read Bandwidth', [20, 30, 40]),
        counter('DRAM Write Bandwidth', [40, 10, 20]),
        counter('NVLink TX Requests User Data', [5, 6, 7]),
        counter('NVLink RX Requests User Data', [9, 3, 1]),
        counter('SMs Active', [50, 60, 70]),
    ]


class CanonicalSummaryTests(unittest.TestCase):
    def test_complete_nvidia_summary_has_exact_ids_and_values(self):
        summary, series = evaluate_hardware_utilization(
            complete_counters(), platform='nv', communication_scope='same_host')
        self.assertEqual(tuple(summary['metrics']), CANONICAL_METRIC_IDS)
        self.assertTrue(summary['all_available'])
        expected = {
            'compute_util': 20.0,
            'memory_util': 53.333333,
            'communication_util': 7.333333,
            'sm_active': 60.0,
        }
        for metric_id, value in expected.items():
            item = summary['metrics'][metric_id]
            self.assertEqual(item['semantic_id'], metric_id)
            self.assertEqual(item['display_name'], metric_id)
            self.assertEqual(item['value'], value)
            self.assertEqual(item['unit'], 'percent_of_peak')
            self.assertEqual(item['status'], 'available')
            self.assertIn(item['evidence_status'], ('measured', 'derived'))
            self.assertTrue(item['source'])
            self.assertEqual(len(series[metric_id]['samples']), 3)
        self.assertEqual(summary['metrics']['compute_util']['evidence_status'], 'measured')
        self.assertEqual(summary['metrics']['memory_util']['evidence_status'], 'derived')
        self.assertEqual(summary['metrics']['communication_util']['scope'], 'same_host')
        self.assertFalse(summary['metrics']['communication_util']['cross_host_claim_allowed'])

    def test_missing_evidence_is_null_and_unavailable_not_zero(self):
        summary, series = evaluate_hardware_utilization([], platform='nv')
        self.assertFalse(summary['all_available'])
        self.assertEqual(series, {})
        for metric_id in CANONICAL_METRIC_IDS:
            item = summary['metrics'][metric_id]
            self.assertIsNone(item['value'])
            self.assertIsNone(series.get(metric_id))
            self.assertEqual(item['evidence_status'], 'unavailable')
            self.assertEqual(item['status'], 'unavailable')
            self.assertTrue(item['reason'])
            self.assertTrue(item['source'])

    def test_measured_zero_remains_a_real_zero(self):
        counters = [
            counter('GPU Throughput', [0, 0, 0]),
            counter('DRAM Read Bandwidth', [0, 0, 0]),
            counter('DRAM Write Bandwidth', [0, 0, 0]),
            counter('NVLink TX Requests User Data', [0, 0, 0]),
            counter('SMs Active', [0, 0, 0]),
        ]
        summary, _series = evaluate_hardware_utilization(
            counters, platform='nv', communication_scope='same_host')
        for metric_id in CANONICAL_METRIC_IDS:
            item = summary['metrics'][metric_id]
            self.assertEqual(item['value'], 0.0)
            self.assertEqual(item['status'], 'available')
            self.assertIn(item['evidence_status'], ('measured', 'derived'))

    def test_non_percent_counter_is_not_coerced(self):
        counters = [
            counter('DRAM Throughput', [1, 2, 3], unit='GB/s'),
            counter('SMs Active', [10, 20, 30], unit='cycles'),
        ]
        summary, _series = evaluate_hardware_utilization(counters, platform='nv')
        self.assertIsNone(summary['metrics']['memory_util']['value'])
        self.assertIsNone(summary['metrics']['sm_active']['value'])
        self.assertIn('not percent_of_peak', summary['metrics']['memory_util']['reason'])
        self.assertIn('not percent_of_peak', summary['metrics']['sm_active']['reason'])

    def test_cross_host_rejects_nvlink_only(self):
        summary, _series = evaluate_hardware_utilization(
            complete_counters(), platform='nv', communication_scope='cross_host')
        communication = summary['metrics']['communication_util']
        self.assertIsNone(communication['value'])
        self.assertEqual(communication['evidence_status'], 'unavailable')
        self.assertIn('NIC/RDMA/fabric', communication['reason'])
        self.assertIn('NVLink/XGMI', communication['reason'])

    def test_cross_host_accepts_explicit_rdma_evidence(self):
        evidence = {
            'communication_util': {
                'value': 17.5,
                'source': 'mlx5_0 port_xmit_data utilization',
                'evidence_status': 'measured',
                'evidence_kind': 'rdma',
                'scope': 'cross_host',
            }
        }
        summary, _series = evaluate_hardware_utilization(
            [], platform='nv', communication_scope='cross_host',
            evidence_document={'utilization_evidence': evidence})
        communication = summary['metrics']['communication_util']
        self.assertEqual(communication['value'], 17.5)
        self.assertEqual(communication['status'], 'available')
        self.assertEqual(communication['evidence_kind'], 'rdma')
        self.assertTrue(communication['cross_host_claim_allowed'])
        self.assertEqual(communication['scope'], 'cross_host')

    def test_amd_pmc_never_becomes_utilization_percent(self):
        counters = [
            counter('SQ_BUSY_CU_CYCLES', [100, 200], unit='cycles'),
            counter('TCC_EA0_RDREQ_DRAM_sum', [10, 20], unit='requests'),
            counter('TCC_EA0_RDREQ_GMI_CREDIT_STALL_sum', [3, 4], unit='cycles'),
        ]
        summary, series = evaluate_hardware_utilization(
            counters, platform='amd', communication_scope='cross_host')
        self.assertEqual(series, {})
        for metric_id in CANONICAL_METRIC_IDS:
            item = summary['metrics'][metric_id]
            self.assertIsNone(item['value'])
            self.assertEqual(item['evidence_status'], 'unavailable')
        self.assertIn('raw counts', summary['metrics']['compute_util']['reason'])
        self.assertIn('NIC/RDMA/fabric', summary['metrics']['communication_util']['reason'])

    def test_filter_counters_strict_window(self):
        from hardware_utilization import filter_counters_to_window
        filtered = filter_counters_to_window(complete_counters(), 2000, 3000)
        self.assertEqual(len(filtered), 6)
        self.assertTrue(all(sample[0] in (2000, 3000)
                            for item in filtered for sample in item['samples']))

    def test_explicit_evidence_is_clipped_to_selected_window(self):
        evidence = {'utilization_evidence': {'communication_util': {
            'value': 99.0,
            'unit': 'percent_of_peak',
            'source': 'rdma:mlx5_0 port_xmit_data utilization',
            'evidence_status': 'measured',
            'evidence_kind': 'rdma',
            'scope': 'cross_host',
            'samples': [[900, 12.0], [1100, 17.0], [1300, 19.0], [2100, 88.0]],
        }}}
        clipped = filter_utilization_evidence_to_window(evidence, 1000, 2000)
        summary, _series = evaluate_hardware_utilization(
            [], platform='nv', communication_scope='cross_host',
            evidence_document=clipped)
        communication = summary['metrics']['communication_util']
        self.assertEqual(communication['value'], 18.0)
        self.assertEqual(communication['sample_count'], 2)

        outside = {'utilization_evidence': {'communication_util': dict(
            evidence['utilization_evidence']['communication_util'],
            samples=[[100, 11.0], [200, 12.0]])}}
        clipped_outside = filter_utilization_evidence_to_window(outside, 1000, 2000)
        unavailable, _series = evaluate_hardware_utilization(
            [], platform='nv', communication_scope='cross_host',
            evidence_document=clipped_outside)
        item = unavailable['metrics']['communication_util']
        self.assertIsNone(item['value'])
        self.assertEqual(item['evidence_status'], 'unavailable')
        self.assertIn('selected kernel window', item['reason'])

    def test_window_filter_preserves_auxiliary_tensor_and_simd_evidence(self):
        evidence = {'utilization_evidence': {
            'compute_util': {
                'value': 42.0, 'unit': 'percent_of_peak',
                'source': 'rocprofv3:MfmaUtil', 'evidence_status': 'derived',
                'native_metric': 'MfmaUtil', 'compute_pipe': 'matrix_core',
                'samples': [[900, 40.0], [1100, 42.0], [2100, 44.0]],
            },
            'tensor_util': {
                'value': 42.0, 'unit': 'percent_of_peak',
                'source': 'rocprofv3:MfmaUtil', 'evidence_status': 'derived',
                'native_metric': 'MfmaUtil', 'compute_pipe': 'matrix_core',
                'samples': [[900, 40.0], [1100, 42.0]],
            },
            'simd_util': {
                'value': 10.0, 'unit': 'percent_of_peak',
                'source': 'rocprofv3:SIMD_UTILIZATION*100',
                'evidence_status': 'derived', 'native_metric': 'SIMD_UTILIZATION',
                'compute_pipe': 'simd', 'samples': [[1000, 10.0], [1500, 30.0]],
            },
        }}
        clipped = filter_utilization_evidence_to_window(evidence, 1000, 2000)
        filtered = clipped['utilization_evidence']
        self.assertIn('tensor_util', filtered)
        self.assertIn('simd_util', filtered)
        self.assertEqual(filtered['tensor_util']['samples'], [[1100, 42.0]])
        self.assertEqual(filtered['simd_util']['samples'], [[1000, 10.0], [1500, 30.0]])
        self.assertTrue(filtered['tensor_util']['_window_filtered'])
        self.assertTrue(filtered['simd_util']['_window_filtered'])


class MergeIntegrationTests(unittest.TestCase):
    def test_merge_puts_same_four_metrics_in_final_trace(self):
        with tempfile.TemporaryDirectory(prefix='weaveprofiler-util-merge-') as td:
            root = Path(td)
            sw_name = (
                'trace__demomoe_decode__card-h20__bs8_sq4096_kv16384__fp8e4m3'
                '__grid78_warp12_ph6__sw.perfetto.json.gz')
            sw_path = root / sw_name
            with gzip.open(sw_path, 'wt', encoding='utf-8') as handle:
                json.dump({
                    'traceEvents': [{
                        'ph': 'X', 'name': 'demo', 'pid': 1, 'tid': 1,
                        'ts': 0.0, 'dur': 100.0,
                    }],
                    'otherData': {'shift_ns': 0},
                }, handle)

            events_path = root / 'events.json'
            events_path.write_text(json.dumps({
                'metadata': {'rank': 0, 'device': 0},
                'events': [{'start_ns': 1_000_000, 'end_ns': 1_100_000}],
            }), encoding='utf-8')

            counters = []
            for name, base in (
                    ('GPU Throughput', 10.0),
                    ('DRAM Read Bandwidth', 20.0),
                    ('DRAM Write Bandwidth', 25.0),
                    ('NVLink TX Requests User Data', 5.0),
                    ('SMs Active', 50.0)):
                counters.append({
                    'name': name,
                    'unit': 'Throughput %',
                    'samples': [[1_000_000 + index * 1000, base + index]
                                for index in range(121)],
                })
            hw_path = root / 'hw_metrics.json'
            hw_path.write_text(json.dumps({
                'schema': 'gpu-hw-metrics/v1',
                'platform': 'nv',
                'communication_scope': 'same_host',
                'device': 0,
                'frequency_hz': 1_000_000,
                'sampling_interval_ns': {'available': True, 'median': 1000},
                'kernel_launches': [{
                    'correlation_id': 7, 'device': 0, 'name': 'demo_kernel',
                    'start_ns': 1_000_000, 'end_ns': 1_120_000,
                }],
                'counters': counters,
            }), encoding='utf-8')

            out_dir = root / 'out'
            out_dir.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                rc = merge_hw_into_perfetto.main([
                    '--trace', str(sw_path), '--hw-metrics', str(hw_path),
                    '--events', str(events_path), '--out', str(out_dir),
                ])
            self.assertEqual(rc, 0)
            output = out_dir / sw_name.replace('__sw.perfetto', '__sw_hw.perfetto')
            with gzip.open(output, 'rt', encoding='utf-8') as handle:
                trace = json.load(handle)
            hardware = trace['otherData']['hardware_summary']
            self.assertEqual(set(hardware['metrics']), set(CANONICAL_METRIC_IDS))
            self.assertTrue(hardware['all_available'])
            canonical_names = {event['name'] for event in trace['traceEvents']
                               if event.get('cat') == 'hardware_utilization'}
            self.assertEqual(canonical_names, set(CANONICAL_METRIC_IDS))
            self.assertEqual(hardware['metrics']['compute_util']['value'], 70.0)
            self.assertGreaterEqual(
                trace['otherData']['hw_merge']['canonical_metric_event_count'], 400)


if __name__ == '__main__':
    unittest.main(verbosity=2)
