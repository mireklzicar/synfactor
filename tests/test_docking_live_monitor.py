"""Supervisor tests use synthetic files and mocks; never contact AWS or SSH."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from tempfile import TemporaryDirectory
from experiments.docking import live_monitor as m

def dump(path, value):
    path.write_text(json.dumps(value))

def make_summary():
    return dict(comparison_verified=True, history_sha256='history', grid_sha256='grid', warm_sha256='csv', cells=[dict(method=method, domains={domain: {'portfolios': [dict(metric='OPDiv', feasible_lower_bound=8.0, upper_bound=8.0)]} for domain in ('all-valid', 'druglike-v1')}) for method in sorted(m.METHODS)])

def owned_args(tmp_path):
    audit = tmp_path / 'ownership.json'
    dump(audit, dict(instance_id='i-owned', profile='test', region='test-region', public_ip='test-ip', previous_state='stopped', start_requested_by_this_campaign=True))
    args = SimpleNamespace(instance_audit=audit, host='ubuntu@test-ip', key=Path('test-key'))
    monitor = dict(process={'pid': 123}, binding={'instance_audit_sha256': m.sha(audit)})
    return (args, monitor)

def aws_instance(state):
    return {'Reservations': [{'Instances': [{'State': {'Name': state}, 'PublicIpAddress': 'test-ip'}]}]}

class LiveMonitorTests(unittest.TestCase):

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_process_identity_handles_parentheses_and_pid_reuse(self):
        tmp_path = self.tmp_path
        folder = tmp_path / '123'
        folder.mkdir()
        boot = tmp_path / 'sys/kernel/random'
        boot.mkdir(parents=True)
        (boot / 'boot_id').write_text('boot-a')
        (folder / 'stat').write_text('123 (odd ) command) S ' + '0 ' * 18 + '456 0')
        (folder / 'cmdline').write_bytes(b'python\x00-m\x00experiments.docking.live\x00')
        identity = m.process_identity(123, tmp_path)
        self.assertTrue(identity['start_ticks'] == 456)
        self.assertTrue(identity['boot_id'] == 'boot-a')
        with patch.object(m, 'process_identity', return_value={**identity, 'start_ticks': 457}):
            self.assertTrue(m.process_state(identity) == 'pid_reused')
        with patch.object(m, 'process_identity', return_value=None):
            self.assertTrue(m.process_state(identity) == 'gone')

    def test_cell_validation_rejects_duplicate_budget_and_grid(self):
        tmp_path = self.tmp_path
        config = dict(target='tyk2', seed=0, budget=10000, warm_charged=1000, grid_sha256='grid', warm={'ids_sha256': 'ids', 'csv_sha256': 'csv'})
        preflight = dict(target='tyk2', seed=0, grid_sha256='grid', warm_ordered_ids_sha256='ids')
        history = [dict(product_id=str(i), iteration=-1 if i < 1000 else 1) for i in range(10000)]
        dump(tmp_path / 'config.json', config)
        dump(tmp_path / 'remote_preflight.json', preflight)
        dump(tmp_path / 'state.json', {'history': history})
        (tmp_path / 'CELL_DONE').touch()
        self.assertTrue(m.validate_cell(tmp_path) == config)
        history[-1]['product_id'] = '0'
        dump(tmp_path / 'state.json', {'history': history})
        with self.assertRaisesRegex(ValueError, 'unique'):
            m.validate_cell(tmp_path)
        history[-1]['product_id'] = '9999'
        dump(tmp_path / 'state.json', {'history': history})
        dump(tmp_path / 'remote_preflight.json', {**preflight, 'grid_sha256': 'different'})
        with self.assertRaisesRegex(ValueError, 'Grid'):
            m.validate_cell(tmp_path)

    def test_evaluation_requires_complete_four_method_matching_artifact(self):
        tmp_path = self.tmp_path
        summary = make_summary()
        config = {'grid_sha256': 'grid', 'warm': {'csv_sha256': 'csv'}}
        dump(tmp_path / 'summary.json', summary)
        self.assertTrue(not m.evaluation_complete(tmp_path, 'history', config))
        (tmp_path / 'comparison.csv').touch()
        self.assertTrue(m.evaluation_complete(tmp_path, 'history', config))
        self.assertTrue(not m.evaluation_complete(tmp_path, 'different', config))
        summary['cells'].pop()
        dump(tmp_path / 'summary.json', summary)
        self.assertTrue(not m.evaluation_complete(tmp_path, 'history', config))

    def test_bound_comparisons_do_not_claim_float_ties(self):
        summary = make_summary()
        cp = next((cell for cell in summary['cells'] if cell['method'] == 'synfactor'))
        cp['domains']['all-valid']['portfolios'][0]['feasible_lower_bound'] += 1e-08
        self.assertTrue(all((row['claim'] == 'not_separated_by_bounds' for row in m.portfolio_comparisons(summary))))
        cp['domains']['all-valid']['portfolios'][0]['feasible_lower_bound'] = 8.01
        self.assertTrue(sum((row['claim'] == 'certified_synfactor_win' for row in m.portfolio_comparisons(summary))) == 3)

    def test_restoration_refuses_active_campaign_and_changed_ownership(self):
        tmp_path = self.tmp_path
        args, monitor = owned_args(tmp_path)
        with patch.object(m, 'process_state', return_value='running'), patch.object(m, 'aws') as aws:
            with self.assertRaisesRegex(ValueError, 'alive'):
                m.restore_instance(args, monitor)
            aws.assert_not_called()
        args.instance_audit.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'changed'):
            m.restore_instance(args, monitor)

    def test_restoration_never_stops_on_uncertain_remote(self):
        tmp_path = self.tmp_path
        args, monitor = owned_args(tmp_path)
        with patch.object(m, 'process_state', return_value='gone'), patch.object(m, 'aws', return_value=aws_instance('running')) as aws, patch.object(m, 'remote_idle', return_value={'inspection_complete': False, 'idle': False}), patch.object(m.time, 'sleep'):
            result = m.restore_instance(args, monitor)
        self.assertTrue(result['status'] == 'not_stopped_remote_not_proven_idle')
        self.assertTrue([call.args[1] for call in aws.call_args_list] == ['describe-instances'])

    def test_restoration_needs_two_idle_checks_and_verifies_stopped(self):
        tmp_path = self.tmp_path
        args, monitor = owned_args(tmp_path)
        with patch.object(m, 'process_state', return_value='gone'), patch.object(m, 'aws', side_effect=[aws_instance('running'), {'StoppingInstances': []}, aws_instance('stopped')]) as aws, patch.object(m, 'remote_idle', return_value={'inspection_complete': True, 'idle': True}) as idle, patch.object(m.time, 'sleep'):
            result = m.restore_instance(args, monitor)
        self.assertTrue(result['status'] == 'stopped_verified')
        self.assertTrue(idle.call_count == 2)
        self.assertTrue([call.args[1] for call in aws.call_args_list] == ['describe-instances', 'stop-instances', 'describe-instances'])
        self.assertTrue(all((call.args[-1] == 'i-owned' for call in aws.call_args_list)))

    def test_remote_inspection_program_is_valid_python(self):
        compile(m.REMOTE_IDLE, '<remote-idle>', 'exec')
        self.assertTrue(".split('\\0')" in m.REMOTE_IDLE)
if __name__ == '__main__':
    unittest.main()
