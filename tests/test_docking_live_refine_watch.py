from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from experiments.docking.live_refine_watch import METHODS, initial_pending, tick, verified_result
from experiments.docking.live_monitor import sha, write


class RefineWatcherTests(unittest.TestCase):
    def fixture(self, root):
        cell=root/'sf_example';cell.mkdir()
        (cell/'CELL_DONE').write_text('done')
        write(cell/'config.json',{})
        write(cell/'state.json',{})
        evaluation=cell/'evaluation_live';evaluation.mkdir()
        (evaluation/'comparison.csv').write_text('complete')
        portfolio=dict(metric='OPDiv',feasible_lower_bound=6.,upper_bound=6.)
        write(evaluation/'summary.json',dict(target='tyk2',seed=0,cells=[
            dict(method=m,domains={'all-valid':{},'druglike-v1':dict(portfolios=[portfolio])}) for m in METHODS]))
        outputs=cell/'refinement';outputs.mkdir();sources=outputs/'sources';sources.mkdir()
        original=root/'source.py';original.write_text('source')
        digest=sha(original);(sources/(digest+'.py')).write_text('source')
        for method in METHODS:
            csv=evaluation/f'{method}__all-valid__molecules.csv.gz';csv.write_text(method)
            write(outputs/f'{method}__all-valid.json',dict(input_sha256=sha(csv),
                selected_product_ids=list(map(str,range(100))),value=7.,upper_bound=7.,certified=True,
                status='optimal',wall_seconds=2.,source_sha256={str(original):digest}))
        write(root/'monitor_state.json',dict(process={'pid':1},evaluations={}))
        args=SimpleNamespace(campaign=root,frozen_root=root/'frozen',eval_python=Path('/python'),
                             benchmark_repo=root/'benchmark',expected_targets=12)
        state=dict(jobs={},status='watching')
        return args,state,cell

    def test_existing_result_requires_same_csv_and_source_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,state,cell=self.fixture(Path(tmp))
            output=cell/'refinement/synfactor__all-valid.json'
            molecules=cell/'evaluation_live/synfactor__all-valid__molecules.csv.gz'
            self.assertEqual(verified_result(output,molecules)['value'],7.)
            molecules.write_text('changed')
            with self.assertRaisesRegex(ValueError,'input hash'):verified_result(output,molecules)
            molecules.write_text('synfactor')
            next((cell/'refinement/sources').glob('*.py')).write_text('changed')
            with self.assertRaisesRegex(ValueError,'source snapshot'):verified_result(output,molecules)

    def test_finished_campaign_recovers_four_outputs_without_launching(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,state,cell=self.fixture(Path(tmp))
            with patch('experiments.docking.live_refine_watch.process_state',return_value='gone'), \
                 patch('experiments.docking.live_refine_watch.evaluation_complete',return_value=True), \
                 patch('experiments.docking.live_refine_watch.subprocess.Popen') as launch:
                self.assertTrue(tick(args,state));launch.assert_not_called()
            self.assertEqual(len(state['jobs']),4)
            self.assertTrue((args.campaign/'refined_progress.md').exists())

    def test_failed_child_is_not_retried_and_only_local_refiner_is_launched(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,state,cell=self.fixture(Path(tmp))
            (cell/'refinement/synfactor__all-valid.json').unlink()
            with patch('experiments.docking.live_refine_watch.process_state',return_value='gone'), \
                 patch('experiments.docking.live_refine_watch.process_identity',return_value={'pid':55}), \
                 patch('experiments.docking.live_refine_watch.evaluation_complete',return_value=True), \
                 patch('experiments.docking.live_refine_watch.subprocess.Popen',return_value=SimpleNamespace(pid=55)) as launch:
                self.assertFalse(tick(args,state))
                self.assertTrue(tick(args,state))
                self.assertTrue(tick(args,state))
                self.assertEqual(launch.call_count,1)
                command=launch.call_args.args[0]
                self.assertEqual(command[:3],['/python','-m','experiments.docking.live_refine'])
                self.assertEqual(launch.call_args.kwargs['cwd'],args.frozen_root)
            self.assertEqual(state['jobs']['sf_example/synfactor']['status'],'failed_no_retry')

    def test_pending_initial_evaluation_prevents_premature_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            cell=Path(tmp)/'cell';cell.mkdir()
            self.assertTrue(initial_pending([cell],dict(evaluations={})))
            monitor=dict(evaluations={'cell':dict(status='running',process={'pid':5})})
            with patch('experiments.docking.live_refine_watch.process_state',return_value='running'):
                self.assertTrue(initial_pending([cell],monitor))
            with patch('experiments.docking.live_refine_watch.process_state',return_value='gone'):
                self.assertFalse(initial_pending([cell],monitor))


if __name__=='__main__':
    unittest.main()
