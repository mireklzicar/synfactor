import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).resolve().parents[1]/'benchmarks/docking_fullspace/tyk2_v2/collect_results.py'
SPEC=importlib.util.spec_from_file_location('tyk2_collect',SOURCE)
collector=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(collector)


class Tyk2CollectTests(unittest.TestCase):
    def setUp(self):
        temporary=TemporaryDirectory();self.addCleanup(temporary.cleanup);self.runs=Path(temporary.name)
        self.proof=dict(lineage_sha256='lineage',parent_run_id='portfolio',destination_run_id='branch',
                        inherited_calls=4000,development_fork=True,independent_replicate=False)

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

    def cell(self,name,policy,proof=None,with_evaluation=True):
        root=self.runs/'cells'/name
        config=dict(target='tyk2',seed=0,run_id=name,policy=policy,budget=10000,warm_charged=1000,
                    batch_size=1000,grid_sha256='grid',warm={'csv_sha256':'warm'},druglike_filter=False)
        self.write(root/'config.json',config)
        self.write(root/'state.json',dict(history=[{}]*4000,rounds=[]))
        if not with_evaluation:
            return root
        cells,flat=[],[]
        for method in sorted(collector.EXPECTED):
            metrics={key:None for key in collector.NUMBERS}
            metrics.update(eligible_molecules=3900,top100_raw_mean=-8.,GPDiv_value=7.,
                           GPDiv_raw_mean=-7.,GPDiv_selected_count=100)
            evidence={'config_sha256':collector.load(root/'config.json')[1]}
            if proof is not None and method=='synfactor_v2':evidence['lineage']=proof
            cells.append(dict(method=method,eligible_molecules=3900,scores={'top100_raw_mean':-8.},
                              portfolios=[{'metric':'GPDiv','value':7.}],identity_evidence=evidence))
            flat.append(dict(method=method,charged_calls=4000,**metrics,GPDiv_status='greedy'))
        folder=root/'evaluation_4k'
        self.write(folder/'summary.json',dict(target='tyk2',seed=0,charged_calls=4000,grid_sha256='grid',
             warm_sha256='warm',rdkit='test',fingerprint='strict test',comparison_verified=True,cells=cells))
        collector.write_csv(folder/'comparison.csv',flat)
        return root

    def lineage(self,cell,*_):
        return self.proof if cell.name=='branch' else None

    def test_branch_without_evaluation_does_not_replace_core_raw(self):
        self.cell('raw','raw');self.cell('portfolio','portfolio');self.cell('branch','raw',with_evaluation=False)
        with patch.object(collector,'lineage_identity',side_effect=self.lineage):
            rows,evidence=collector.collect(self.runs)
        self.assertEqual({r['method'] for r in rows},set(collector.CORE_METHODS))
        self.assertEqual(len(evidence['policy_progress']),3)

    def test_branch_evaluation_requires_verified_lineage(self):
        self.cell('raw','raw');self.cell('portfolio','portfolio');self.cell('branch','raw')
        with patch.object(collector,'lineage_identity',side_effect=self.lineage):
            with self.assertRaisesRegex(ValueError,'lineage evidence'):
                collector.collect(self.runs)

    def test_verified_branch_is_separate_and_control_repeats_merge(self):
        self.cell('raw','raw');self.cell('portfolio','portfolio');self.cell('branch','raw',proof=self.proof)
        with patch.object(collector,'lineage_identity',side_effect=self.lineage):
            rows,_=collector.collect(self.runs)
        self.assertEqual({r['method'] for r in rows},set(collector.LABELS))
        self.assertEqual(len(rows),7)
        for row in rows:
            if not row['method'].startswith('synfactor_v2_'):
                self.assertEqual(row['evidence_count'],3)

    def test_lineage_proof_must_match_budget_and_fork_identity(self):
        collector.require_lineage_proof(dict(self.proof),self.proof)
        for key,value in [('inherited_calls',3000),('parent_run_id','other'),('independent_replicate',True)]:
            with self.assertRaisesRegex(ValueError,'lineage evidence'):
                collector.require_lineage_proof({**self.proof,key:value},self.proof)


if __name__=='__main__':
    unittest.main()
