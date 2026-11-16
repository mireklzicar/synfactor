import json
import csv
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rdkit import DataStructs

from experiments.docking.live_evaluate import (molecular_rows, read_history, strict_conflicts,
    warm_evidence, verify_comparison_identity)


class LiveEvaluationTests(unittest.TestCase):
    def test_strict_similarity_boundary_is_conflict(self):
        a, b, c = [DataStructs.ExplicitBitVect(2048) for _ in range(3)]
        for i in range(6): a.SetBit(i)
        for i in [0, 1, 2, 6, 7, 8, 9]: b.SetBit(i)
        c.SetBit(20)
        self.assertEqual(DataStructs.TanimotoSimilarity(a, b), .3)
        class Generator:
            def __init__(self): self.values=iter([a,b,c])
            def GetFingerprint(self, mol): return next(self.values)
        with patch('experiments.docking.live_evaluate.rdFingerprintGenerator.GetMorganGenerator', return_value=Generator()):
            graph=strict_conflicts(['C', 'CC', 'CCC'])
        self.assertTrue(graph[0,1]); self.assertTrue(graph[1,0])
        self.assertFalse(graph[0,2]); self.assertFalse(graph.diagonal().any())

    def test_gate_sees_original_forms_failed_calls_excluded_best_molecule_retained(self):
        class Gate:
            def __init__(self): self.seen=[]
            def evaluate(self, smiles): self.seen.append(smiles); return None
        gate=Gate()
        rows=[dict(product_id='a',smiles='C(C)O',score=-4,status='valid',iteration=-1),
              dict(product_id='b',smiles='CCO',score=-6,status='valid',iteration=0),
              dict(product_id='c',smiles='CC',score=-999,status='failed',iteration=0)]
        selected,audit=molecular_rows(rows,gate)
        self.assertEqual(gate.seen,['C(C)O','CCO'])
        self.assertEqual(selected[0]['product_id'],'b')
        self.assertEqual(selected[0]['score'],-6)
        self.assertEqual(audit['charged_input_rows'],3)
        self.assertEqual(audit['duplicate_molecules_removed'],1)
        self.assertEqual(audit['exclusions']['docking_status_not_valid'],1)

    def test_history_charges_warm_and_failures_and_rejects_duplicates(self):
        rows=[dict(product_id='a',smiles='C',score=-4,status='valid',iteration=-1),
              dict(product_id='b',smiles=None,score=None,status='failed',iteration=0)]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.json';path.write_text(json.dumps({'history':rows}))
            loaded,_=read_history(path,expected_calls=2,expected_warm=1)
            self.assertEqual(len(loaded),2)
            with self.assertRaisesRegex(ValueError,'charged history'):
                read_history(path,expected_calls=1,expected_warm=1)
            rows[1]['product_id']='a';path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError,'Duplicate charged'):
                read_history(path,expected_calls=2,expected_warm=1)

    def fixture_comparison(self, root):
        cohort=root/'benchmark/artifacts/seeds/freedom_296b/seed_0/selected_10k/subselections'
        cohort.mkdir(parents=True)
        (cohort/'budget_01000.ids').write_text(''.join(f'w{i}\n' for i in range(1000)))
        with (cohort/'budget_01000_candidates.csv').open('w',newline='') as handle:
            writer=csv.writer(handle); writer.writerow(['product_id','smiles'])
            writer.writerows((f'w{i}','CC') for i in range(1000))
        matrix=root/'benchmark/artifacts/dock12_seed_screen/freedom/seed_0/results_long.csv'
        matrix.parent.mkdir(parents=True)
        with matrix.open('w',newline='') as handle:
            writer=csv.writer(handle);writer.writerow(['target_id','product_id','valid','score'])
            writer.writerows(('tyk2',f'w{i}','true' if i else 'false','-3.4000' if i else '') for i in range(1000))
        warm,hashes=warm_evidence(root/'benchmark','tyk2',0)
        self.assertEqual(warm[0]['score'],None)
        self.assertEqual(warm[1]['score'],-3.4)
        cell=root/'cell';cell.mkdir()
        protocol=dict(target='tyk2',seed=0,grid_sha256='grid')
        config=dict(target='tyk2',seed=0,budget=10000,warm_charged=1000,batch_size=1000,
                    warm=hashes,grid_sha256='grid',oracle_protocol_sha256=hashlib.sha256(
                        json.dumps(protocol,sort_keys=True).encode()).hexdigest())
        (cell/'config.json').write_text(json.dumps(config))
        (cell/'remote_preflight.json').write_text(json.dumps(dict(protocol,preflight_only=True,free_disk_gib=123)))
        history=list(warm)
        for iteration in range(1,10):
            folder=cell/f'batch_{iteration:02d}';folder.mkdir()
            rows=[dict(product_id=f'b{iteration}_{i}',smiles='CC',score=-4.,valid=True) for i in range(1000)]
            proposal=dict(target='tyk2',seed=0,run_id='run',batch_id=folder.name,
                          product_ids=[r['product_id'] for r in rows])
            response={k:proposal[k] for k in ('target','seed','run_id','batch_id')}
            response.update(rows=rows,protocol=protocol,smoke_only=False,request_sha256=hashlib.sha256(
                json.dumps(proposal,sort_keys=True).encode()).hexdigest())
            (folder/'proposal.json').write_text(json.dumps(proposal))
            (folder/'normalized.json').write_text(json.dumps(response))
            history.extend(dict(product_id=r['product_id'],smiles=r['smiles'],score=r['score'],status='valid',
                                iteration=iteration,source='live_glide',charged=True) for r in rows)
        return cell/'state.json',history

    def test_comparison_binds_warm_sources_target_seed_and_actual_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,rows=self.fixture_comparison(root)
            verify=lambda **kw:verify_comparison_identity(path,rows,kw.pop('target','tyk2'),
                kw.pop('seed',0),root/'benchmark',**kw)
            grid,warm,evidence=verify()
            self.assertEqual(grid,'grid');self.assertEqual(len(evidence['batch_response_sha256']),9)
            for kwargs in ({'target':'brd4'},{'seed':1},{'grid_assertion':'other'},{'warm_assertion':'other'}):
                with self.assertRaises(ValueError):verify(**kwargs)
            rows[0]['score']=-99
            with self.assertRaisesRegex(ValueError,'warm rows/source hashes'):verify()
            rows[0]['score']=None
            rows[-1]['score']=-99
            with self.assertRaisesRegex(ValueError,'observations differ'):verify()
            rows[-1]['score']=-4.
            matrix=root/'benchmark/artifacts/dock12_seed_screen/freedom/seed_0/results_long.csv'
            matrix.write_text(matrix.read_text().replace('-3.4000','-3.40'))
            with self.assertRaisesRegex(ValueError,'warm rows/source hashes'):verify()

    def test_comparison_rejects_changed_protocol_and_unbound_standalone_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,rows=self.fixture_comparison(root)
            verify=lambda:verify_comparison_identity(path,rows,'tyk2',0,root/'benchmark')
            response=path.parent/'batch_09/normalized.json'
            value=json.loads(response.read_text());value['protocol']['grid_sha256']='other'
            response.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'oracle protocol'):verify()
            (path.parent/'config.json').unlink()
            with self.assertRaisesRegex(ValueError,'adjacent config.json'):verify()


if __name__=='__main__':unittest.main()
