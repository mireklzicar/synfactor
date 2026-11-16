import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.docking import tyk2_evaluate as e


class Tyk2EvaluateTests(unittest.TestCase):
    def setUp(self):
        directory=TemporaryDirectory();self.addCleanup(directory.cleanup)
        self.root=Path(directory.name)

    def ledger(self, offset=1, rounds=2):
        return [dict(product_id=str(i),iteration=-1 if i<1000 else i//1000-1+offset,
                     smiles='CC',score=-float(i%10),status='valid',charged=True)
                for i in range(1000*(rounds+1))]

    def dump(self,path,payload):
        path.write_text(json.dumps(payload))

    def evidence(self):
        rows=self.ledger(rounds=1)
        warm=rows[:1000]
        hashes={'csv_sha256':'warm'}
        protocol=dict(target='tyk2',seed=0,grid_sha256='grid')
        config=dict(target='tyk2',seed=0,budget=10000,warm_charged=1000,batch_size=1000,
                    grid_sha256='grid',warm=hashes,oracle_protocol_sha256=e.protocol_hash(protocol))
        proposal=dict(target='tyk2',seed=0,run_id='test',batch_id='batch_01',
                      product_ids=[r['product_id'] for r in rows[1000:]])
        response={k:proposal[k] for k in ('target','seed','run_id','batch_id')}
        import hashlib
        response.update(request_sha256=hashlib.sha256(json.dumps(proposal,sort_keys=True).encode()).hexdigest(),
                        protocol=protocol,rows=[dict(product_id=r['product_id'],smiles=r['smiles'],
                            score=r['score'],valid=True) for r in rows[1000:]])
        (self.root/'batch_01').mkdir()
        for name,value in [('state.json',{'history':rows}),('config.json',config),
                           ('remote_preflight.json',protocol),('batch_01/proposal.json',proposal),
                           ('batch_01/normalized.json',response)]:
            self.dump(self.root/name,value)
        return rows,warm,hashes

    def test_charged_prefix_maps_old_and_new_iteration_conventions(self):
        new=e.checked_prefix(self.ledger(1),1,adaptive_offset=1)
        old=e.checked_prefix(self.ledger(0),1,adaptive_offset=0)
        self.assertEqual([r['product_id'] for r in old],[r['product_id'] for r in new])
        self.assertEqual(len(new),2000)

    def test_charge_validation_rejects_missing_failures_duplicate_ids_and_gaps(self):
        rows=self.ledger()
        with self.assertRaisesRegex(ValueError,'complete'):
            e.checked_prefix(rows[:-1],1,adaptive_offset=1)
        rows[-1]['product_id']='0'
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            e.checked_prefix(rows,1,adaptive_offset=1)
        rows=self.ledger();rows[-1]['iteration']=3
        with self.assertRaisesRegex(ValueError,'coverage'):
            e.checked_prefix(rows,1,adaptive_offset=1)

    def test_partial_identity_accepts_only_completed_response_bound_prefix(self):
        rows,warm,hashes=self.evidence()
        with patch.object(e,'warm_evidence',return_value=(warm,hashes)):
            actual,_,proof=e.verify_partial(self.root/'state.json',1,self.root)
        self.assertEqual(rows,actual)
        self.assertEqual(set(proof['response_sha256']),{'batch_01'})

    def test_partial_identity_rejects_changed_warm(self):
        rows,warm,hashes=self.evidence()
        altered=[dict(r) for r in warm];altered[0]['score']=-999.
        with patch.object(e,'warm_evidence',return_value=(altered,hashes)):
            with self.assertRaisesRegex(ValueError,'Warm'):
                e.verify_partial(self.root/'state.json',1,self.root)

    def test_partial_identity_cannot_ignore_invalid_development_lineage(self):
        _,warm,hashes=self.evidence()
        self.dump(self.root/'lineage.json',{})
        with patch.object(e,'warm_evidence',return_value=(warm,hashes)):
            with self.assertRaisesRegex(ValueError,'lineage'):
                e.verify_partial(self.root/'state.json',1,self.root)

    def test_partial_identity_rejects_changed_batch_score(self):
        rows,warm,hashes=self.evidence()
        rows[-1]['score']=-999.
        self.dump(self.root/'state.json',{'history':rows})
        with patch.object(e,'warm_evidence',return_value=(warm,hashes)):
            with self.assertRaisesRegex(ValueError,'observations'):
                e.verify_partial(self.root/'state.json',1,self.root)

    def test_partial_identity_rejects_grid_and_protocol_change(self):
        _,warm,hashes=self.evidence()
        self.dump(self.root/'remote_preflight.json',dict(target='tyk2',seed=0,grid_sha256='wrong'))
        with patch.object(e,'warm_evidence',return_value=(warm,hashes)):
            with self.assertRaisesRegex(ValueError,'protocol'):
                e.verify_partial(self.root/'state.json',1,self.root)

    def test_gpdiv_rejects_exact_similarity_threshold(self):
        rows=[dict(product_id=str(i),smiles='CC',score=-3+i) for i in range(3)]
        with patch.object(e.DataStructs,'BulkTanimotoSimilarity',side_effect=[[.30],[.29]]):
            result=e.greedy_portfolio(rows,k=2)
        self.assertEqual(result['selected_product_ids'],['0','2'])
        self.assertEqual(result['value'],2.)

    def test_all_valid_has_no_druglike_gate_and_deduplicates_best_score(self):
        rows=[dict(product_id='a',smiles='C'*40,status='valid',score=-5,iteration=1),
              dict(product_id='b',smiles='C'*40,status='valid',score=-6,iteration=1),
              dict(product_id='c',smiles='CC',status='failed',score=None,iteration=1)]
        result,unique=e.evaluate(rows,2000,True,False,3,self.root)
        self.assertEqual(len(unique),1)
        self.assertEqual(unique[0]['product_id'],'b')
        self.assertIsNone(unique[0]['druglike_failure'])
        self.assertEqual(result['audit']['charged_calls'],2000)
        self.assertEqual(result['scores']['top1_raw_mean'],-6.)


if __name__=='__main__':
    unittest.main()
