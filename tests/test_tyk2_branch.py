import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.docking import tyk2_branch as b
from experiments.docking import tyk2_evaluate as e


class Tyk2BranchTests(unittest.TestCase):
    def setUp(self):
        temporary=TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.parent=self.root/'parent_portfolio';self.parent.mkdir()
        self.out=self.root/'child_explore_then_raw'
        self.protocol=dict(target='tyk2',seed=0,grid_sha256='grid')
        self.hashes=dict(csv_sha256='warm')
        self.config=dict(target='tyk2',seed=0,run_id=self.parent.name,policy='portfolio',
                         budget=10000,warm_charged=1000,batch_size=1000,warm=self.hashes,
                         grid_sha256='grid',oracle_protocol_sha256=e.protocol_hash(self.protocol))
        self.history=[dict(product_id=str(i),smiles='CC',score=-float(i%7),status='valid',
                           iteration=-1 if i<1000 else i//1000,charged=True)
                      for i in range(5000)]
        self.rounds=[dict(iteration=i,calls=1000*(i+1),search={'tag':f'round{i}'}) for i in range(1,5)]
        self.rounds[-1]['later_only_secret']='not_for_child'
        self.write(self.parent/'state.json',dict(history=self.history,rounds=self.rounds))
        self.write(self.parent/'config.json',self.config)
        self.write(self.parent/'remote_preflight.json',self.protocol)
        for iteration in range(1,5):
            folder=self.parent/f'batch_{iteration:02d}';folder.mkdir()
            observations=self.history[1000*iteration:1000*(iteration+1)]
            proposal=dict(target='tyk2',seed=0,run_id=self.parent.name,batch_id=folder.name,
                          product_ids=[r['product_id'] for r in observations])
            response={k:proposal[k] for k in ('target','seed','run_id','batch_id')}
            response.update(request_sha256=hashlib.sha256(json.dumps(proposal,sort_keys=True).encode()).hexdigest(),
                            protocol=self.protocol,rows=[dict(product_id=r['product_id'],smiles=r['smiles'],
                                score=r['score'],valid=True) for r in observations])
            self.write(folder/'proposal.json',proposal);self.write(folder/'normalized.json',response)
            self.write(folder/'search.json',{'round':iteration})
            self.write(folder/'never_copy_model.json',{'extra':True})
        self.warm_patch=patch.object(e,'warm_evidence',return_value=(self.history[:1000],self.hashes))
        self.warm_patch.start();self.addCleanup(self.warm_patch.stop)

    def write(self,path,value):
        path.write_text(json.dumps(value))

    def create(self):
        return b.create_branch(self.parent,self.out,self.root)

    def child_config(self):
        config={**self.config,'run_id':self.out.name,'policy':'raw'}
        self.write(self.out/'config.json',config)
        return config

    def test_copy_has_only_exact_4000_prefix_and_required_evidence(self):
        lineage=self.create();state=json.loads((self.out/'state.json').read_text())
        self.assertEqual(len(state['history']),4000);self.assertEqual(len(state['rounds']),3)
        self.assertEqual(lineage['inherited_calls'],4000);self.assertEqual(lineage['remaining_calls'],6000)
        self.assertFalse(lineage['independent_replicate']);self.assertFalse((self.out/'config.json').exists())
        expected={'state.json','lineage.json','lineage_parent_config.json','remote_preflight.json'}
        expected|={f'batch_{i:02d}/{name}' for i in range(1,4) for name in ('proposal.json','normalized.json','search.json')}
        actual={str(p.relative_to(self.out)) for p in self.out.rglob('*') if p.is_file()}
        self.assertEqual(actual,expected)
        self.assertFalse(any(p.is_symlink() for p in self.out.rglob('*')))
        self.assertFalse(any('later_only_secret' in p.read_text() for p in self.out.rglob('*.json')))

    def test_parent_prefix_tamper_is_rejected_before_destination_creation(self):
        state=json.loads((self.parent/'state.json').read_text());state['history'][2000]['score']=-999.
        self.write(self.parent/'state.json',state)
        with self.assertRaisesRegex(ValueError,'observations'):
            self.create()
        self.assertFalse(self.out.exists())

    def test_wrong_parent_batch_run_binding_is_rejected(self):
        folder=self.parent/'batch_02'
        proposal=json.loads((folder/'proposal.json').read_text());proposal['run_id']='other_branch'
        response=json.loads((folder/'normalized.json').read_text());response['run_id']='other_branch'
        response['request_sha256']=hashlib.sha256(json.dumps(proposal,sort_keys=True).encode()).hexdigest()
        self.write(folder/'proposal.json',proposal);self.write(folder/'normalized.json',response)
        with self.assertRaisesRegex(ValueError,'run/target/seed'):
            self.create()

    def test_fork_keeps_same_budget_and_lineage_verifies_after_new_config(self):
        self.create();config=self.child_config();state=json.loads((self.out/'state.json').read_text())
        proof=b.verify_lineage(self.out,state,config)
        rows,_,_=e.verify_partial(self.out/'state.json',3,self.root)
        self.assertEqual(proof['inherited_calls'],len(rows));self.assertEqual(config['budget'],10000)
        self.assertEqual(proof['parent_run_id'],self.parent.name)
        self.assertTrue(proof['development_fork']);self.assertFalse(proof['independent_replicate'])

    def test_modified_child_prefix_or_inherited_evidence_is_rejected(self):
        self.create();config=self.child_config();state=json.loads((self.out/'state.json').read_text())
        state['history'][1000]['score']=-123.
        with self.assertRaisesRegex(ValueError,'prefix'):
            b.verify_lineage(self.out,state,config)
        state=json.loads((self.out/'state.json').read_text())
        self.write(self.out/'batch_01/search.json',{'tampered':True})
        with self.assertRaisesRegex(ValueError,'evidence hash'):
            b.verify_lineage(self.out,state,config)

    def test_destination_and_nested_fork_guards(self):
        self.out.mkdir();(self.out/'keep.txt').write_text('existing')
        with self.assertRaisesRegex(ValueError,'new or empty'):
            self.create()
        (self.out/'keep.txt').unlink()
        self.write(self.parent/'lineage.json',{})
        with self.assertRaisesRegex(ValueError,'Nested'):
            self.create()

    def test_parent_state_is_read_once_even_if_parent_advances(self):
        original=b.regular_bytes;reads=[]
        def advancing(path):
            value=original(path)
            if path==self.parent/'state.json':
                reads.append(path)
                newer=json.loads(value);newer['history'].extend([dict(row,product_id='later'+row['product_id'],iteration=5)
                                                                for row in self.history[:1000]])
                self.write(path,newer)
            return value
        with patch.object(b,'regular_bytes',side_effect=advancing):
            lineage=self.create()
        self.assertEqual(len(reads),1);self.assertEqual(lineage['parent_state_calls_at_snapshot'],5000)
        self.assertEqual(len(json.loads((self.out/'state.json').read_text())['history']),4000)

    def test_new_batch_must_use_child_run_id(self):
        lineage=self.create();config=self.child_config()
        folder=self.out/'batch_04';folder.mkdir()
        wrong=dict(target='tyk2',seed=0,run_id=self.parent.name,batch_id='batch_04')
        self.write(folder/'proposal.json',wrong);self.write(folder/'normalized.json',wrong)
        with self.assertRaisesRegex(ValueError,'run/target/seed'):
            b.verify_run_bindings(self.out,config,4,lineage)
        correct={**wrong,'run_id':self.out.name}
        self.write(folder/'proposal.json',correct);self.write(folder/'normalized.json',correct)
        b.verify_run_bindings(self.out,config,4,lineage)


if __name__=='__main__':
    unittest.main()
