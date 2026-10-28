"""Actual production tensor loader across legacy and new shard boundaries."""
import tempfile
import time
from pathlib import Path
import unittest

import torch
from ogo.campaign_data import build_index,Pool,pool_seed
from ogo.campaign_runtime import read,write,sha
from ogo.checkpoint import atomic_save


class ProductionIO(unittest.TestCase):
    def test_rows_seeds_labels_boundaries_and_corrupt_resume(self):
        config=read(Path(__file__).parents[1]/'configs/imagenet256-baselines-512.json')
        config['reference_reconstruction_probe_ids']=[]
        torch.set_default_dtype(torch.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);legacy=root/'legacy';legacy.mkdir()
            endpoint=torch.arange(256*4096,dtype=torch.float32).reshape(256,4096)
            atomic_save(legacy/'class-0000.pt',dict(latent=endpoint))
            write(legacy/'class-0000.json',dict(identity='fixture',class_id=0,sha256=sha(legacy/'class-0000.pt'),
                seeds=[pool_seed(0,i) for i in range(256)]))
            write(root/'audit.json',dict(backbones=dict(sit256=dict(reserved_profile_ids=[]))))
            assets=dict(sit_pool=str(legacy),sit_pool_identity='fixture',reuse_audit=str(root/'audit.json'))
            index=build_index(config,assets,root,'fixture');new=[r for r in index['records'] if r['kind']=='new']
            folder=root/'references';(folder/'shards').mkdir();hashes={};expected={}
            for i in (0,1):
                rows=new[i*32:(i+1)*32];values=torch.stack([torch.full((4,32,32),float(i*32+j)) for j in range(32)])
                path=folder/rows[0]['file'];atomic_save(path,dict(latent=values,ids=[r['id'] for r in rows]))
                hashes[path.with_suffix('.json').name]=sha(path)
                expected.update({r['id']:values[j] for j,r in enumerate(rows)})
            write(folder/'ready.json',dict(index_sha256=sha(folder/'index.json'),shards=hashes,fixture=True))
            pool=Pool(config,root);rows=[pool.records[0]['final'][0],new[0],new[31],new[32],new[63]]
            data=pool.materialize(rows,device='cpu')
            self.assertEqual(data['ids'],[r['id'] for r in rows])
            self.assertEqual(data['labels'].tolist(),[r['class_id'] for r in rows])
            self.assertTrue(torch.equal(data['reference'][0],endpoint[130].reshape(4,32,32)))
            for i,row in enumerate(rows):
                initial=torch.randn(4,32,32,generator=torch.Generator().manual_seed(row['seed']))
                self.assertTrue(torch.equal(initial,data['initial'][i]))
                if i:self.assertTrue(torch.equal(data['reference'][i],expected[row['id']]))
            resumed=Pool(config,root).materialize(rows[3:],device='cpu')
            self.assertTrue(torch.equal(resumed['reference'],data['reference'][3:]))
            bad=folder/new[32]['file'];bad.write_bytes(b'corrupted')
            with self.assertRaises(ValueError):Pool(config,root).materialize(rows[3:],device='cpu')


if __name__=='__main__':unittest.main()
