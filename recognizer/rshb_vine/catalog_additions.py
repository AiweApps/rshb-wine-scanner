"""Gallery-only expansion of the frozen recognition components, with alias evidence."""
from pathlib import Path
from collections import defaultdict
import copy,io,time
import numpy as np
from PIL import Image
from rshb_vine.io import read_json,read_jsonl,verify,seal,sha256
from rshb_vine.visual_core import LabelFirstIndex
from rshb_vine.catalog_constrained_title import CatalogConstrainedTitle
from rshb_vine.product_name_selector import ProductNameSelector
from rshb_vine.within_producer_title import WithinProducerTitle
from rshb_vine.typed_catalog_lexicon import TypedCatalogLexicon,normalize
from rshb_vine.label_specialist_consensus import VariantGuard

class SourceBoundGuard(VariantGuard):
    def __init__(self,catalog,signatures,groups,added):
        super().__init__(catalog,signatures,groups);self.added=set(added)
    def check(self,before,after,observations):
        # Owner policy: source-linked identity is not vetoed by disputed card text.
        if after in self.added:return True,['source_bound_added_slug_attributes_nonbinding']
        return super().check(before,after,observations)

def extend_profile(engine,catalog,added,checksum,root):
    engine.lexicon=TypedCatalogLexicon(catalog,root)
    fields={r['slug']:r['fields'] for r in catalog}
    for slug in added:
        f=fields[slug]
        # Newly added source metadata has not received field-level admission.
        # Keep its provenance in the catalog; only producer participates here.
        keys=['producer','product_name','grapes','wine_color','sweetness','sparkling_style']
        engine.profiles[slug]={'product':{k:{'comparison_value':f['Винодельня'] if k=='producer' else None} for k in keys}}
    engine.checksum=checksum
    engine.name_producers=defaultdict(set)
    for item in engine.lexicon.items.values():
        if item['product_phrase']:engine.name_producers[item['product_phrase']].add(normalize(item['raw_producer']))
    # Preserve released unique producer prefix aliases on old records.
    for item in getattr(engine,'derived_aliases',[]):
        if item['slug'] in engine.lexicon.items:engine.lexicon.items[item['slug']]['brand_forms'].append(item['form'])

class ExpandedCatalogPipeline:
    def __init__(self,root,directory):
        from scripts.run_fanagoria_gallery_candidate import load
        from scripts.run_label_specialist_consensus import GuardedConsensus
        from rshb_vine.square_rescue import SquareRescue
        from rshb_vine.oversized_label_speed import OversizedLabelRescue
        from rshb_vine.bottle_rescue import LowBottleDetector
        from rshb_vine.models import Detector,OCR
        from rshb_vine.conditional_line_profile import ConditionalLineProfile
        from rshb_vine.sparse_alternative_ocr_v2 import SparseAlternativeOCR
        self.root=Path(root);directory=Path(directory)
        checks=[('runs/oversized-label-speed-v1/protocol-metadata-v2.json','code_sha','rshb_vine/oversized_label_speed.py'),('runs/remaining-lines-v2/protocol.json','core_sha','rshb_vine/line_evidence_verifier_v2.py'),('runs/sparse-alternative-ocr-v2/protocol.json','code_sha','rshb_vine/sparse_alternative_ocr_v2.py')]
        for path,key,code in checks:
            if verify(read_json(self.root/path))[key]!=sha256(self.root/code):raise ValueError('Frozen component changed: '+code)
        reader=read_json(self.root/'runs/alternative-ocr-v1/protocol.json')
        for path,h in reader['weights'].items():
            if sha256(self.root/path)!=h:raise ValueError('OCR weight changed: '+path)
        if sha256(self.root/'rshb_vine/models.py')!=reader['recipe_sha256']:raise ValueError('OCR recipe changed')
        m=verify(read_json(directory/'manifest.json'))
        for path,h in m['files'].items():
            if sha256(self.root/path)!=h:raise ValueError('Catalog addition changed: '+path)
        for r in verify(read_json(directory/'intake.json'))['records']:
            if sha256(self.root/r['image_path'])!=r['image_sha256']:raise ValueError('Reference image changed')
        audit=read_json(self.root/'runs/catalog-additions-20260921/admission-audit.json')
        for source in audit['query_and_hold_metadata_checks']:
            if sha256(self.root/source['path'])!=source['sha256']:raise ValueError('Protected pool metadata changed')
        catalog=read_jsonl(directory/'catalog.jsonl');self.aliases=m['aliases'];added=m['added_canonical_slugs']
        title=load(directory/'startup-receipts'/('parent-'+str(time.time_ns())+'.json'));instance=title.parent
        self.core=GuardedConsensus(SquareRescue(title))
        signatures=verify(read_json(self.root/'runs/gallery-variant-source-review-v1/gallery/signatures.json'))['signatures']
        signatures={s:v for s,v in signatures.items() if s not in added}
        for arm,holder in [('B0',instance.base),('B3',self.core)]:
            g=verify(read_json(directory/arm/'gallery.json'));v=np.load(directory/arm/'vectors.npy');old=holder.index
            assert old.encoder_id==g['encoder_id'] and g['references'][:len(old.references)]==old.references
            assert np.array_equal(v[:len(old.vectors)],old.vectors)
            holder.index=LabelFirstIndex(v,g['references'],g['encoder_id'])
        allowed={r['slug'] for r in instance.base.index.references}
        selected=[r for r in catalog if r['slug'] in allowed]
        instance.variant_policy=ProductNameSelector(selected,signatures)
        title.policy=CatalogConstrainedTitle(selected,signatures)
        self.core.single=CatalogConstrainedTitle(catalog,signatures);self.core.multi=ProductNameSelector(catalog,signatures)
        self.core.title=WithinProducerTitle(self.core.single)
        self.core.guard=SourceBoundGuard(catalog,signatures,self.core.guard.groups,added)
        det=Detector(self.root,'mps');det.processor.size={'height':640,'width':640}
        self.rescue=OversizedLabelRescue(self.retry,LowBottleDetector(det),png_compression=1)
        self.lines=ConditionalLineProfile(self.root);self.alternative=SparseAlternativeOCR(self.root,OCR(self.root))
        for profile in [self.lines,self.alternative]:extend_profile(profile.engine,catalog,added,m['checksum'],self.root)
        self.manifest=seal({'kind':'expanded-catalog-no-fit-v1','catalog_additions':m['checksum'],'parent':self.core.manifest,'weights_changed':False,'canonical_slugs':len({r['slug'] for r in instance.base.index.references}),'aliases':self.aliases,'calibrated':False,'source_sha':sha256(__file__)})
    def retry(self,data):
        size=Image.open(io.BytesIO(data)).size
        return self.core.recognize(data,[0,0,*size])
    def recognize(self,data,roi=None):
        started=time.perf_counter();out=self.core.recognize(data,roi)
        out=self.rescue.apply(data,out,roi);out=self.lines.apply(data,out);out=self.alternative.apply(data,out)
        reverse=defaultdict(list)
        for alias,canonical in self.aliases.items():reverse[canonical].append(alias)
        for t in out.get('targets',[]):
            ret=t['retrieval'];ret['related_catalog_slugs']=sorted(reverse.get(ret.get('best_candidate'),[]))
        out['catalog_additions']={'manifest':self.manifest['catalog_additions'],'weights_changed':False,'related_catalog_slugs':sorted(reverse.get(out.get('best_candidate'),[]))}
        out.setdefault('timing_ms',{})['total']=(time.perf_counter()-started)*1000
        return out
