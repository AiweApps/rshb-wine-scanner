"""Lossless, non-decision lexical evidence for separately typed catalogue roles.

A partial name match is a hypothesis, never an identity proof or a ranking bonus.
"""
from copy import deepcopy
from rshb_vine.typed_catalog_lexicon import normalize
from rshb_vine.resolution.identity import GRAPES, COLOR, SUGAR
from rshb_vine.gallery_variant_text import STYLE_ALIASES

MAPS = {'grape_blend':GRAPES, 'color':COLOR, 'sugar':SUGAR, 'style':STYLE_ALIASES}
NAMES = ('manufacturer','visible_brand','commercial_name','series')


def contains_tokens(haystack, needle):
    return bool(needle) and any(haystack[i:i+len(needle)] == needle for i in range(len(haystack)-len(needle)+1))


def extract_packet(packet, source):
    """Keep every reading; repeated normalized lines share one evidence group."""
    lines=[]
    for i, observation in enumerate(packet.get('observations', [])):
        raw=observation.get('raw_text','');tokens=normalize(raw).split();entities={}
        for role, aliases in MAPS.items():
            matches=[]
            for value, forms in aliases.items():
                for form in forms:
                    phrase=normalize(form).split()
                    if contains_tokens(tokens,phrase):matches.append({'value':value,'form':form,'tokens':phrase})
            # Retain more specific forms rather than also claiming their substring.
            entities[role]=[m for m in matches if not any(len(n['tokens'])>len(m['tokens']) and contains_tokens(n['tokens'],m['tokens']) for n in matches)]
        score=observation.get('score')
        lines.append({'line_id':i,'raw':deepcopy(observation),'normalized':normalize(raw),
                      'tokens':tokens,'same_reading_group':normalize(raw),
                      'legacy_threshold_admitted':isinstance(score,(int,float)) and score>=.85,
                      'entities':entities})
    return {'source':deepcopy(source),'packet_context':{k:deepcopy(v) for k,v in packet.items() if k!='observations'},
            'lines':lines,'confidence':'reader_specific_uncalibrated','is_ground_truth':False}


def match_card(extracted, card):
    """Return sparse matches; missing roles default to UNKNOWN, never conflict."""
    matches={}
    for role, claim in card['roles'].items():
        if not claim.get('comparison_allowed'):continue
        findings=[]
        for line in extracted['lines']:
            if not line['tokens']:continue
            if role in NAMES:
                for value in claim.get('values',[]):
                    expected=normalize(value).split()
                    shared=sorted(set(line['tokens']) & set(expected) - {t for t in expected if t.isdigit()})
                    if not shared:continue
                    full=contains_tokens(line['tokens'],expected)
                    other_roles=sorted({r for r, es in line['entities'].items() if any(set(e['tokens']) & set(shared) for e in es)})
                    findings.append({'line_id':line['line_id'],'catalog_value':value,'matched_tokens':shared,
                        'match_kind':'full_phrase' if full else 'partial_tokens','also_explained_by_roles':other_roles,
                        'threshold_admitted':line['legacy_threshold_admitted']})
            elif role in MAPS:
                expected=set(claim.get('values',[]))
                for e in line['entities'][role]:
                    if e['value'] in expected:findings.append({'line_id':line['line_id'],'catalog_value':e['value'],
                         'matched_form':e['form'],'match_kind':'typed_value','threshold_admitted':line['legacy_threshold_admitted']})
        if findings:
            matches[role]={'state':'WEAK_SUPPORT','findings':findings,'catalog_status':claim.get('status'),
                'catalog_sources':[deepcopy(c.get('source')) for c in claim.get('claims',[])],
                'identity_proven':False,'hard_veto_allowed':False}
    return {'slug':card['slug'],'product_id':card['product_id'],'roles':matches,
            'default_state':'UNKNOWN','ranking_effect':None,'strong_contradictions':[],
            'limits':'Lexical candidates only; partial/generic/grape overlap is not distinctive identity. No year inference.'}
