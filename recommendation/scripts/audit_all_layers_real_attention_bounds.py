#!/usr/bin/env python3
"""CPU-only exact-real attention range audit of one immutable JSONL byte prefix.

This reads actual recorded scalar arguments but does not certify GPU arithmetic,
integer offsets, initial destination bytes, or complete training coverage.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

ATTENTION = 'hstu_attention_bwd'


def outward(value):
    result = float(value)
    return math.nextafter(result, math.inf) if Fraction(result) < value else result


def tensor_magnitude(t):
    if t.get('dtype') != 'torch.bfloat16' or t.get('scanned') is not True:
        raise ValueError('requires_scanned_BF16_tensor')
    if t.get('finite') is not True:
        raise ValueError('nonfinite_tensor')
    values = [t.get('min'), t.get('max')]
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError('invalid_finite_tensor_extrema')
    if values[0] > values[1]:
        raise ValueError('reversed_tensor_extrema')
    return Fraction(max(map(abs, values)))


def summary_map(tensors):
    result = {t['name']:t for t in tensors}
    if len(result) != len(tensors):
        raise ValueError('duplicate_tensor_name')
    return result


def positive_shape(t):
    shape = t.get('shape')
    if (not isinstance(shape,list) or len(shape)!=3
            or any(type(n) is not int or n<1 for n in shape)):
        raise ValueError('requires_positive_rank3_tensor_shape')
    return shape


def bounds_for_pair(before, after):
    scalars = before.get('scalar_arguments')
    if not isinstance(scalars,dict) or scalars != after.get('scalar_arguments'):
        raise ValueError('missing_or_mismatched_actual_scalar_arguments')
    n, alpha = scalars.get('N'), scalars.get('alpha')
    if type(n) is not int or n<1:
        raise ValueError('requires_positive_actual_N')
    if type(alpha) not in (int,float) or not math.isfinite(alpha):
        raise ValueError('requires_finite_actual_alpha')
    if scalars.get('num_softmax_heads') != 0 or scalars.get('enable_tma') is not False:
        raise ValueError('requires_non_TMA_SiLU_attention_without_softmax_heads')
    for key in ('max_attn_len','contextual_seq_len'):
        if type(scalars.get(key)) is not int or scalars[key]<0:
            raise ValueError('invalid_mask_scalar_argument')
    ins = summary_map(before['inputs'])
    outs = summary_map(after['outputs'])
    if not {'q','k','v','dout'}<=set(ins) or set(outs)!={'dq','dk','dv'}:
        raise ValueError('unexpected_attention_tensor_names')
    shapes = {name:positive_shape(ins[name]) for name in ('q','k','v','dout')}
    rows, heads, dq = shapes['q']
    vr, vh, dv = shapes['v']
    if shapes['k']!=[rows,heads,dq] or (rows,heads)!=(vr,vh) or shapes['dout']!=[rows,heads,dv]:
        raise ValueError('input_shape_contract_mismatch')
    if positive_shape(outs['dq'])!=[rows,heads,dq] or positive_shape(outs['dk'])!=[rows,heads,dq] or positive_shape(outs['dv'])!=[rows,heads,dv]:
        raise ValueError('output_shape_contract_mismatch')
    q,k,v,g = [tensor_magnitude(ins[name]) for name in ('q','k','v','dout')]
    a = abs(Fraction(alpha))
    # L<=N is an explicit semantic contract. L<=total_rows uses the tensor shape.
    length_ratio = Fraction(min(rows,n),n)
    b = {'dq':a*length_ratio*Fraction(11,10)*dv*v*g*k,
         'dk':a*length_ratio*Fraction(11,10)*dv*v*g*q,
         'dv':length_ratio*a*dq*q*k*g}
    results = {}
    for name,t in outs.items():
        if t.get('dtype')!='torch.bfloat16' or t.get('scanned') is not True:
            raise ValueError('requires_scanned_BF16_outputs')
        if t.get('finite') is False:
            results[name]={'observed_max_abs':None,'output_finite':False,
                           'bound':outward(b[name]),'ratio_to_bound':'nonfinite',
                           'exceeds_real_bound':True}
        else:
            observed=tensor_magnitude(t)
            ratio=float(observed/b[name]) if b[name] else (0.0 if observed==0 else 'infinite')
            results[name]={'observed_max_abs':float(observed),'output_finite':True,
                           'bound':outward(b[name]),'ratio_to_bound':ratio,
                           'exceeds_real_bound':observed>b[name]}
    return {'actual_scalar_arguments':scalars,'Q_shape':shapes['q'],'V_shape':shapes['v'],
            'input_maxima':dict(zip(('q','k','v','dout'),map(float,(q,k,v,g)))),
            'assumed_maximum_L_over_N':float(length_ratio),'outputs':results}


def audit_bytes(raw):
    # A racing writer may leave a partial last JSON record. Pin only full lines.
    end = raw.rfind(b'\n')+1
    prefix = raw[:end]
    events = [(line,json.loads(row)) for line,row in enumerate(prefix.splitlines(),1)]
    pairs={};problems=[]
    for line,e in events:
        if e.get('operation')!=ATTENTION or e.get('event') not in ('before','after'):
            continue
        attempt=e['attempt']
        key=(e['session'],attempt['mode'],attempt['repeat'],attempt['step'],e['layer'],e['call'])
        pair=pairs.setdefault(key,{})
        if e['event'] in pair:
            problems.append({'problem':'duplicate_attention_phase','line':line,'key':list(key),'phase':e['event']})
            pair['invalid']=True
        else:
            pair[e['event']]=(line,e)
    rows=[];skips=[]
    for key,pair in sorted(pairs.items()):
        ctx={'session':key[0],'mode':key[1],'repeat':key[2],'step':key[3],'layer':key[4],'call':key[5]}
        if pair.get('invalid'):
            skips.append({**ctx,'reason':'duplicate_phase'});continue
        if set(pair)!={'before','after'}:
            skips.append({**ctx,'reason':'incomplete_attention_pair','present_phases':sorted(pair)});continue
        bl,b=pair['before'];al,a=pair['after'];ctx.update(before_line=bl,after_line=al)
        if bl>=al:
            problems.append({**ctx,'problem':'reversed_attention_phases'});continue
        try:
            result=bounds_for_pair(b,a)
        except (ValueError,KeyError,TypeError) as error:
            skips.append({**ctx,'reason':str(error)});continue
        rows.append({**ctx,**result})
    violations=[row for row in rows if any(v['exceeds_real_bound'] for v in row['outputs'].values())]
    selected={(row['session'],row['mode'],row['repeat'],row['step']) for row in violations}
    flows=[]
    for line,e in events:
        attempt=e.get('attempt') or {}
        key=(e.get('session'),attempt.get('mode'),attempt.get('repeat'),attempt.get('step'))
        if key not in selected or e.get('event') not in ('before','after'):
            continue
        field='inputs' if e['event']=='before' else 'outputs'
        flows.append({'line':line,'session':key[0],'step':key[3],'layer':e['layer'],
                      'call':e['call'],'operation':e['operation'],'phase':e['event'],
                      field:[{k:t.get(k) for k in ('name','dtype','shape','min','max','max_abs_finite','finite','scanned')}
                             for t in e.get(field,[]) if t.get('scanned')]})
    counts=Counter(row['layer'] for row in rows)
    maxima={}
    for row in rows:
        for name,output in row['outputs'].items():
            ratio=output['ratio_to_bound'];rank=ratio if isinstance(ratio,(int,float)) else math.inf
            key=row['layer']+'/'+name
            if key not in maxima or rank>maxima[key][0]:maxima[key]=(rank,{'step':row['step'],'before_line':row['before_line'],**output})
    proof=sum((Fraction(12,5)**n/math.factorial(n) for n in range(13)),Fraction())
    assert proof>11
    return {
        'format':'all_layers_actual_scalar_real_attention_bound_prefix_v1',
        'event_prefix_bytes':len(prefix),'event_prefix_sha256':hashlib.sha256(prefix).hexdigest(),
        'read_bytes':len(raw),'excluded_trailing_partial_bytes':len(raw)-len(prefix),'event_records':len(events),
        'arithmetic':'Exact Fraction real arithmetic with outward-rounded display bounds. GPU arithmetic is not certified.',
        'observed_attempt_range':[min((e.get('attempt') or {}).get('step',math.inf) for _,e in events),
                                  max((e.get('attempt') or {}).get('step',0) for _,e in events)],
        'complete_attention_pairs_checked':len(rows),'checked_pairs_by_layer':dict(counts),
        'real_bound_exceeding_calls':len(violations),
        'real_bound_exceeding_output_counts':dict(Counter(name for row in violations for name,v in row['outputs'].items() if v['exceeds_real_bound'])),
        'pairing_problems':problems,'skips':skips,
        'maximum_output_to_bound_by_layer_and_output':{key:value[1] for key,value in maxima.items()},
        'formulas':{'dq':'abs(alpha)*(L_keys/N)*(11/10)*Dv*Vmax*dOutmax*Kmax',
                    'dk':'abs(alpha)*(L_queries/N)*(11/10)*Dv*Vmax*dOutmax*Qmax',
                    'dv':'(L_queries/N)*abs(alpha)*Dq*Qmax*Kmax*dOutmax'},
        'derivative_proof':'|SiLU_prime(x)|<=11/10: for x>=0 the upper inequality is exp(x)+12+11exp(-x)-10x>=0, with minimum24-10ln11>0. A positive13-term exact Taylor prefix proves exp(12/5)>11. For negative x use SiLU_prime(-x)=1-SiLU_prime(x).',
        'scope':[
            'Actual scalar_arguments.N/alpha are taken only from before and identical matching after events; no configuration-derived replacement.',
            'Finite scanned BF16 read-input extrema and rank3 shape contracts are checked. Finite extrema do not certify every bit, integer sequence offsets, input immutability or intermediate values.',
            'Valid sequence lengths L<=N and valid disjoint sequence partition remain semantic assumptions because integer offsets are not scanned. Boolean masking only removes terms.',
            'Mathematical DQ initially zero, exact exp/reciprocal and exact arithmetic are assumed. Actual initial destination bytes, FP32 reductions, BF16 conversions/recurrence, overflow/underflow and runtime binary scheduling are outside this bound.',
            'No whole-training endpoint/phase coverage audit is performed here. Incomplete pairs are explicitly skipped; finite prefix observations are not a terminal training verdict.',
            'Flow records contain successive scalar ranges for all available operations in violating attempts; they do not prove elementwise propagation, tensor identity or exclude mutation.',
            'Real-bound violations identify a conditional producing-boundary lead, not a hardware instruction or a link to historical enormous-value captures. Passing bounds do not establish correct outputs.',
        ],
        'violations':violations,'flow_for_violating_attempts':flows,'all_checked_calls':rows,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('events',type=Path)
    parser.add_argument('--report',required=True,type=Path)
    args=parser.parse_args()
    report=audit_bytes(args.events.read_bytes())
    report['event_file']=str(args.events)
    report['analysis_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.report.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:report[k] for k in ('event_prefix_bytes','event_prefix_sha256','complete_attention_pairs_checked','checked_pairs_by_layer','real_bound_exceeding_calls','real_bound_exceeding_output_counts','pairing_problems','skips')},indent=2))
    print('violation coordinates',[(v['step'],v['layer'],{n:o['ratio_to_bound'] for n,o in v['outputs'].items() if o['exceeds_real_bound']}) for v in report['violations']])


if __name__=='__main__':main()
