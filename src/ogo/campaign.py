"""Production baseline campaign CLI, separate from disposable ogo-study profiles."""
import argparse
from pathlib import Path
from .campaign_runtime import METHODS,BudgetStop,Guard,allocated,configure_torch,initialize,protected,lock,read


def command_tag(a):
    if a.command in ('verify','audit','report'):return a.command
    if a.command=='collect':return 'collect-'+str(a.steps)
    if a.command=='train':return 'train-'+str(a.method)+'-'+str(a.steps)
    if a.command=='evaluate':return '-'.join(str(v) for v in ('evaluate',a.method,a.steps,a.role,a.update))
    return 'prepare-'+a.part+('-'+str(a.partition) if a.part=='references' else '')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('verify','prepare','collect','train','evaluate','audit','report'))
    for name in ('config','assets','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--deadline',type=float)
    p.add_argument('--method',choices=(*METHODS,'control'));p.add_argument('--steps',type=int,choices=(16,32))
    p.add_argument('--role',choices=('development','final'));p.add_argument('--update',type=int,choices=(0,128,256,384,512))
    p.add_argument('--part',choices=('index','references','seal','real'),default='references')
    p.add_argument('--partition',type=int,choices=(0,1),default=0)
    a=p.parse_args();config,assets,identity=initialize(a.config,a.assets,a.output)
    if a.command in ('audit','report'):
        from .campaign_audit import audit,report
        passed=(audit if a.command=='audit' else report)(config,assets,a.output,identity)['passed']
        print(passed)
        if not passed:raise SystemExit(2)
        return
    allocated();guard=Guard(a.output,config,a.deadline);protected(assets);configure_torch(config['seed'])
    tag=command_tag(a)
    with lock(Path(a.output)/'.locks'/tag) as owned:
        if not owned:raise RuntimeError('Duplicate campaign command: '+tag)
        if a.command!='verify' and not (a.command=='prepare' and a.part=='index'):
            gate=read(Path(a.output)/'verification/verified.json')
            if gate['campaign_identity']!=identity or not gate['passed'] or not gate['feasible_with_reserve']:
                raise RuntimeError('Production preflight gate is closed')
        execute(a,p,config,assets,identity,guard)


def execute(a,p,config,assets,identity,guard):
    try:
        if a.command=='verify':
            from .campaign_verify import verify
            verify(config,assets,a.output,identity,guard,a.config,a.assets)
        elif a.command=='prepare':
            from .campaign_data import build_index,prepare,seal_references
            if a.part=='index':build_index(config,assets,a.output,identity)
            elif a.part=='references':prepare(config,assets,a.output,identity,guard,a.partition)
            elif a.part=='seal':
                if not seal_references(a.output,identity):raise RuntimeError('Reference shards are not all committed')
            else:
                from .campaign_eval import fid_process
                fid_process(a.config,a.assets,a.output,assets,guard,'real')
        elif a.command=='collect':
            if a.steps is None:p.error('collect requires --steps')
            from .campaign_engine import collect_preferences
            collect_preferences(config,assets,a.output,identity,guard,a.steps)
        elif a.command=='train':
            if a.method not in METHODS or a.steps is None:p.error('train requires a baseline --method and --steps')
            from .campaign_engine import train
            train(config,assets,a.output,identity,guard,a.method,a.steps)
        else:
            if a.method is None or a.steps is None or a.role is None or a.update is None:
                p.error('evaluate requires --method, --steps, --role and --update')
            if ((a.method=='control' and a.update!=0) or
                (a.method!='control' and a.update not in ([128,256,384] if a.role=='development' else [512]))):
                p.error('Evaluation endpoint differs from the frozen protocol')
            from .campaign_eval import evaluate
            evaluate(config,assets,a.output,identity,guard,a.method,a.steps,a.role,a.update,a.config,a.assets)
        protected(assets)
    except BudgetStop as error:
        print(str(error),flush=True);raise SystemExit(75)


if __name__=='__main__':main()
