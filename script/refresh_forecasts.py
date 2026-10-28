"""Refresh selected notebooks from the latest source, preserving executed outputs.

Examples:
  py script/refresh_forecasts.py                 # monthly, quarterly, monthly-based FY
  py script/refresh_forecasts.py --types monthly quarterly
  py script/refresh_forecasts.py --types annual scenario
"""
from pathlib import Path
import argparse
import json
import os
import hashlib
from datetime import datetime, timezone
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from traitlets.config import Config

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    'monthly':'monthly_GL_forecast_dashboard',
    'quarterly':'quarterly_GL_forecast_dashboard',
    'fy':'allGL_FY_monthly_actuals_forecast',
    'annual':'allGLcode_forecast',
    'scenario':'forecast_scenario_dashboard',
}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--types',nargs='+',choices=NOTEBOOKS,default=['monthly','quarterly','fy'])
    args=parser.parse_args()
    os.chdir(ROOT)
    source=ROOT/'data/dawn_revenue_accounts_combined_history.csv'
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    output=ROOT/'result'
    output.mkdir(exist_ok=True)
    manifest=dict(started_utc=datetime.now(timezone.utc).isoformat(),source=str(source),
                  input_sha256=source_hash,status='running',notebooks=[])
    manifest_path=output/'latest_refresh.json'
    def save(): manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    save()
    try:
        for kind in dict.fromkeys(args.types):
            # Fresh namespace avoids variables from one notebook changing another.
            InteractiveShell.clear_instance()
            config=Config()
            config.HistoryManager.enabled=False
            shell=InteractiveShell.instance(config=config)
            path=ROOT/'notebook'/f'{NOTEBOOKS[kind]}.ipynb'
            notebook=nbformat.read(path,as_version=4)
            for cell in notebook.cells:
                if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
            count=0
            for cell in notebook.cells:
                if cell.cell_type!='code':continue
                count+=1
                print(f'{kind}: cell {count}',flush=True)
                with capture_output() as captured:
                    result=shell.run_cell(cell.source,store_history=False)
                cell.execution_count=count
                for name in ['stdout','stderr']:
                    text=getattr(captured,name)
                    if text:cell.outputs.append(nbformat.v4.new_output('stream',name=name,text=text))
                for item in captured.outputs:
                    cell.outputs.append(nbformat.v4.new_output('display_data',data=item.data,metadata=item.metadata))
                nbformat.write(notebook,path)
                if result.error_before_exec or result.error_in_exec:
                    raise RuntimeError(f'{kind}, cell {count}: {captured.stdout}\n{captured.stderr}')
            if hashlib.sha256(source.read_bytes()).hexdigest()!=source_hash:
                raise RuntimeError('Source CSV changed during the run. Finish updating it, then rerun all requested notebooks.')
            manifest['notebooks'].append(dict(type=kind,path=str(path),status='complete'))
            save()
        manifest['status']='complete'
        print('Refresh completed. Reopen the generated HTML dashboards.',flush=True)
    except BaseException as error:
        manifest.update(status='failed',error=str(error))
        raise
    finally:
        manifest['finished_utc']=datetime.now(timezone.utc).isoformat()
        save()

if __name__=='__main__':main()
