"""Execute the FY notebook, saving cell outputs and showing fitting progress."""
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from traitlets.config import Config

root=Path(__file__).resolve().parents[1]
path=root/'notebook/allGL_FY_monthly_actuals_forecast.ipynb'
nb=nbformat.read(path,as_version=4)
config=Config()
config.HistoryManager.enabled=False
shell=InteractiveShell.instance(config=config)
count=0
for cell in nb.cells:
    if cell.cell_type!='code': continue
    count+=1
    print(f'Executing cell {count}',flush=True)
    # Keep the expensive fitting cell's progress visible in the terminal.
    progress="functions_to_hash =" in cell.source
    with capture_output(stdout=not progress) as captured:
        result=shell.run_cell(cell.source,store_history=False)
    cell.execution_count=count
    cell.outputs=[]
    for name in ['stdout','stderr']:
        value=getattr(captured,name)
        if value: cell.outputs.append(nbformat.v4.new_output('stream',name=name,text=value))
    for item in captured.outputs:
        cell.outputs.append(nbformat.v4.new_output('display_data',data=item.data,metadata=item.metadata))
    nbformat.write(nb,path)
    if result.error_before_exec or result.error_in_exec:
        raise RuntimeError(captured.stdout+captured.stderr)
print('Notebook execution completed',flush=True)
