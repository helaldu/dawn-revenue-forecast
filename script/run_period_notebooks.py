"""Execute both notebooks and retain their tables/outputs, without a kernel server."""
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from traitlets.config import Config

root = Path(__file__).resolve().parents[1]
config = Config()
config.HistoryManager.enabled = False
shell = InteractiveShell.instance(config=config)
for frequency in ['monthly', 'quarterly']:
    path = root / 'notebook' / f'{frequency}_GL_forecast_dashboard.ipynb'
    notebook = nbformat.read(path, as_version=4)
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        count += 1
        print(f'{frequency}: executing cell {count}', flush=True)
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=False)
        cell.execution_count = count
        cell.outputs = []
        for name in ['stdout', 'stderr']:
            value = getattr(captured, name)
            if value:
                cell.outputs.append(nbformat.v4.new_output('stream', name=name, text=value))
        for item in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data', data=item.data, metadata=item.metadata))
        nbformat.write(notebook, path)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(captured.stdout + captured.stderr)
    print(f'{frequency}: completed', flush=True)
