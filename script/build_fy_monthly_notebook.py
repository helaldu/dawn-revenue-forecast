"""Rebuild the readable notebook from its percent-format Python source."""
from pathlib import Path
import re
import nbformat

root=Path(__file__).resolve().parents[1]
source=(root/'script/allGL_fy_monthly_models.py').read_text(encoding='utf-8')
cells=[]
for part in re.split(r'^# %%',source,flags=re.M)[1:]:
    first,_,body=part.partition('\n')
    if '[markdown]' in first:
        text='\n'.join(line[2:] if line.startswith('# ') else '' if line=='#' else line for line in body.strip().splitlines())
        cells.append(nbformat.v4.new_markdown_cell(text))
    else:
        cells.append(nbformat.v4.new_code_cell(body.strip()))
nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'}})
nbformat.write(nb,root/'notebook/allGL_FY_monthly_actuals_forecast.ipynb')
print('Notebook written:',len(cells),'cells')
