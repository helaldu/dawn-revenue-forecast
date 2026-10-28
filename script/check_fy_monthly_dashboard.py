"""Browser smoke checks for the fiscal-year dashboard (requires local Chrome)."""
from pathlib import Path
import json, subprocess, tempfile, time, urllib.request
import websocket

ROOT = Path(__file__).resolve().parents[1]
profile = Path(tempfile.mkdtemp(prefix='period-dashboard-check-'))
chrome = Path('C:/Program Files/Google/Chrome/Application/chrome.exe')
process = subprocess.Popen([str(chrome), '--headless=new', '--disable-gpu', '--no-first-run',
    '--no-default-browser-check', '--remote-debugging-port=0', f'--user-data-dir={profile}', 'about:blank'],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
try:
    for _ in range(100):
        if (profile / 'DevToolsActivePort').exists(): break
        time.sleep(.1)
    port = (profile / 'DevToolsActivePort').read_text().splitlines()[0]
    pages = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json'))
    ws = websocket.create_connection(next(p['webSocketDebuggerUrl'] for p in pages if p['type']=='page'), suppress_origin=True, timeout=20)
    seq = 0
    errors = []
    def call(method, params=None):
        global seq
        seq += 1
        ws.send(json.dumps(dict(id=seq,method=method,params=params or {})))
        while True:
            r=json.loads(ws.recv())
            if r.get('method')=='Runtime.exceptionThrown': errors.append(r)
            if r.get('id')==seq:
                if 'error' in r: raise RuntimeError(r)
                return r.get('result',{})
    def js(expression):
        r=call('Runtime.evaluate',dict(expression=expression,returnByValue=True))
        if 'exceptionDetails' in r: raise RuntimeError(r)
        return r['result'].get('value')
    call('Runtime.enable')
    for freq,horizon in [('fy_monthly',3)]:
        path=ROOT/'result/fy_monthly_forecast/fy_monthly_revenue_dashboard.html'
        call('Page.navigate',dict(url=path.as_uri()))
        for _ in range(100):
            if js("Boolean(document.getElementById('chart')?.data?.length)"): break
            time.sleep(.1)
        assert js("document.querySelectorAll('#yearCards .card').length")==horizon
        assert js("document.querySelectorAll('#projectionRows tr').length")==horizon+2
        assert js("document.getElementById('chart').data.length")>=6
        js("byId('adjustment').value=10;byId('adjustment').dispatchEvent(new Event('input'))")
        assert js("currentRows.filter(r=>r.type==='Forecast').every(r=>Math.abs(r.adjusted-(r.actual+(r.base-r.actual)*1.1))<.001)")
        assert js("orderedValidation(currentSeries).map(r=>r.model).join('|')")== 'Naive baseline|Linear|Holt|Monte Carlo|ML (Random forest)|ARIMA|SARIMA'
        js("byId('gl').value='3113';byId('gl').dispatchEvent(new Event('change'))")
        assert js("currentSeries.gl")=='3113'
        js("byId('reportNotes').value='Test note';byId('reportNotes').dispatchEvent(new Event('input'))")
        assert js("readNotes(currentSeries)")=='Test note'
        js("window.Blob=class{constructor(parts){window.testCSV=parts.join('')}};URL.createObjectURL=()=> 'blob:test';URL.revokeObjectURL=()=>{};HTMLAnchorElement.prototype.click=function(){};byId('downloadScenario').click()")
        assert 'Test note' in js('window.testCSV')
        for width in [1440,390]:
            call('Emulation.setDeviceMetricsOverride',dict(width=width,height=1100,deviceScaleFactor=1,mobile=False))
            time.sleep(.3)
            assert js('document.documentElement.scrollWidth<=innerWidth')
        js("byId('adjustment').value=-100;byId('adjustment').dispatchEvent(new Event('input'))")
        assert js("currentRows.filter(r=>r.type==='Forecast').every(r=>Math.abs(r.adjusted-r.actual)<.001)")
        assert js("document.querySelectorAll('#diagnosticRows tr').length")>=2
        js("byId('adjustment').value=0;byId('adjustment').dispatchEvent(new Event('input'))")
        call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1400,deviceScaleFactor=1,mobile=False))
        time.sleep(.5)
        import base64
        shot=call('Page.captureScreenshot',dict(format='png'))['data']
        (ROOT/'result/fy_monthly_forecast/dashboard_preview.png').write_bytes(base64.b64decode(shot))
        assert not errors, errors
        print(freq+': chart, scenario, filters, notes, CSV and responsive layout passed',flush=True)
    ws.close()
finally:
    process.terminate()
