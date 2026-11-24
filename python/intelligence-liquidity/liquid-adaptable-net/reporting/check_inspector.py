"""Exercise every exported phase and role in a local headless Chromium."""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from evolution import ROOT, digest

ASSERTIONS = r"""
<script>
(()=>{try{
 let checks=0;
 for(const r of data.controlled){
  $('run').value=r.seed;
  for(let phase=0;phase<6;phase++){
   $('phase').value=phase;
   for(const role of ['active','candidate','predecessor']){
    $('role').value=role;update();
    const state=r.phases[phase][role];
    if(state){
     if($('network').querySelectorAll('rect').length<state.profile.boards)throw Error('Missing board');
     if(!$('raw').textContent.includes(state.weights_sha256))throw Error('Wrong raw state');
     if(!$('stats').textContent.includes(state.profile.parameters.toLocaleString()))throw Error('Wrong parameter count');
    }else if(!$('network').textContent.includes('No model in this role'))throw Error('Missing empty role');
    if(!$('phaseText').textContent.includes(String(r.phases[phase].revealed_labels)))throw Error('Wrong phase count');
    checks++;
   }
  }
 }
 for(let i=0;i<data.houses.length;i++){
  $('house').value=i;house();
  if($('houseEvents').children.length!==data.houses[i].events.length)throw Error('Missing house event');
  checks++;
 }
 $('run').value=68;$('phase').value=2;$('role').value='candidate';update();
 $('next').click();if($('phase').value!=='3')throw Error('Next button');
 $('back').click();if($('phase').value!=='2')throw Error('Previous button');
 document.body.dataset.testResult='passed';document.body.dataset.testChecks=checks+2;
}catch(e){document.body.dataset.testResult='failed';document.body.dataset.testError=e.stack;}})();
</script>
"""


def check(screenshot=None):
    browser = shutil.which("google-chrome") or shutil.which("chromium")
    if not browser:
        raise RuntimeError("Install Chromium to run the optional local browser check")
    viewer = ROOT / "evolution/inspector.html"
    source = viewer.read_text()
    with tempfile.TemporaryDirectory(prefix="liquid-viewer-test-") as folder:
        page = Path(folder) / "test.html"
        page.write_text(source.replace("</body>", ASSERTIONS + "</body>"))
        args = [
            browser,
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-background-networking",
            "--no-first-run",
            f"--user-data-dir={folder}/profile",
            "--virtual-time-budget=3000",
            "--window-size=1500,1400",
            "--dump-dom",
        ]
        if screenshot:
            args.append(f"--screenshot={screenshot.resolve()}")
        result = subprocess.run(
            args + [page.as_uri()], capture_output=True, text=True, timeout=40
        )
        body = re.search(r"<body[^>]+>", result.stdout)
        if result.returncode or not body or 'data-test-result="passed"' not in body[0]:
            raise AssertionError(
                (body[0] if body else "No rendered body") + result.stderr[-1500:]
            )
        checks = int(re.search(r'data-test-checks="(\d+)"', body[0])[1])
        if checks != 116:
            raise AssertionError("Browser did not exercise all views")
    record = dict(
        status="passed",
        checks=checks,
        coverage="108 run/phase/role combinations, six house event lists, previous and next controls",
        browser=subprocess.run(
            [browser, "--version"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        viewer_sha256=digest(viewer),
        checker_sha256=digest(__file__),
    )
    if screenshot:
        record["screenshot_sha256"] = digest(screenshot)
    (ROOT / "evolution/inspector-check.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot", type=Path)
    check(parser.parse_args().screenshot)
