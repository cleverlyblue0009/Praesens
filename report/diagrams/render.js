const { chromium } = require('playwright');
const fs = require('fs'), path = require('path');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ deviceScaleFactor: 3, viewport: { width: 2400, height: 1600 } });
  const mjs = fs.readFileSync(path.join(__dirname, 'node_modules/mermaid/dist/mermaid.min.js'), 'utf8');
  await page.setContent(`<html><body style="margin:0;background:#fff"><div id="c"></div></body></html>`);
  await page.addScriptTag({ content: mjs });
  await page.evaluate(() => mermaid.initialize({ startOnLoad: false, theme: 'base',
    themeVariables: { fontFamily: 'Liberation Serif, Times New Roman, serif', fontSize: '16px',
      primaryColor: '#D6EAF8', primaryBorderColor: '#1F3A68', lineColor: '#34495E',
      secondaryColor: '#D1F2EB', tertiaryColor: '#FEF5E7', actorBkg: '#1F3A68', actorTextColor: '#fff',
      actorBorder: '#1F3A68', signalColor: '#1F3A68', noteBkgColor: '#FEF9E7', activationBkgColor: '#D1F2EB',
      sectionBkgColor: '#D6EAF8', altSectionBkgColor: '#E8F8F5', taskBkgColor: '#0F8B8D', doneTaskBkgColor: '#0F8B8D',
      doneTaskBorderColor: '#0B6466', activeTaskBkgColor: '#E07A1F', activeTaskBorderColor: '#A85A12',
      taskTextColor: '#fff', taskTextDarkColor: '#fff', gridColor: '#BBB', critBkgColor: '#C0392B',
      edgeLabelBackground: '#FFFFFF', clusterBkg: '#FBFCFC', clusterBorder: '#99A3A4' },
    flowchart: { curve: 'basis', htmlLabels: true, padding: 12, useMaxWidth: false, nodeSpacing: 35, rankSpacing: 45 },
    sequence: { mirrorActors: false, actorMargin: 25, useMaxWidth: false, messageFontSize: 17, actorFontSize: 17, noteFontSize: 16, width: 120 }, gantt: { barHeight: 24, fontSize: 13, sectionFontSize: 13, leftPadding: 260 },
    class: { htmlLabels: true, useMaxWidth: false } }));
  for (const f of fs.readdirSync(path.join(__dirname, 'defs')).filter(x => x.endsWith('.mmd'))) {
    const src = fs.readFileSync(path.join(__dirname, 'defs', f), 'utf8');
    const svg = await page.evaluate(async (s) => (await mermaid.render('g' + Math.floor(Math.random()*1e9), s)).svg, src);
    await page.evaluate((svg) => { document.getElementById('c').innerHTML = svg; }, svg);
    const el = await page.$('#c svg');
    await el.screenshot({ path: path.join(process.argv[2], 'dg_' + f.replace('.mmd', '.png')), omitBackground: false });
    console.log('rendered', f);
  }
  await browser.close();
})();
