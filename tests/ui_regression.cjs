const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

// Execute the real inline script in an isolated DOM stub; no server or packages.
const html = readFileSync(join(__dirname, '../pipeline/static/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
function ui(respond = () => ({ answer: 'Mock answer', filters: {} })) {
  const nodes = new Map();
  const node = selector => {
    if (!nodes.has(selector)) nodes.set(selector, {
      innerHTML: '', value: '', textContent: '', disabled: false,
      dataset: {}, classList: { toggle() {}, add() {}, remove() {} },
      querySelectorAll: () => [], addEventListener() {}, focus() {},
    });
    return nodes.get(selector);
  };
  const requests = [];
  const context = vm.createContext({
    URLSearchParams, console, setTimeout, clearTimeout,
    document: { querySelector: node, querySelectorAll: () => [], documentElement: { dataset: {} } },
    localStorage: { getItem: () => null, setItem() {} },
    history: { replaceState() {} }, location: { search: '', href: 'http://localhost/' },
    window: {}, navigator: {},
    fetch: async url => {
      requests.push(new URL(url, 'http://localhost'));
      const data = await respond(requests.at(-1));
      return { ok: true, json: async () => data };
    },
  });
  // Suppress only boot's network activity; compile and execute all other UI code.
  vm.runInContext(script.replace(/boot\(\);\s*$/, ''), context);
  const run = code => vm.runInContext(code, context);
  run('Object.assign(S,{q:"example",since:"2019",until:"2022",concept:{concept_id:"D1",concept_name:"One"},view:"ask"})');
  return { run, node, requests, context };
}

test('Ask sends scope and warns when HTTP200 answer does not confirm it', async () => {
  const u = ui();
  await u.run('ask()');
  const request = u.requests[0];
  assert.equal(request.pathname, '/api/ask');
  for (const [key, value] of Object.entries({ q: 'example', since_year: '2019', until_year: '2022', concept_id: 'D1' })) {
    assert.equal(request.searchParams.get(key), value, key);
  }
  assert.match(u.node('#work').innerHTML, /did not confirm the requested scope/);
});

test('comparison sends both year boundaries', async () => {
  const u = ui(() => ({ a: { papers: 1 }, b: { papers: 2 } }));
  u.run('S.concepts=[{concept_id:"D1"},{concept_id:"D2"}];compareView()');
  u.node('#ca').value = 'D1';
  u.node('#cb').value = 'D2';
  await u.node('#compareForm').onsubmit({ preventDefault() {} });
  const request = u.requests[0];
  assert.equal(request.pathname, '/api/compare');
  assert.equal(request.searchParams.get('since_year'), '2019');
  assert.equal(request.searchParams.get('until_year'), '2022');
});

test('each analytics error is visible without suppressing successful panels', async () => {
  const u = ui(url => {
    if (['/api/trend', '/api/countries'].includes(url.pathname)) return { error: 'Scope <unsupported>', results: [], caveat: 'Limited coverage' };
    if (['/api/study_types', '/api/kols'].includes(url.pathname)) throw Error('Network unavailable');
    return { results: [{ title: 'Retained paper', journal_title: 'Retained journal', papers: 3 }], caveat: 'Identity <provisional>' };
  });
  for (const view of ['overview', 'entities']) {
    await u.run(`S.view='${view}';${view}()`);
    const output = u.node('#work').innerHTML;
    assert.match(output, /Scope &lt;unsupported&gt;/);
    assert.match(output, /Network unavailable/);
    assert.match(output, /Limited coverage/);
    assert.match(output, /Identity &lt;provisional&gt;/);
    assert.match(output, /Retained (paper|journal)/);
    assert.doesNotMatch(output, /No data available for this scope/);
    if (view === 'overview') assert.match(output, /Publication trend/);
  }
});

function pendingAsk() {
  let resolve, reject;
  const response = new Promise((yes, no) => { resolve = yes; reject = no; });
  const u = ui(url => url.pathname === '/api/ask' ? response : { results: [], concepts: [] });
  return { ...u, resolve, reject };
}

test('scope changes invalidate cached Ask and discard stale successes and failures', async () => {
  const transitions = {
    select: async u => { await u.run('S.concepts=[{concept_id:"D2",concept_name:"Two"}];pick(0)'); },
    clearConcept: async u => { await u.node('#clearScope').onclick(); },
    clearAll: async u => { u.node('#clear').onclick(); },
    search: async u => {
      u.node('#q').value = 'different question';
      u.node('#since').value = '2020';
      u.node('#until').value = '2021';
      u.node('#k').value = '20';
      await u.run('search()');
    },
    // Returning to the original scope must not revive the earlier request.
    roundTrip: async u => { await u.run('S.concepts=[S.concept];pick(-1)'); await u.run('pick(0)'); },
  };
  for (const [name, change] of Object.entries(transitions)) {
    for (const failure of [false, true]) {
      const u = pendingAsk();
      u.run('S.ask={answer:"Cached old answer"}');
      const pending = u.run('ask()');
      await change(u);
      assert.equal(u.run('S.ask'), null, `${name}: cache cleared`);
      const before = u.node('#work').innerHTML;
      if (failure) u.reject(Error('Stale failure'));
      else u.resolve({ answer: 'Stale answer' });
      await pending;
      assert.equal(u.run('S.ask'), null, `${name}: stale reply discarded`);
      assert.equal(u.node('#work').innerHTML, before, `${name}: no stale render`);
      assert.equal(u.run('S.askPending'), false);
    }
  }
});

test('an older request cannot replace a newer answer even in the same scope', async () => {
  const resolvers = [];
  const u = ui(() => new Promise(resolve => resolvers.push(resolve)));
  const old = u.run('ask()');
  u.run('invalidateAsk()');
  const fresh = u.run('ask()');
  resolvers[1]({ answer: 'Newest answer' });
  await fresh;
  resolvers[0]({ answer: 'Old answer' });
  await old;
  assert.equal(u.run('S.ask.answer'), 'Newest answer');
});

test('Ask completion does not overwrite another tab or article', async () => {
  for (const change of ['S.view="evidence"', 'S.article={pmid:"12345"}']) {
    const u = pendingAsk();
    const pending = u.run('ask()');
    u.run(change);
    u.node('#work').innerHTML = 'Current screen';
    u.resolve({ answer: 'Ready to view later' });
    await pending;
    assert.equal(u.node('#work').innerHTML, 'Current screen');
    assert.equal(u.run('S.ask.answer'), 'Ready to view later');
    assert.equal(u.run('S.askPending'), false);
  }
});

test('confirmed Ask scope has no mismatch warning and exposes disclosures', async () => {
  const u = ui(() => ({ answer: 'Verified excerpt', filters: { since_year: 2019, until_year: 2022, concept_id: 'D1' }, disclosures: ['Partial <coverage>'] }));
  await u.run('ask()');
  assert.doesNotMatch(u.node('#work').innerHTML, /did not confirm/);
  assert.match(u.node('#work').innerHTML, /Partial &lt;coverage&gt;/);
});

test('empty, malformed and error analytics payloads remain distinguishable', () => {
  const u = ui();
  assert.match(u.run('panelBody({results:[]},"ignored")'), /No data available/);
  assert.match(u.run('panelBody({},"ignored")'), /no results array/);
  const error = u.run('panelBody({error:"bad <scope>",results:[]},"ignored")');
  assert.match(error, /bad &lt;scope&gt;/);
  assert.doesNotMatch(error, /No data available/);
});

test('HTTP failure and unreadable JSON are reported per panel', async () => {
  const u = ui();
  u.context.fetch = async url => url.includes('/trend')
    ? { ok: false, status: 503, json: async () => ({ detail: 'Trend unavailable' }) }
    : { ok: true, json: async () => { throw Error('not JSON'); } };
  await u.run('S.view="overview";overview()');
  assert.match(u.node('#work').innerHTML, /Trend unavailable/);
  assert.match(u.node('#work').innerHTML, /unreadable response/);
});

// Static contracts only: these do not execute Python or PowerShell and cannot
// establish runtime correctness of argparse, Parquet access or env loading.
test('viewer source delegates selection and preserves documented CLI options (static)', () => {
  const source = readFileSync(join(__dirname, '../view.py'), 'utf8');
  const dataset = readFileSync(join(__dirname, '../pipeline/dataset.py'), 'utf8');
  assert.match(source, /from pipeline\.dataset import paths/);
  assert.match(source, /store, _, _ = paths\(\)/);
  assert.match(source, /os\.environ\["PUBMED_DATASET"\] = args\.dataset/);
  for (const option of ['table', 'rows', '--dataset', '--wide', '--csv']) {
    assert.ok(source.includes(`parser.add_argument("${option}"`), option);
  }
  assert.ok(source.indexOf('os.environ["PUBMED_DATASET"] = args.dataset') < source.indexOf('store, _, _ = paths()'));
  assert.match(dataset, /DEFAULT_MANIFEST = ROOT \/ "covid-files" \/ "dataset.json"/);
  assert.match(dataset, /if not explicit and os\.environ\.get\("PUBMED_STORE"\)/);
  assert.doesNotMatch(source, /STORE\s*=|sys\.argv/);
});

test('launcher source uses only literal PUBMED entries and no implicit consent (static)', () => {
  // NEVER read llm.env. Inspect only the launcher source.
  const source = readFileSync(join(__dirname, '../start_all.ps1'), 'utf8');
  assert.match(source, /Test-Path -LiteralPath \$EnvFile -PathType Leaf/);
  assert.match(source, /\$Line -cnotmatch '\^\\s\*\(PUBMED_\[A-Z0-9_\]\+\)\\s\*=\(\.\*\)\$'/);
  assert.match(source, /GetEnvironmentVariables\('Process'\)/);
  assert.match(source, /OrdinalIgnoreCase/);
  assert.match(source, /if \(\$ExistingNames\.Contains\(\$Name\)\) \{ continue \}/);
  assert.match(source, /SetEnvironmentVariable\(\$Name, \$Value, 'Process'\)/);
  assert.doesNotMatch(source, /Invoke-Expression|iex\s|PUBMED_ALLOW_CLOUD|PUBMED_LLM|PUBMED_API_KEY/);
  assert.doesNotMatch(source, /Write-\w+[^\r\n]*\$(?:Value|Line)(?!\w)/);
  assert.equal((source.match(/^python /gm) || []).length, 1);
  assert.match(source, /manage\.py" serve --dataset "\$Dataset" --port \$Port/);
});

test('README documents runtime limits, selection and explicit consent (static)', () => {
  const readme = readFileSync(join(__dirname, '../README.md'), 'utf8');
  for (const phrase of ['pipeline.dataset.paths()', 'PUBMED_ALLOW_CLOUD=1', '**do not load `llm.env`**', 'not a startup', '**Scope enforcement:**']) {
    assert.ok(readme.includes(phrase), phrase);
  }
  assert.doesNotMatch(readme, /roughly 90% right|95% of the corpus|38\.8%|The Papers tab|switching is one line/);
});

test('missing analytics totals are unavailable, while a genuine zero stays zero', () => {
  const u = ui();
  assert.equal(u.run('total({error:"failed",total_papers:0},"total_papers")'), '—');
  assert.equal(u.run('total({},"total_papers")'), '—');
  assert.equal(u.run('total({total_papers:0},"total_papers")'), '0');
});

test('analytics from an earlier scope cannot replace current results', async () => {
  const pending = [];
  const u = ui(() => new Promise(resolve => pending.push(resolve)));
  u.run('S.view="overview"');
  const old = u.run('overview()');
  u.run('invalidateAsk();S.concept={concept_id:"D2",concept_name:"Two"}');
  const fresh = u.run('overview()');
  pending.slice(4).forEach(resolve => resolve({results:[],total_papers:22}));
  await fresh;
  pending.slice(0,4).forEach(resolve => resolve({results:[],total_papers:11}));
  await old;
  assert.equal(u.run('S.overview.trend.total_papers'),22);
});

test('changing scope while overview is open reloads its analytics', async () => {
  const u = ui(() => ({results:[],total_papers:4}));
  await u.run('S.view="overview";S.concepts=[{concept_id:"D2",concept_name:"Two"}];pick(0)');
  assert.ok(u.requests.some(r=>r.pathname==='/api/trend' && r.searchParams.get('concept_id')==='D2'));
  assert.match(u.node('#work').innerHTML,/Research overview/);
});

test('registry links recognize NCT IDs and reject arbitrary external links', () => {
  const u = ui();
  assert.match(u.run('trialLinks([{accession:"NCT01234567",databank:"ClinicalTrials.gov"}])'), /href="https:\/\/clinicaltrials.gov\/study\/NCT01234567"/);
  assert.doesNotMatch(u.run('trialLinks([{accession:"javascript:alert(1)"}])'), /href=/);
});

test('year chart preserves calendar gaps and gives an exact accessible table', () => {
  const u = ui();
  const chart = u.run('trendChart([{pub_year:2022,papers:12},{pub_year:2020,papers:4}])');
  assert.match(chart,/2021<\/td><td>0/);
  assert.match(chart,/2020 to 2022/);
  assert.match(chart,/View yearly counts/);
  assert.match(chart,/height="0"/);
});

test('an open article cannot be replaced by an older detail request', async () => {
  const pending = [];
  const u = ui(() => new Promise(resolve => pending.push(resolve)));
  const old = u.run('openArticle("1")');
  const fresh = u.run('openArticle("2")');
  pending[1]({pmid:'2',title:'New record'});
  await fresh;
  pending[0]({pmid:'1',title:'Old record'});
  await old;
  assert.equal(u.run('S.article.data.pmid'),'2');
  assert.doesNotMatch(u.node('#work').innerHTML,/Old record/);
});

test('restored overview loads its analytics without performing an evidence search', async () => {
  const u = ui(url => url.pathname==='/api/resolve' ? {concepts:[]} : {results:[],total_papers:0});
  u.node('#q').value='covid'; u.node('#k').value='20';
  await u.run('S.restoreView="overview";search()');
  assert.ok(u.requests.some(r=>r.pathname==='/api/trend'));
  assert.ok(!u.requests.some(r=>r.pathname==='/api/search'));
  assert.match(u.node('#work').innerHTML,/Research overview/);
});
