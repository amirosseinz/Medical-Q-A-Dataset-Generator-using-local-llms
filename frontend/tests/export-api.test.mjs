import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import ts from 'typescript';

// Compile the real client so these tests exercise its download entry point.
const source = await readFile(new URL('../src/lib/api.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText;
const { exportApi } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`);
const config = { format: 'jsonl', train_split: 0.8, val_split: 0.1, test_split: 0.1, split_strategy: 'source' };

function browser(t, responses) {
  const calls = [], downloads = [], revoked = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    calls.push({ url, options });
    assert.ok(responses.length, 'Unexpected fetch');
    return responses.shift();
  });
  const originalDocument = Object.getOwnPropertyDescriptor(globalThis, 'document');
  Object.defineProperty(globalThis, 'document', { configurable: true, value: {
    createElement: () => ({ click() { downloads.push(this.download); } }),
  } });
  t.after(() => {
    if (originalDocument) Object.defineProperty(globalThis, 'document', originalDocument);
    else delete globalThis.document;
  });
  t.mock.method(URL, 'createObjectURL', () => 'blob:test');
  t.mock.method(URL, 'revokeObjectURL', (url) => revoked.push(url));
  return { calls, downloads, revoked };
}
const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'Content-Type': 'application/json' },
});

test('source strategy reaches export API and returns actual split counts after downloads', async (t) => {
  const result = { total_pairs: 10, files: { train: 'train.jsonl', test: 'test.jsonl' },
    download_base: '/download', split_counts: { train: 8, val: 0, test: 2 } };
  const browserState = browser(t, [json(result), new Response('train'), new Response('test')]);
  assert.deepEqual(await exportApi.download('project', config), result);
  assert.equal(JSON.parse(browserState.calls[0].options.body).split_strategy, 'source');
  assert.deepEqual(browserState.downloads, ['train.jsonl', 'test.jsonl']);
  assert.equal(browserState.revoked.length, 2);
});

test('a failed split download rejects with its filename', async (t) => {
  const browserState = browser(t, [json({ total_pairs: 10, files: { train: 'train.jsonl', test: 'test.jsonl' },
    download_base: '/download' }), new Response('train'), new Response('failed', { status: 503 })]);
  await assert.rejects(exportApi.download('project', config), /Failed to download test.jsonl/);
  assert.deepEqual(browserState.downloads, ['train.jsonl']);
});

test('incomplete-source errors reach the caller before any download', async (t) => {
  const browserState = browser(t, [json({ detail: 'Row 1 requires complete cited source IDs.' }, 422)]);
  await assert.rejects(exportApi.download('project', config), /complete cited source IDs/);
  assert.deepEqual(browserState.downloads, []);
});

test('empty split file list cannot report success', async (t) => {
  browser(t, [json({ total_pairs: 0, files: {}, download_base: '/download' })]);
  await assert.rejects(exportApi.download('project', config), /no download URL/);
});

test('single-file export remains supported', async (t) => {
  const result = { total_pairs: 10, filename: 'all.jsonl', download_url: '/download/all.jsonl' };
  const browserState = browser(t, [json(result), new Response('all')]);
  assert.deepEqual(await exportApi.download('project', { ...config, train_split: 1, val_split: 0, test_split: 0 }), result);
  assert.deepEqual(browserState.downloads, ['all.jsonl']);
});
