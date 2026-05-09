import { evaluate, formatDiagnostic } from './rml-playground-runtime.mjs';
import { PLAYGROUND_EXAMPLES, defaultExampleId, findExample } from './examples.mjs';
import {
  decodePlaygroundState,
  encodePlaygroundState,
  normalizePlaygroundState,
} from './url-state.mjs';

const editor = document.querySelector('#source-editor');
const exampleSelect = document.querySelector('#example-select');
const runButton = document.querySelector('#run-button');
const copyLinkButton = document.querySelector('#copy-link-button');
const resetButton = document.querySelector('#reset-button');
const statusLine = document.querySelector('#status-line');
const resultsList = document.querySelector('#results-list');
const diagnosticsBlock = document.querySelector('#diagnostics-block');

let suppressHashSync = false;
let hashTimer = null;

function formatValue(value) {
  if (typeof value === 'string') return value;
  if (!Number.isFinite(value)) return String(value);
  return String(+value.toFixed(6)).replace(/\.0+$/g, '');
}

function stateFromControls() {
  return normalizePlaygroundState({
    example: exampleSelect.value,
    source: editor.value,
  });
}

function replaceHashFromControls() {
  if (suppressHashSync) return;
  const hash = encodePlaygroundState(stateFromControls());
  if (window.location.hash !== hash) {
    window.history.replaceState(null, '', hash);
  }
}

function scheduleHashUpdate() {
  window.clearTimeout(hashTimer);
  hashTimer = window.setTimeout(replaceHashFromControls, 150);
}

function setStatus(text, tone = 'neutral') {
  statusLine.textContent = text;
  statusLine.dataset.tone = tone;
}

function clearRenderedOutput() {
  resultsList.replaceChildren();
  diagnosticsBlock.textContent = '';
  diagnosticsBlock.hidden = true;
}

function renderResults(results) {
  resultsList.replaceChildren();
  if (results.length === 0) {
    const empty = document.createElement('li');
    empty.className = 'empty-result';
    empty.textContent = 'No query results';
    resultsList.append(empty);
    return;
  }
  for (const result of results) {
    const item = document.createElement('li');
    item.textContent = formatValue(result);
    resultsList.append(item);
  }
}

function renderDiagnostics(diagnostics, source) {
  if (diagnostics.length === 0) {
    diagnosticsBlock.textContent = '';
    diagnosticsBlock.hidden = true;
    return;
  }
  diagnosticsBlock.hidden = false;
  diagnosticsBlock.textContent = diagnostics
    .map((diagnostic) => formatDiagnostic(diagnostic, source))
    .join('\n\n');
}

function runSource() {
  clearRenderedOutput();
  const source = editor.value;
  try {
    const out = evaluate(source, { file: 'playground.lino' });
    renderResults(out.results);
    renderDiagnostics(out.diagnostics, source);
    if (out.diagnostics.length > 0) {
      setStatus(`${out.diagnostics.length} diagnostic${out.diagnostics.length === 1 ? '' : 's'}`, 'error');
    } else {
      setStatus(`${out.results.length} result${out.results.length === 1 ? '' : 's'}`, 'ok');
    }
  } catch (err) {
    resultsList.replaceChildren();
    diagnosticsBlock.hidden = false;
    diagnosticsBlock.textContent = err && err.stack ? err.stack : String(err);
    setStatus('Runtime error', 'error');
  }
  replaceHashFromControls();
}

function applyState(state, { run = true } = {}) {
  const fallback = findExample(defaultExampleId);
  const selected = findExample(state?.example || fallback.id);
  suppressHashSync = true;
  exampleSelect.value = selected.id;
  editor.value = state?.source || selected.source;
  suppressHashSync = false;
  if (run) runSource();
  else replaceHashFromControls();
}

function loadInitialState() {
  const decoded = decodePlaygroundState(window.location.hash);
  if (decoded?.source) return decoded;
  const example = findExample(defaultExampleId);
  return { example: example.id, source: example.source };
}

function populateExamples() {
  for (const example of PLAYGROUND_EXAMPLES) {
    const option = document.createElement('option');
    option.value = example.id;
    option.textContent = example.title;
    exampleSelect.append(option);
  }
}

function copyShareLink() {
  replaceHashFromControls();
  const text = window.location.href;
  const done = () => {
    copyLinkButton.textContent = 'Copied';
    window.setTimeout(() => {
      copyLinkButton.textContent = 'Copy link';
    }, 1200);
  };
  const fallback = () => {
    window.prompt('Copy link', text);
    done();
  };
  if (navigator.clipboard?.writeText) {
    navigator.clipboard.writeText(text).then(done, fallback);
  } else {
    fallback();
  }
}

populateExamples();
applyState(loadInitialState());

runButton.addEventListener('click', runSource);
copyLinkButton.addEventListener('click', copyShareLink);
resetButton.addEventListener('click', () => {
  const example = findExample(exampleSelect.value);
  applyState({ example: example.id, source: example.source });
});
exampleSelect.addEventListener('change', () => {
  const example = findExample(exampleSelect.value);
  applyState({ example: example.id, source: example.source });
});
editor.addEventListener('input', () => {
  setStatus('Edited', 'neutral');
  scheduleHashUpdate();
});
window.addEventListener('hashchange', () => {
  const decoded = decodePlaygroundState(window.location.hash);
  if (decoded?.source) applyState(decoded);
});
