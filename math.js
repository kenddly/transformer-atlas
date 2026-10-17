/* Local TeX rendering for lesson content and changing laboratory calculations. */
(() => {
  'use strict';
  const ignored = '.katex, .katex-error, .no-math, pre, code, textarea, input, select, option, svg, math, script, style';
  const explicit = '.math-block, .math-display[data-latex], .math-inline[data-latex], .equation[data-latex]';
  const sources = new WeakMap();
  const options = {
    output: 'htmlAndMathml',
    throwOnError: false,
    trust: false,
    strict: 'ignore',
    maxExpand: 1000,
    maxSize: 20,
    errorColor: '#ad765d'
  };
  let observer;
  let scheduled = false;
  const pending = new Set();

  function typeset(root) {
    if (!window.katex || !window.renderMathInElement || !root) return;
    if (root.nodeType === Node.TEXT_NODE) root = root.parentElement;
    if (!root || root.closest(ignored)) return;
    const blocks = [...root.querySelectorAll(explicit)];
    if (root.matches(explicit)) blocks.unshift(root);
    blocks.forEach(block => {
      const hasMath = !!block.querySelector('.katex, .katex-error');
      const source = block.dataset.latex ?? (hasMath ? sources.get(block) : block.textContent);
      if (source == null || (hasMath && sources.get(block) === source)) return;
      sources.set(block, source);
      window.katex.render(source.trim(), block, {...options, displayMode: !block.classList.contains('math-inline')});
      block.classList.add('math-rendered');
    });
    window.renderMathInElement(root, {
      ...options,
      delimiters: [
        {left: '\\[', right: '\\]', display: true},
        {left: '\\(', right: '\\)', display: false}
      ],
      ignoredTags: ['script', 'noscript', 'style', 'textarea', 'input', 'select', 'pre', 'code', 'option', 'svg', 'math'],
      ignoredClasses: ['katex', 'katex-error', 'math-rendered', 'no-math']
    });
  }

  function render(root = document.getElementById('main')) {
    observer?.disconnect();
    try { typeset(root); }
    finally { observer?.observe(document.body, {childList: true, subtree: true, characterData: true}); }
  }

  window.AtlasMath = Object.freeze({render});
  observer = new MutationObserver(records => {
    records.forEach(record => {
      const target = record.target.nodeType === Node.ELEMENT_NODE ? record.target : record.target.parentElement;
      if (!target || target.closest(ignored)) return;
      pending.add(target);
    });
    if (scheduled || !pending.size) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      const roots = [...pending].filter(root => root.isConnected);
      pending.clear();
      observer.disconnect();
      try {
        roots.filter(root => !roots.some(other => other !== root && other.contains(root))).forEach(typeset);
      } finally {
        observer.observe(document.body, {childList: true, subtree: true, characterData: true});
      }
    });
  });
  render(document.body);
})();
