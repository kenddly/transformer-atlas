/* Apply a saved or system theme before styles render, then wire the toggle. */
(() => {
  'use strict';
  const key = 'transformer-atlas-theme';
  const system = window.matchMedia('(prefers-color-scheme: dark)');
  let choice;
  try {
    const saved = localStorage.getItem(key);
    if (saved === 'light' || saved === 'dark') choice = saved;
  } catch { /* The toggle still works when browser storage is unavailable. */ }

  function apply(theme) {
    const dark = theme === 'dark';
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.content = dark ? '#252d29' : '#f8f7f3';
    const toggle = document.getElementById('theme-toggle');
    if (toggle) {
      toggle.setAttribute('aria-pressed', String(dark));
      toggle.title = `Switch to ${dark ? 'light' : 'dark'} mode`;
    }
  }

  apply(choice || (system.matches ? 'dark' : 'light'));
  document.addEventListener('DOMContentLoaded', () => {
    apply(document.documentElement.dataset.theme);
    document.getElementById('theme-toggle')?.addEventListener('click', () => {
      choice = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      try { localStorage.setItem(key, choice); } catch {}
      apply(choice);
    });
  });
  system.addEventListener('change', () => {
    if (!choice) apply(system.matches ? 'dark' : 'light');
  });
  window.addEventListener('storage', event => {
    if (event.key !== key && event.key !== null) return;
    choice = event.newValue === 'dark' || event.newValue === 'light' ? event.newValue : undefined;
    apply(choice || (system.matches ? 'dark' : 'light'));
  });
})();
