/* Version sync for iOS home-screen launches and restored browser tabs.
 * Checks server VERSION on activation and reloads stale HTML using a unique URL.
 * A pre-existing cached HTML shell cannot run new code until it is reloaded once.
 */
(function () {
  'use strict';
  const current = document.querySelector('meta[name="app-version"]')?.content?.trim();
  const script = document.currentScript;
  if (!current || !script?.src) return;
  const endpoint = new URL('../VERSION', script.src);
  const badge = document.querySelector('.version');
  let inFlight = false;
  let lastChecked = 0;
  const attemptKey = 'loto-analysis-version-reload';

  function compareVersions(a, b) {
    if (!/^\d+\.\d+\.\d+$/.test(a) || !/^\d+\.\d+\.\d+$/.test(b)) return 0;
    const aa = a.split('.').map(Number), bb = b.split('.').map(Number);
    for (let i = 0; i < 3; i++) {
      if (aa[i] !== bb[i]) return aa[i] > bb[i] ? 1 : -1;
    }
    return 0;
  }
  function navigate(version, force) {
    const now = Date.now();
    if (!force) {
      try {
        const last = JSON.parse(sessionStorage.getItem(attemptKey) || 'null');
        if (last?.version === version && now - last.at < 30000) return;
        sessionStorage.setItem(attemptKey, JSON.stringify({ version, at: now }));
      } catch (_) {}
    }
    const url = new URL(location.href);
    url.searchParams.set('v', version);
    url.searchParams.set('_reload', String(now));
    location.replace(url.href);
  }
  async function check(force = false) {
    if (inFlight || (!force && document.visibilityState === 'hidden')) return;
    if (!force && Date.now() - lastChecked < 4000) return;
    inFlight = true;
    lastChecked = Date.now();
    try {
      const url = new URL(endpoint.href);
      url.searchParams.set('_t', String(Date.now()));
      const res = await fetch(url.href, { cache: 'no-store', redirect: 'follow' });
      if (!res.ok) throw new Error('VERSION request failed');
      const published = (await res.text()).trim();
      if (compareVersions(published, current) > 0) {
        navigate(published, force);
        return;
      }
      if (force) navigate(current, true);
    } catch (_) {
      if (force) navigate(current, true);
    } finally {
      inFlight = false;
    }
  }
  if (badge) {
    badge.textContent = 'App v' + current + ' ↻';
    badge.setAttribute('role', 'button');
    badge.setAttribute('tabindex', '0');
    badge.setAttribute('aria-label', '最新版を確認して画面を更新');
    badge.style.cursor = 'pointer';
    badge.addEventListener('click', () => check(true));
    badge.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); check(true); }
    });
  }
  window.addEventListener('pageshow', () => check());
  window.addEventListener('focus', () => check());
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') check();
  });
  if (document.readyState === 'complete') check();
})();
