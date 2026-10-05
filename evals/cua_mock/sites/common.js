// Shared helpers for the mock sites: run-scoped browser state and the oracle log.
const Mock = {
  log(site, type, data = {}) {
    fetch("/api/log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // The server drops events from a tab opened before the last reset.
      body: JSON.stringify({ site, type, ...data, run: localStorage.getItem("mock.run") }),
      keepalive: true,
    });
  },
  // Drop state from an earlier run, then call ready(). Every page waits on this.
  async start(ready) {
    try {
      const { run } = await (await fetch("/api/run")).json();
      if (localStorage.getItem("mock.run") !== run) {
        localStorage.clear();
        localStorage.setItem("mock.run", run);
      }
    } catch (e) {}
    ready();
  },
  get(key, fallback) {
    const raw = localStorage.getItem(key);
    return raw === null ? fallback : JSON.parse(raw);
  },
  set(key, value) {
    localStorage.setItem(key, JSON.stringify(value));
  },
  money(n) {
    return "$" + n.toFixed(2);
  },
};
