export const PRIMARY_VIEWS = Object.freeze([
  ["dashboard", "Dashboard", "⌂"],
  ["train", "Train", "▶"],
  ["history", "History", "◷"],
]);

export const MORE_VIEWS = Object.freeze([
  ["rewards", "Rewards", "★"],
  ["terrain", "Terrain", "◇"],
  ["physics", "Physics", "⚙"],
  ["deploy", "Deploy", "⇧"],
  ["detection", "Detection", "◎"],
  ["activity", "Activity", "≋"],
  ["connection", "Connection", "⌁"],
]);

export const ALL_VIEWS = Object.freeze([...PRIMARY_VIEWS, ...MORE_VIEWS]);
export const ALL_VIEW_IDS = Object.freeze(ALL_VIEWS.map(([id]) => id));

export function routeFromLocation(locationValue = globalThis.location) {
  const params = new URLSearchParams(locationValue?.search || "");
  const view = ALL_VIEW_IDS.includes(params.get("view")) ? params.get("view") : "dashboard";
  return {
    view,
    folder: params.get("folder") || "all",
    run: params.get("run") || "",
    search: params.get("search") || "",
    status: params.get("status") || "all",
    sort: params.get("sort") || "",
  };
}

export function syncRouteToLocation(route = {}, { replace = true } = {}) {
  if (!globalThis.history || !globalThis.location) return;
  const url = new URL(globalThis.location.href);
  const values = {
    view: route.view || "dashboard",
    folder: route.folder && route.folder !== "all" ? route.folder : "",
    run: route.run || "",
    search: route.search || "",
    status: route.status && route.status !== "all" ? route.status : "",
    sort: route.sort || "",
  };
  for (const [key, value] of Object.entries(values)) {
    if (value) url.searchParams.set(key, value);
    else url.searchParams.delete(key);
  }
  globalThis.history[replace ? "replaceState" : "pushState"]({}, "", url);
}

export function isMoreView(view) {
  return MORE_VIEWS.some(([id]) => id === view);
}
