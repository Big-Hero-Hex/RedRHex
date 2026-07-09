import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const remoteWebDir = dirname(fileURLToPath(import.meta.url));

function localAssetPath(reference) {
  if (!reference.startsWith("./")) return "";
  return reference.slice(0, reference.search(/[?#]/) === -1 ? undefined : reference.search(/[?#]/));
}

test("index.html only references files that exist in remote_web", () => {
  const html = readFileSync(resolve(remoteWebDir, "index.html"), "utf-8");
  const references = [
    ...[...html.matchAll(/\b(?:href|src)="([^"]+)"/g)].map((match) => match[1]),
    ...[...html.matchAll(/\bimport\("([^"]+)"\)/g)].map((match) => match[1]),
  ];
  const missing = references
    .map(localAssetPath)
    .filter(Boolean)
    .filter((assetPath) => !existsSync(resolve(remoteWebDir, assetPath)));

  assert.deepEqual(missing, []);
});
