import assert from "node:assert/strict";
import test from "node:test";

function installMemoryStorage() {
  const values = new Map();
  globalThis.localStorage = {
    getItem(key) {
      return values.get(key) || "";
    },
    setItem(key, value) {
      values.set(key, String(value));
    },
    removeItem(key) {
      values.delete(key);
    },
  };
}

test("signIn explains Supabase network failures with config guidance", async () => {
  installMemoryStorage();
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new TypeError("Failed to fetch");
  };

  try {
    const { signIn } = await import("./api.js");
    await assert.rejects(
      () => signIn("team@example.com", "bad-password"),
      (error) => {
        assert.match(error.message, /Cannot reach Supabase/);
        assert.match(error.message, /tqvopodmsprhujyagaan\.supabase\.co/);
        assert.match(error.message, /remote_web\/config\.js/);
        assert.match(error.message, /Failed to fetch/);
        return true;
      },
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});
