import assert from "node:assert/strict";
import test from "node:test";

import {
  REMOTE_PROTOCOL_VERSION,
  buildDeployPayload,
  buildTrainingJob,
  checkpointReference,
  normalizePhysicsValues,
  remoteCompatibility,
  sanitizeTrainingParams,
} from "./core.js";

test("route-specific training payloads omit irrelevant and host-path fields", () => {
  const full = sanitizeTrainingParams({
    training_route: "sensor_v2_full",
    task: "ignored",
    max_iterations: 99,
    checkpoint: "/host/model.pt",
    checkpoint_ref: { run_id: "ignored", checkpoint_iteration: 10 },
    teacher_iterations: 20,
    distillation_iterations: 30,
    ppo_iterations: 40,
  });
  assert.equal(full.training_route, "sensor_v2_full");
  assert.equal(full.spring_backend, "native");
  assert.equal(full.teacher_iterations, 20);
  assert.equal(full.checkpoint, undefined);
  assert.equal(full.checkpoint_ref, undefined);
  assert.equal(full.task, undefined);
  assert.equal(full.max_iterations, undefined);

  const f2 = sanitizeTrainingParams({
    training_route: "sensor_v2_distillation",
    task: "ignored",
    max_iterations: 80,
    checkpoint_ref: { run_id: "run-a", checkpoint_iteration: 120 },
  });
  assert.deepEqual(f2.checkpoint_ref, { run_id: "run-a", checkpoint_iteration: 120 });
  assert.equal(f2.task, undefined);
});

test("checkpoint references reject malformed iterations", () => {
  assert.deepEqual(checkpointReference("run-a", 0), { run_id: "run-a", checkpoint_iteration: 0 });
  assert.throws(() => checkpointReference("run-a", "1.5"), /non-negative integer/);
  assert.throws(() => checkpointReference("", 1), /run is required/);
});

test("training jobs carry stable idempotency and normalized Physics", () => {
  const job = buildTrainingJob({
    machineId: "mother-a",
    params: { training_route: "standard", task: "Task-v0", num_envs: 8, max_iterations: 20 },
    preset: { id: "reward", values: { reward: 1 } },
    terrainPreset: { id: "terrain", values: { terrain: "plane" } },
    physicsPreset: { id: "physics", values: { stiffness: "4.5", unknown: 9 } },
    physicsSchema: [{ key: "stiffness", min: 0, max: 10 }],
    role: "operator",
    userId: "actor",
    clientRequestId: "request-1",
  });
  assert.equal(job.client_request_id, "request-1");
  assert.equal(job.payload.client_request_id, "request-1");
  assert.equal(job.payload.spring_backend, "native");
  assert.deepEqual(job.payload.physics_overrides, { stiffness: 4.5 });
});

test("Physics normalization is sparse, bounded, and schema allowlisted", () => {
  const schema = [
    { key: "a", min: 0, max: 2 },
    { key: "b", min: null, max: null },
  ];
  assert.deepEqual(normalizePhysicsValues({ a: 1.5, b: "-2", c: 9 }, schema), { a: 1.5, b: -2 });
  assert.deepEqual(normalizePhysicsValues({ a: 4, b: "nope" }, schema), {});
});

test("deploy payload accepts only fixed actions and enumerated scenarios", () => {
  assert.deepEqual(
    buildDeployPayload("record_mujoco_video", { runId: "run-a", scenario: "stand_zero", allowedScenarios: ["stand_zero"] }),
    { run_id: "run-a", scenario: "stand_zero" },
  );
  assert.throws(
    () => buildDeployPayload("record_mujoco_video", { runId: "run-a", scenario: "../../shell", allowedScenarios: ["stand_zero"] }),
    /repository-owned/,
  );
  assert.throws(() => buildDeployPayload("viewer", { runId: "run-a" }), /Unsupported/);
});

test("older infrastructure is inspection-only with exact recovery guidance", () => {
  const old = remoteCompatibility({ panel_version: "3.6.4-drive-export" }, null);
  assert.equal(old.mode, "read-only");
  assert.match(old.message, /20260814_370_remote_parity.sql/);
  assert.match(old.message, /restart the worker/);

  const ready = remoteCompatibility(
    { remote_protocol_version: REMOTE_PROTOCOL_VERSION },
    { protocol_version: REMOTE_PROTOCOL_VERSION },
  );
  assert.equal(ready.mode, "read-write");
});
