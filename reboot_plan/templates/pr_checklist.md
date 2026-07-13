# Core-reboot change checklist

<!-- Paste into the PR description (or run through it mentally when self-merging).
     This checklist is the stand-in for GPU CI — do not skip the preflight line. -->

- [ ] One issue/step per commit; messages name the plan step or finding
- [ ] Current `STATUS.md` gate and exact required commands identified
- [ ] Focused test and required aggregate CPU tier green; no test weakened/skipped
- [ ] Frozen tree guard green; no panel/remote/reward-agent/ROS source changed
- [ ] Isaac change? Runtime provenance passed and required local adapter/sim gate ran
- [ ] Contract/layout change? Read-only legacy/ROS parity result recorded
- [ ] Behavior-changing? Kept out of extraction or explicitly approved with ADR and rerun
- [ ] New failure mode discovered? → regression test added in the cheapest tier
- [ ] `STATUS.md` and evidence links updated only if the gate truly advanced
- [ ] Independent review findings addressed or explicitly accepted with a reason
