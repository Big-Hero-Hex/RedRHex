# Vendored browser dependencies

Third-party JavaScript served directly to the panel. The panel has no bundler and makes
no network requests at runtime, so browser dependencies are committed here rather than
fetched from a CDN or installed by a package manager.

| File | Package | Version | License | Source |
| --- | --- | --- | --- | --- |
| `three.module.min.js` | three | 0.169.0 | MIT | `https://unpkg.com/three@0.169.0/build/three.module.min.js` |

`three` ships ES modules only; there is no UMD build. `index.html` resolves the bare
`three` specifier through an import map, which is why `robot_view.js` is the panel's one
`<script type="module">`.

To update, re-download the pinned URL with the version bumped, update this table, and
re-run `tools/training_panel/ui_tests/test_local_panel_ui.py`.
