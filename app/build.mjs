// Inline each chapter into a single self-contained dist/<page>.html (no fetch, no modules).
// These are the "static bundle" the plan targets — openable via file:// and trivially shareable.
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const read = (p) => readFileSync(join(here, p), "utf8");
const css = read("styles.css");
const model = read("src/model.js").replace(/^export\s+/gm, "");
const arch = read("src/arch.js").replace(/^export\s+/gm, "");

// pages: [html, appModule, artifactJson]
const pages = [
  ["index.html", "src/app_ch0.js", "data/artifact_ch0.json"],
  ["ch1.html", "src/app.js", "data/artifact.json"],
  ["ch2.html", "src/app_ch2.js", "data/artifact_ch2.json"],
  ["ch3.html", "src/app_ch3.js", "data/artifact_ch3.json"],
];

mkdirSync(join(here, "dist"), { recursive: true });
for (const [htmlFile, appFile, artFile] of pages) {
  const artifact = read(artFile);
  const app = read(appFile)
    .replace(/^import[\s\S]*?from\s+"\.\/model\.js(?:\?v=\d+)?";\s*/m, "")
    .replace(/^import\s*\{[^}]*\}\s*from\s+"\.\/arch\.js(?:\?v=\d+)?";\s*/m, "")    // drop model import
    .replace(/await loadArtifact\("data\/[\w.]+\.json"\)(?:\.catch\(\(e\) => \{ showFatal\(e\); throw e; \}\))?/, "window.__ART");   // inline the data

  const html = read(htmlFile)
    .replace(/<link rel="stylesheet" href="styles.css" \/>/, `<style>\n${css}\n</style>`)
    .replace(
      /<script type="module" src="src\/[\w]+\.js(?:\?v=\d+)?"><\/script>/,
      `<script>window.__ART=${artifact};</script>\n<script type="module">\n${model}\n${arch}\n${app}\n</script>`
    )
    // cross-chapter links keep working inside dist/ (files sit side by side)
    ;
  const out = join(here, "dist", htmlFile);
  writeFileSync(out, html);
  console.log(`wrote ${out}  (${(Buffer.byteLength(html) / 1024).toFixed(0)} KB, self-contained)`);
}
