// Node check: the browser's forward pass must reproduce the pipeline's reference.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { thetaAt, topk, forgetTargetProb, retainAccuracy, selfTest } from "./model.js";

const here = dirname(fileURLToPath(import.meta.url));
const art = JSON.parse(readFileSync(join(here, "..", "data", "artifact.json"), "utf8"));

console.log("self-test max |live - reference| =", selfTest(art).toExponential(3));
console.log("\nalpha  p(Kyiv)  retain  top-3 for 'capital of ukraine is ___'");
for (const a of [0, 0.25, 0.5, 0.75, 1.0, 1.25]) {
  const th = thetaAt(art, a);
  const { top } = topk(art, th, art.prompts.forget.ids, 3);
  const t = top.map((x) => `${x.token} ${(x.p * 100).toFixed(0)}%`).join(", ");
  console.log(
    `${a.toFixed(2)}   ${(forgetTargetProb(art, th) * 100).toFixed(1).padStart(5)}%  ` +
    `${(retainAccuracy(art, th) * 100).toFixed(0).padStart(4)}%   ${t}`
  );
}
