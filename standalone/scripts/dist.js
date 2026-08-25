// 体験版 / 製品版のインストーラを electron-builder で出力する。
//
//   node scripts/dist.js full    → release/full/BadAutoAlign-<ver>-full-setup.exe
//   node scripts/dist.js trial   → release/trial/BadAutoAlign Trial-<ver>-trial-setup.exe
//   node scripts/dist.js all     → 両方
//
// エディションは package.json の "edition" に extraMetadata として焼き込まれ、
// main.js が実行時に参照する（体験版は保存不可）。appId / productName を分けて
// 両エディションを同一PCに共存インストールできるようにしている。
// 前提: ../dist/vat-serve にPyInstaller済みバックエンドがあること（README参照）。

const path = require("node:path");
const fs = require("node:fs");
const builder = require("electron-builder");

const EDITIONS = {
  full: {
    appId: "com.badgateway503.badautoalign",
    productName: "BadAutoAlign",
  },
  trial: {
    appId: "com.badgateway503.badautoalign.trial",
    productName: "BadAutoAlign Trial",
  },
};

async function buildEdition(edition) {
  const ed = EDITIONS[edition];
  if (!ed) throw new Error(`未知のエディション: ${edition}（full | trial | all）`);
  const backend = path.resolve(__dirname, "..", "..", "dist", "vat-serve", "vat-serve.exe");
  if (!fs.existsSync(backend)) {
    throw new Error(`バックエンドが見つかりません: ${backend}\n` +
      "先に uv run python -m PyInstaller build/vat-serve.spec --distpath dist --workpath build/pyi -y を実行してください");
  }
  console.log(`\n=== ${edition} 版をビルド (${ed.productName}) ===`);
  await builder.build({
    targets: builder.Platform.WINDOWS.createTarget("nsis"),
    config: {
      appId: ed.appId,
      productName: ed.productName,
      extraMetadata: { edition },
      artifactName: `\${productName}-\${version}-${edition}-setup.\${ext}`,
      directories: { output: `release/${edition}` },
    },
  });
}

(async () => {
  const which = process.argv[2] || "all";
  const list = which === "all" ? ["full", "trial"] : [which];
  for (const ed of list) await buildEdition(ed);
  console.log("\n完了: standalone/release/{full,trial}/ を確認してください");
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
