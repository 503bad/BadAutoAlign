// Electronメインプロセス: ウィンドウ管理とPythonバックエンド(vat serve)の橋渡し。
//
// バックエンドはstdio上の行区切りJSON-RPC（src/vat/service.py参照）。
// 開発時は `uv --directory <リポジトリルート> run vat serve` で起動する。
// 配布時は同梱Pythonのパスを環境変数 VAT_SERVE_CMD で差し替える想定。

const { app, BrowserWindow, Menu, dialog, ipcMain } = require("electron");
const { spawn } = require("node:child_process");
const fs = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const readline = require("node:readline");

const REPO_ROOT = path.resolve(__dirname, "..");

// エディション: "full"（製品版） | "trial"（体験版: 保存不可）
// 配布時は scripts/dist.js が package.json に "edition" を焼き込む。
// 開発時は `electron . --edition=trial` または環境変数 VAT_EDITION で切替。
const EDITION = (() => {
  const arg = process.argv.find((a) => a.startsWith("--edition="));
  if (arg) return arg.split("=")[1];
  if (process.env.VAT_EDITION) return process.env.VAT_EDITION;
  try { return require("./package.json").edition || "full"; } catch { return "full"; }
})();
const IS_TRIAL = EDITION === "trial";
const EDITION_LABEL = IS_TRIAL ? "体験版" : "製品版";

// ---------------------------------------------------------------- backend

class VatBackend {
  constructor() {
    this.proc = null;
    this.pending = new Map(); // id -> {resolve, reject}
    this.nextId = 1;
    this.progress = null;        // {done, total}
    this.progressSender = null;  // 進捗の通知先webContents
  }

  ensureStarted() {
    if (this.proc) return;
    const custom = process.env.VAT_SERVE_CMD;
    const [cmd, ...args] = custom
      ? custom.split(" ")
      : app.isPackaged
        // 配布パッケージ: PyInstallerでexe化したバックエンドを同梱
        ? [path.join(process.resourcesPath, "vat-serve", "vat-serve.exe")]
        : ["uv", "--directory", REPO_ROOT, "run", "vat", "serve"];
    this.proc = spawn(cmd, args, {
      stdio: ["pipe", "pipe", "pipe"],
      // Windows(cp932等)でもJSON-RPCと日本語ログをUTF-8で受け取る
      env: { ...process.env, PYTHONUTF8: "1" },
    });

    const rl = readline.createInterface({ input: this.proc.stdout });
    rl.on("line", (line) => {
      let msg;
      try {
        msg = JSON.parse(line);
      } catch {
        console.error("[vat] 不正な応答:", line);
        return;
      }
      const waiter = this.pending.get(msg.id);
      if (!waiter) return;
      this.pending.delete(msg.id);
      if (msg.ok) waiter.resolve(msg.result);
      else waiter.reject(new Error(msg.error));
    });

    // stderr（処理ログ）を行単位でパースし、フレーズ進捗を通知する
    const rlErr = readline.createInterface({ input: this.proc.stderr });
    rlErr.on("line", (line) => {
      process.stderr.write(`[vat] ${line}\n`);
      if (!this.progressSender) return;
      const total = line.match(/対応 (\d+)$/);
      if (total) {
        this.progress = { done: 0, total: parseInt(total[1], 10) };
        this.progressSender.send("vat:progress", this.progress);
      } else if (/フレーズ .*:\s*timing=/.test(line) && this.progress) {
        this.progress.done += 1;
        this.progressSender.send("vat:progress", this.progress);
      } else {
        // フレーズ進捗が確定する前の待ち時間に「何をしているか」を見せる
        this.progressSender.send("vat:progress", { log: line });
      }
    });
    this.proc.on("exit", (code) => {
      for (const { reject } of this.pending.values()) {
        reject(new Error(`バックエンドが終了しました (code=${code})`));
      }
      this.pending.clear();
      this.proc = null;
    });
  }

  call(method, params) {
    this.ensureStarted();
    const id = this.nextId++;
    const payload = JSON.stringify({ id, method, params }) + "\n";
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.proc.stdin.write(payload);
    });
  }

  stop() {
    if (this.proc) {
      this.proc.stdin.end();
      this.proc.kill();
      this.proc = null;
    }
  }
}

const backend = new VatBackend();

// ---------------------------------------------------------------- ipc

ipcMain.handle("vat:version", () => backend.call("version", {}));

// ガイドのノート列（MIDIガイドのレーン表示用）
ipcMain.handle("vat:guideNotes", (_ev, guidePath) =>
  backend.call("guide_notes", { guide: guidePath }));

// ---------------------------------------------------------------- settings

const SETTINGS_PATH = () => path.join(app.getPath("userData"), "settings.json");

async function loadSettings() {
  try {
    return JSON.parse(await fs.readFile(SETTINGS_PATH(), "utf-8"));
  } catch {
    return {};
  }
}

async function saveSettings(patch) {
  const cur = await loadSettings();
  const next = { ...cur, ...patch };
  await fs.mkdir(path.dirname(SETTINGS_PATH()), { recursive: true });
  await fs.writeFile(SETTINGS_PATH(), JSON.stringify(next, null, 2), "utf-8");
  return next;
}

// MIDIガイドのβ版告知（「次回から表示しない」を settings.json に保存）
ipcMain.handle("ui:midiBetaNotice", async (ev) => {
  const settings = await loadSettings();
  if (settings.midiBetaNoticeDismissed) return { shown: false };
  const win = BrowserWindow.fromWebContents(ev.sender);
  const { checkboxChecked } = await dialog.showMessageBox(win, {
    type: "info",
    title: "MIDIガイド（ベータ版）",
    message: "MIDIでのガイドはベータ版です",
    detail:
      "MIDIにはガイド音声が無いため、タイミング補正はノートの先頭を「芯」とみなし、" +
      "ボーカル側の芯検出のみで位置を合わせます（WAVガイドで使う包絡相関による" +
      "密なラグ推定は行われません）。\n" +
      "ピッチ補正はWAVガイドと同じ挙動です。結果は必ず試聴して確認してください。",
    buttons: ["OK"],
    checkboxLabel: "次回から表示しない",
    checkboxChecked: false,
  });
  if (checkboxChecked) await saveSettings({ midiBetaNoticeDismissed: true });
  return { shown: true };
});

ipcMain.handle("vat:process", async (ev, params) => {
  if (!params.output) {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), "vat-"));
    params.output = path.join(dir, "corrected.wav");
  }
  backend.ensureStarted();
  backend.progress = null;
  backend.progressSender = ev.sender;
  try {
    const report = await backend.call("process", params);
    return { report, output: params.output };
  } finally {
    backend.progressSender = null;
  }
});

ipcMain.handle("app:edition", () => ({
  edition: EDITION, trial: IS_TRIAL, label: EDITION_LABEL,
  name: app.getName(), version: app.getVersion(),
}));

// 補正済み音源を保存ダイアログ経由でコピー保存する（体験版は不可）
ipcMain.handle("file:saveAs", async (ev, { sourcePath, vocalPath }) => {
  if (IS_TRIAL) {
    await dialog.showMessageBox(BrowserWindow.fromWebContents(ev.sender), {
      type: "info",
      title: "体験版",
      message: "体験版につきデータは保存できません。",
      detail: "補正結果の保存は製品版でご利用いただけます。",
      buttons: ["OK"],
    });
    return { saved: false, trial: true };
  }
  const base = vocalPath
    ? path.basename(vocalPath, path.extname(vocalPath))
    : "vocal";
  const defaultPath = path.join(
    vocalPath ? path.dirname(vocalPath) : app.getPath("music"),
    `${base}_corrected.wav`,
  );
  const win = BrowserWindow.fromWebContents(ev.sender);
  const { canceled, filePath } = await dialog.showSaveDialog(win, {
    title: "補正済み音源を保存",
    defaultPath,
    filters: [{ name: "WAV", extensions: ["wav"] }],
  });
  if (canceled || !filePath) return { saved: false };
  await fs.copyFile(sourcePath, filePath);
  return { saved: true, filePath };
});

ipcMain.handle("file:read", async (_ev, filePath) => {
  const buf = await fs.readFile(filePath);
  // ArrayBufferとして返す（レンダラでdecodeAudioDataに渡す）
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
});

// ---------------------------------------------------------------- menu

const LICENSE_TEXT = `${app.getName()} v${app.getVersion()}（${EDITION_LABEL}）
© 2026 503 bad gateway
本アプリ本体は MIT License で提供されます。

主なサードパーティコンポーネント:
・Electron / Node.js — MIT、Chromium — BSD系
・Python / NumPy / SciPy / librosa / scikit-learn / numba — PSF / BSD / ISC
・WORLD (pyworld) — 修正BSD、pretty_midi / mido — MIT
・Signalsmith Stretch — MIT (c) Geraint Luff / Signalsmith Audio Ltd.
・libsndfile (python-soundfile)、libsoxr (python-soxr)、FFmpeg (Electron) — LGPL-2.1+
  （改変せず、差し替え可能な独立ファイルとして同梱）

全リストと各ライセンス条件は同梱の THIRD_PARTY_NOTICES.md を参照してください。`;

// 同梱の THIRD_PARTY_NOTICES.md（配布時は resources/、開発時はリポジトリルート）
function noticesPath() {
  const cands = [
    path.join(process.resourcesPath || "", "THIRD_PARTY_NOTICES.md"),
    path.join(REPO_ROOT, "THIRD_PARTY_NOTICES.md"),
  ];
  return cands.find((p) => require("node:fs").existsSync(p)) || null;
}

async function showLicense(win) {
  const { response } = await dialog.showMessageBox(win, {
    type: "info",
    title: "ライセンス表記",
    message: `${app.getName()} のライセンス`,
    detail: LICENSE_TEXT,
    buttons: ["閉じる", "サードパーティ表記の全文を開く"],
    defaultId: 0,
  });
  if (response === 1) {
    const p = noticesPath();
    if (p) require("electron").shell.openPath(p);
  }
}

function buildMenu() {
  const template = [
    {
      label: "ライセンス",
      submenu: [
        {
          label: "ライセンス表記",
          click: (_item, win) => showLicense(win),
        },
        {
          label: `${app.getName()} について`,
          click: (_item, win) => {
            dialog.showMessageBox(win, {
              type: "info",
              title: `${app.getName()} について`,
              message: `${app.getName()} v${app.getVersion()}（${EDITION_LABEL}）`,
              detail: "開発: 503 bad gateway\n" +
                "MIDI/WAVガイドによるボーカルのタイミング・ピッチ一括補正" +
                (IS_TRIAL ? "\n体験版では補正結果の保存はできません。" : ""),
              buttons: ["閉じる"],
            });
          },
        },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

// ---------------------------------------------------------------- window

function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 900,
    title: `${app.getName()}${IS_TRIAL ? "（体験版）" : ""}`,
    backgroundColor: "#16181d",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  win.loadFile(path.join(__dirname, "renderer", "index.html"));
}

app.whenReady().then(() => {
  buildMenu();
  createWindow();
});
app.on("window-all-closed", () => {
  backend.stop();
  app.quit();
});
