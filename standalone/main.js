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

// 補正済み音源を保存ダイアログ経由でコピー保存する
ipcMain.handle("file:saveAs", async (ev, { sourcePath, vocalPath }) => {
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

const LICENSE_TEXT = `${app.getName()} v${app.getVersion()}
© 2026 Igarashi Date / 503 bad gateway
本アプリは MIT License で提供されます。

サードパーティライセンス:
・Signalsmith Stretch / signalsmith-linear — MIT License
  (c) Geraint Luff / Signalsmith Audio Ltd.
・WORLD (pyworld) — 修正BSD License
・librosa — ISC License
・pretty_midi — MIT License
・numpy / scipy / soundfile — BSD系 License
・Electron / Chromium — MIT / BSD License

コピーレフト・クレジット表示義務のある依存は使用していません。`;

function showLicense(win) {
  dialog.showMessageBox(win, {
    type: "info",
    title: "ライセンス表記",
    message: `${app.getName()} のライセンス`,
    detail: LICENSE_TEXT,
    buttons: ["閉じる"],
  });
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
              message: `${app.getName()} v${app.getVersion()}`,
              detail: "開発: Igarashi Date（503 bad gateway）\n" +
                "MIDI/WAVガイドによるボーカルのタイミング・ピッチ一括補正",
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
