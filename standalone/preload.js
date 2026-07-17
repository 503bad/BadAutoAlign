// レンダラへ公開する最小API（contextIsolation境界）。
const { contextBridge, ipcRenderer, webUtils } = require("electron");

contextBridge.exposeInMainWorld("api", {
  version: () => ipcRenderer.invoke("vat:version"),
  process: (params) => ipcRenderer.invoke("vat:process", params),
  readFile: (path) => ipcRenderer.invoke("file:read", path),
  saveAs: (params) => ipcRenderer.invoke("file:saveAs", params),
  onProgress: (cb) => ipcRenderer.on("vat:progress", (_ev, p) => cb(p)),
  // Electron 32以降 File.path が廃止されたため、D&Dのパス取得はこれを使う
  pathForFile: (file) => webUtils.getPathForFile(file),
});
