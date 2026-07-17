"""GUI（スタンドアローン版）向けサービスモード。

標準入出力上の行区切りJSON-RPC風プロトコル:
  リクエスト : {"id": <any>, "method": "<name>", "params": {...}}\n
  レスポンス : {"id": <any>, "ok": true, "result": {...}}\n
               {"id": <any>, "ok": false, "error": "..."}\n

処理中のログ・進捗print()はすべてstderrへ流し、stdoutはプロトコル専用とする。

メソッド:
  version : → {"version": "..."}
  process : {"input": wav, "guide": wav|mid, "output": wav,
             "options": {Configフィールドの部分集合}}
            → レポートdict（フレーズ・alignment表・ノートごとのアンカー時刻を含む）
"""

from __future__ import annotations

import dataclasses
import json
import sys
import traceback
from contextlib import redirect_stdout


def serve() -> int:
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
            with redirect_stdout(sys.stderr):
                result = _dispatch(req.get("method"), req.get("params") or {})
            resp = {"id": req_id, "ok": True, "result": result}
        except Exception as e:  # noqa: BLE001 — サービスは落とさずエラーを返す
            traceback.print_exc(file=sys.stderr)
            resp = {"id": req_id, "ok": False, "error": f"{type(e).__name__}: {e}"}
        out.write(json.dumps(resp, ensure_ascii=False) + "\n")
        out.flush()
    return 0


def _dispatch(method: str | None, params: dict):
    if method == "version":
        from . import __version__

        return {"version": __version__}
    if method == "process":
        return _process(params)
    raise ValueError(f"未知のメソッド: {method}")


def _process(params: dict) -> dict:
    from .config import Config
    from .pipeline import process_file

    for key in ("input", "guide", "output"):
        if not params.get(key):
            raise ValueError(f"パラメータ {key} は必須です")

    valid = {f.name for f in dataclasses.fields(Config)}
    options = {k: v for k, v in (params.get("options") or {}).items() if k in valid}
    cfg = Config(**options)
    return process_file(params["input"], params["guide"], params["output"], cfg,
                        manual_anchors=params.get("manual_anchors"))
