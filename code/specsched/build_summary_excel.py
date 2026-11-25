"""Readable workbook of every experiment run for the idea "online draft-head training x speculative decoding
under one compute budget": Rung 0 (10-04), Rung 1 (10-05/06), Tencent demo sprint (10-08/09), OSC Cardinal H100.

Usage: python build_summary_excel.py <project root> [out.xlsx]
Reads data/runs (local mirror of the OSC runs), data/traffic, data/osc_jobs_sacct.psv (sacct export),
results/summaries/*.csv and code/specsched/sched/model*.json. Numbers that exist only in the conclusion documents
(the 24 h trace replay of the simulator, the aux-layer calibration, the failed worker-only switch run) are typed in
and marked with their source.

Derived columns (ratios, changes, totals) are Excel formulas over the measured cells; their values are also written
into the file (computed here and checked against an independent evaluation of each formula), so viewers without a
calculation engine still show numbers.
"""
import glob
import json
import math
import os
import re
import sys
import zipfile
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import column_index_from_string, get_column_letter
from openpyxl.worksheet.hyperlink import Hyperlink

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_c2  # noqa: E402

ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(HERE))
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/results/全部实验汇总_在线训草稿与投机解码.xlsx"
RUNS = f"{ROOT}/data/runs"
NY, SYD = ZoneInfo("America/New_York"), ZoneInfo("Australia/Sydney")

# ----------------------------------------------------------------------------------------------- styles
FONT = "Arial"
F_BODY = Font(name=FONT, size=10)
F_BOLD = Font(name=FONT, size=10, bold=True)
F_SMALL = Font(name=FONT, size=9, color="595959")
F_TITLE = Font(name=FONT, size=15, bold=True, color="1F3864")
F_LEAD = Font(name=FONT, size=10.5, bold=True, color="262626")
F_SECTION = Font(name=FONT, size=11, bold=True, color="FFFFFF")
F_HEAD = Font(name=FONT, size=10, bold=True, color="1F3864")
F_LINK = Font(name=FONT, size=10, color="0563C1", underline="single")


def fill(hex_):
    return PatternFill("solid", start_color=hex_, end_color=hex_)


FILL_SECTION = fill("1F4E79")
FILL_HEAD = fill("DCE6F1")
FILL_LABEL = fill("F2F2F2")
FILL_LEAD = fill("FFF8E1")
STATUS = {  # verdict word -> (fill, font colour)
    "成立": ("E2EFDA", "375623"), "通过": ("E2EFDA", "375623"), "有效": ("E2EFDA", "375623"),
    "部分成立": ("FFF2CC", "7F6000"), "半过": ("FFF2CC", "7F6000"), "边缘": ("FFF2CC", "7F6000"),
    "推翻原剧本": ("FFF2CC", "7F6000"), "有效（模拟）": ("DDEBF7", "1F4E79"),
    "不成立": ("FCE4D6", "9C0006"), "作废": ("FCE4D6", "9C0006"), "失败": ("FCE4D6", "9C0006"),
    "取消": ("EDEDED", "595959"), "对照": ("EDEDED", "404040"), "参考": ("EDEDED", "404040"),
    "更好": ("E2EFDA", "375623"), "更差": ("FCE4D6", "9C0006"), "分不开": ("EDEDED", "404040"),
}
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

INT, F1, F2, F3, F4 = "#,##0", "0.0", "0.00", "0.000", "0.0000"
PCT, PCTP = "+0.0%;-0.0%;0.0%", "0.0%"
X2 = '0.00"×"'

REG = {}          # (sheet, coord) -> value the formula must evaluate to
DOM_ZH = {"chat": "对话", "sql": "SQL", "code": "代码", "zh": "中文", "fc": "函数调用", "math": "数学（思考）"}
COVER = {"sql": 0.915, "code": 0.976, "zh": 0.426, "chat": 0.958, "fc": 0.913}   # prereg change log, 10-04


def disp_len(s):
    return sum(1.85 if ord(ch) > 0x2E7F else 0.95 for ch in str(s))


def n_lines(s, width):
    usable = max(width - 0.8, 1.0)
    return sum(max(1, math.ceil(disp_len(part) / usable)) for part in str(s).split("\n"))


def height_for(s, width, size=10.0):
    return round(max(15.0, n_lines(s, width) * size * 1.33 + 4), 1)


def finite(x):
    return x is not None and not (isinstance(x, float) and not math.isfinite(x))


# ----------------------------------------------------------------------------------------------- sheet writer
class Sheet:
    def __init__(self, wb, title, widths, back=True):
        self.ws = wb.create_sheet(title)
        self.title = title
        self.widths = widths
        self.ncol = len(widths)
        self.r = 1
        for i, w in enumerate(widths, 1):
            self.ws.column_dimensions[get_column_letter(i)].width = w
        self.ws.sheet_view.showGridLines = False
        self.ws.page_setup.orientation = "landscape"
        self.ws.page_setup.fitToWidth = 1
        self.ws.page_setup.fitToHeight = 0
        self.ws.sheet_properties.pageSetUpPr.fitToPage = True
        self.back = back

    def _merge_text(self, text, font, fill_=None, first=1, size=10.0, border=False):
        c = self.ws.cell(self.r, first, text)
        c.font = font
        c.alignment = Alignment(wrap_text=True, vertical="top", horizontal="left")
        for j in range(first, self.ncol + 1):
            if fill_ is not None:
                self.ws.cell(self.r, j).fill = fill_
            if border:
                self.ws.cell(self.r, j).border = BOX
        if self.ncol > first:
            self.ws.merge_cells(start_row=self.r, start_column=first, end_row=self.r, end_column=self.ncol)
        self.ws.row_dimensions[self.r].height = height_for(text, sum(self.widths[first - 1:]), size)
        self.r += 1
        return c

    def title_(self, text, lead=None):
        c = self.ws.cell(1, 1, text)
        c.font = F_TITLE
        c.alignment = Alignment(vertical="center")
        self.ws.row_dimensions[1].height = 26
        last = self.ncol - 1 if self.back else self.ncol
        if last > 1:
            self.ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=last)
        if self.back:
            b = self.ws.cell(1, self.ncol, "← 返回总览")
            b.font = F_LINK
            b.alignment = Alignment(horizontal="right", vertical="center")
            b.hyperlink = Hyperlink(ref=b.coordinate, location="'总览'!A1", display="← 返回总览")
        self.r = 2
        if lead:
            self._merge_text("结论：" + lead, F_LEAD, FILL_LEAD, size=10.5, border=True)
        self.gap()

    def gap(self, h=7):
        self.ws.row_dimensions[self.r].height = h
        self.r += 1

    def section(self, text):
        self.gap(5)
        self._merge_text(text, F_SECTION, FILL_SECTION, size=11)
        self.ws.row_dimensions[self.r - 1].height = 20

    def kv(self, pairs):
        for k, v in pairs:
            a = self.ws.cell(self.r, 1, k)
            a.font, a.fill, a.border = F_BOLD, FILL_LABEL, BOX
            a.alignment = Alignment(wrap_text=True, vertical="top")
            c = self.ws.cell(self.r, 2, v)
            c.font = F_BODY
            c.alignment = Alignment(wrap_text=True, vertical="top")
            for j in range(2, self.ncol + 1):
                self.ws.cell(self.r, j).border = BOX
            if self.ncol > 2:
                self.ws.merge_cells(start_row=self.r, start_column=2, end_row=self.r, end_column=self.ncol)
            self.ws.row_dimensions[self.r].height = max(height_for(v, sum(self.widths[1:])), height_for(k, self.widths[0]))
            self.r += 1

    def legend(self, items):
        """[(label, status word, meaning)]: coloured label cell + meaning merged across the rest."""
        for label, word, meaning in items:
            fl, col = STATUS[word]
            a = self.ws.cell(self.r, 1, label)
            a.font, a.fill, a.border = Font(name=FONT, size=10, bold=True, color=col), fill(fl), BOX
            a.alignment = Alignment(wrap_text=True, vertical="center")
            c = self.ws.cell(self.r, 2, meaning)
            c.font = F_BODY
            c.alignment = Alignment(wrap_text=True, vertical="center")
            for j in range(2, self.ncol + 1):
                self.ws.cell(self.r, j).border = BOX
            self.ws.merge_cells(start_row=self.r, start_column=2, end_row=self.r, end_column=self.ncol)
            self.ws.row_dimensions[self.r].height = max(height_for(meaning, sum(self.widths[1:])), height_for(label, self.widths[0]))
            self.r += 1

    def para(self, text, font=F_BODY):
        self._merge_text(text, font)

    def bullets(self, items):
        for it in items:
            self._merge_text("• " + it, F_BODY)

    def note(self, text):
        self._merge_text(text, F_SMALL, size=9)

    def source(self, text):
        self.gap(4)
        self._merge_text("数据来源：" + text, F_SMALL, size=9)

    def caption(self, text):
        self.gap(4)
        self._merge_text(text, F_BOLD)

    def table(self, cols, rows, caption=None, note=None):
        """cols: [(id, header, fmt[, span])] ; rows: [(key, {col_id: cellspec})]. span = sheet columns merged into one.
        cellspec: value | dict(v=, f=template, fmt=, bold=, status=, fill=, link=, align=).
        Formula templates: {<col_id>} = that column's letter, {r} = this row, {@<row key>|<col_id>} = that cell.
        Returns pos[(key, col_id)] -> coordinate."""
        if caption:
            self.caption(caption)
        spans = [c[3] if len(c) > 3 else 1 for c in cols]
        starts = [1 + sum(spans[:j]) for j in range(len(cols))]
        assert starts[-1] + spans[-1] - 1 <= self.ncol, (self.title, caption, "table wider than the sheet layout")
        letters = {c[0]: get_column_letter(st) for c, st in zip(cols, starts)}
        width = [sum(self.widths[st - 1:st - 1 + sp]) for st, sp in zip(starts, spans)]

        def frame(row, j):
            for k in range(starts[j], starts[j] + spans[j]):
                self.ws.cell(row, k).border = BOX
            if spans[j] > 1:
                self.ws.merge_cells(start_row=row, start_column=starts[j], end_row=row, end_column=starts[j] + spans[j] - 1)

        hr = self.r
        rowno = {key: hr + 1 + i for i, (key, _c) in enumerate(rows)}
        hmax = 15.0
        for j, col in enumerate(cols):
            head = col[1]
            c = self.ws.cell(hr, starts[j], head)
            for k in range(starts[j], starts[j] + spans[j]):
                self.ws.cell(hr, k).fill = FILL_HEAD
            frame(hr, j)
            c.font = F_HEAD
            c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
            hmax = max(hmax, height_for(head, width[j]))
        self.ws.row_dimensions[hr].height = hmax
        self.r += 1
        def shown(spec):
            return spec.get("v") if isinstance(spec, dict) and "f" not in spec else (None if isinstance(spec, dict) else spec)
        col_left = [j == 0 or any(isinstance(shown(cells.get(col[0])), str) and disp_len(shown(cells.get(col[0]))) > 14
                                  for _k, cells in rows) for j, col in enumerate(cols)]
        pos = {}
        for key, cells in rows:
            rr = self.r
            hrow = 15.0
            for j, col in enumerate(cols):
                cid, fmt = col[0], col[2]
                spec = cells.get(cid)
                c = self.ws.cell(rr, starts[j])
                frame(rr, j)
                c.font = F_BODY
                al = None
                fmt_ = fmt
                if isinstance(spec, dict):
                    if "f" in spec:
                        t = re.sub(r"\{@(.+?)\|(\w+)\}", lambda m: f"{letters[m.group(2)]}{rowno[m.group(1)]}", spec["f"])
                        c.value = t.format(r=rr, **letters)
                        REG[(self.title, c.coordinate)] = float(spec["v"])
                    else:
                        c.value = spec.get("v")
                    fmt_ = spec.get("fmt", fmt)
                    if spec.get("bold"):
                        c.font = F_BOLD
                    if spec.get("status") in STATUS:
                        fl, col = STATUS[spec["status"]]
                        c.fill = fill(fl)
                        c.font = Font(name=FONT, size=10, bold=True, color=col)
                    if spec.get("fill"):
                        c.fill = fill(spec["fill"])
                    if spec.get("link"):
                        c.hyperlink = Hyperlink(ref=c.coordinate, location=f"'{spec['link']}'!A1", display=str(c.value))
                        c.font = F_LINK
                    al = spec.get("align")
                else:
                    c.value = spec
                if isinstance(c.value, float) and not math.isfinite(c.value):
                    c.value = None
                if al is None:
                    al = "left" if col_left[j] else "center"
                if fmt_ and c.value is not None and (not isinstance(c.value, str) or c.value.startswith("=")):
                    c.number_format = fmt_
                c.alignment = Alignment(wrap_text=True, vertical="center", horizontal=al)
                if isinstance(c.value, str) and not c.value.startswith("="):
                    hrow = max(hrow, height_for(c.value, width[j]))
                pos[(key, cid)] = c.coordinate
            self.ws.row_dimensions[rr].height = hrow
            self.r += 1
        if note:
            self.note(note)
        return pos


def F(template, value, fmt=None, **kw):
    """Formula cell; value = what Python computes for it (stored as the cached value and checked)."""
    if not finite(value):
        return None
    d = dict(f=template, v=float(value), **kw)
    if fmt:
        d["fmt"] = fmt
    return d


def V(value, **kw):
    return dict(v=value, **kw)


# ----------------------------------------------------------------------------------------------- data helpers
def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def c1(group, cond):
    return {int(r["C"]): r for r in (jload(f"{RUNS}/c1/{group}/{cond}/summary.json") or [])}


def frontier(group, head):
    return {int(r["C"]): r for r in (jload(f"{RUNS}/frontier/{group}/{head}/summary.json") or [])}


def hotswap(group, mode):
    """B/A1/A2 bench results + acceptance first/last in B, B-phase trainer duty and swaps (as analyze_hotswap.py)."""
    d = f"{RUNS}/hotswap/{group}/{mode}"
    if not os.path.isdir(d):
        return None
    ev, tl = jl(f"{d}/events.jsonl"), jl(f"{d}/timeline.jsonl")
    tr, ctl = jl(f"{d}/trainer.jsonl"), jl(f"{d}/controller.jsonl")
    ph = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(f"{d}/phase*.json")}
    swaps = [dict(head=e["head"], t=e["t"], s=e.get("seconds", 0)) for e in ev if e.get("event") == "swap"]
    swaps += [dict(head=e["head"], t=e["at"], s=e.get("seconds", 0)) for e in ctl if e.get("event") == "swap" and "at" in e]
    swaps.sort(key=lambda s: s["t"])
    tB = next((e["t"] for e in ev if e.get("event") == "phase" and e.get("name") == "B"), None)
    tBe = next((e["t"] for e in ev if e.get("event") == "phase_end"), None)
    out = dict(A1=ph.get("phaseA1", {}), B=ph.get("phaseB", {}), A2=ph.get("phaseA2", {}), acc_first=None, acc_last=None,
               duty=0.0, swaps=[])
    if tB is not None and tl:
        t = np.array([r["t"] - tB for r in tl])
        acc = np.array([r["acc_len"] if r["acc_len"] else np.nan for r in tl], dtype=float)
        end = (tBe - tB) if tBe else t.max()
        inB = (t >= 0) & (t <= end)
        first, last = acc[inB][:3], acc[inB][-6:]
        out["acc_first"] = float(np.nanmean(first)) if np.isfinite(first).any() else None
        out["acc_last"] = float(np.nanmean(last)) if np.isfinite(last).any() else None
        du = [r["duty"] for r in tr if "duty" in r and 0 <= r["t"] - tB <= end]
        out["duty"] = float(np.mean(du)) if du else 0.0
        out["swaps"] = [(s["head"], s["t"] - tB, s["s"]) for s in swaps]
    out["tap"] = None
    if os.path.exists(f"{d}/server.log"):
        for line in open(f"{d}/server.log", errors="ignore"):
            if "[live-tap] steps=" in line:
                out["tap"] = {k: float(v) for k, v in (kv.split("=") for kv in line.split("[live-tap]")[1].split() if "=" in kv)}
    return out


def swaps_text(sw, cap=6):
    if not sw:
        return "—"
    parts = [f"{h} {t:+.0f}s（{s:.2f}s）" for h, t, s in sw]
    return "；".join(parts) if len(parts) <= cap else "；".join(parts[:cap - 1]) + f"；…（共 {len(parts)} 次，最后一次 {sw[-1][1]:+.0f}s）"


# ----------------------------------------------------------------------------------------------- jobs
JOBS = {  # job id -> (experiment, what it ran)
    "15248877": ("E01", "冒烟：环境调试，手动取消"), "15248885": ("E01", "冒烟：环境调试，手动取消"),
    "15248914": ("E01", "冒烟：环境调试，手动取消"), "15248915": ("E01", "冒烟全流程 + M1 aux 层校准"),
    "15249057": ("E01", "生成 M1 五个域的流量 + C1 训练进程用的缓存"),
    "15249094": ("E01", "M2：少量对话流量、aux 层校准、缓存、训练进程单独速度"),
    "15249058": ("E02", "C1 M1 短输出：S0"), "15249151": ("E02", "C1 M1 短输出：S1、S2、S3"),
    "15249092": ("E02", "重复提交，未运行即取消"), "15249093": ("E02", "重复提交，未运行即取消"),
    "15249152": ("E02", "C1 M1 长输出：S0、S1"), "15249153": ("E02", "C1 M1 长输出：S2、S3"),
    "15250579": ("E02", "C1 M1 长输出补跑：S3 的 C=64/96"),
    "15249154": ("E02", "C1 M2 MoE：S0、S1"), "15249155": ("E02", "C1 M2 MoE：S2、S3"),
    "15249453": ("E03", "占空比扫描 0.1 / 0.25 / 0.5"), "15249454": ("E03", "MPS：S1 基线 + S3 不限 SM"),
    "15249455": ("E03", "MPS：S3 训练进程限 30% SM"),
    "15249060": ("E04", "C2 SQL：在线更新 + 8 个头评测"), "15249061": ("E04", "C2 代码"),
    "15249062": ("E04", "C2 中文"), "15249063": ("E04", "C2 函数调用"),
    "15250120": ("E05", "跨实例噪声标定（代码 none/N400、SQL none/N3200，各 3 个实例）"),
    "15251522": ("E05", "固定长度（ignore_eos 256）饱和复测"),
    "15263307": ("E06", "frontier 第一轮（代码）：同节点端口碰撞，作废"),
    "15263308": ("E06", "frontier 第一轮（SQL+中文）：端口碰撞，取消"),
    "15263570": ("E06", "frontier 代码（重跑）"), "15263571": ("E06", "frontier SQL + 中文（重跑）"),
    "15263309": ("E07", "对话域在线更新 + 五个域评测"),
    "15263716": ("E10", "真机占空比控制：fixed0 / costmodel v1 / osd_idle / fixed1"),
    "15264390": ("E10", "真机占空比控制：门控后的 costmodel 重跑"),
    "15283295": ("E11", "代码域 800 条 hidden states 缓存"),
    "15283467": ("E11", "热替换第一版（训练进程不退出，作废）；其中 static 结果有效"),
    "15283468": ("E11", "热替换第一版（训练进程不退出，作废）"),
    "15284127": ("E11", "热替换闭环：train_swap / train_noswap"),
    "15283466": ("E18", "reload_draft_weights e2e（0.27.1 回移植）"),
    "15283773": ("E18", "reload_draft_weights e2e（主干，环境调试轮）"),
    "15283987": ("E18", "reload_draft_weights e2e（主干）"),
    "15342982": ("E15", "M2 三域流量 + 代码 800 条缓存"),
    "15342983": ("E12", "实时闭环第一版（300 条循环池）：static_live 用于导出开销；train_swap_live 记忆泄漏，作废"),
    "15342984": ("E15", "M2 C2 代码域在线更新 + 评测"),
    "15343033": ("E17", "TP=2 e2e + TP=2 服务热替换（gpu 分区 4 卡）"),
    "15343170": ("E15", "M2 frontier（代码）"), "15345395": ("E15", "M2 frontier 补 N3200"),
    "15343234": ("E15", "M2 实时闭环第一次提交，取消（改用唯一池重跑）"),
    "15343331": ("E13", "控制器代码域第一次提交，未运行即取消（改用唯一池）"),
    "15343332": ("E14", "中文场景第一次提交，未运行即取消（改用唯一池）"),
    "15343658": ("E13", "代码唯一池：static_live / train_swap_live / 控制器 v2"),
    "15343659": ("E14", "中文唯一池：static_live / train_swap_live / 控制器 v4"),
    "15343660": ("E15", "MoE 唯一池：static_live / train_swap_live"),
    "15344614": ("E13", "代码：控制器 v4"), "15345274": ("E13", "代码：静态表关投机"),
    "15345623": ("E13", "代码：控制器 v5"), "15345869": ("E13", "代码：控制器 v5.1（最终）"),
    "15344918": ("E14", "中文：静态表关投机"), "15345624": ("E14", "中文：控制器 v5"),
    "15345940": ("E14", "中文：控制器 v5.1"),
    "15345431": ("E15", "MoE：控制器 v4"), "15345625": ("E15", "MoE：控制器 v5，改出 v5.1 后撤下"),
    "15345870": ("E15", "MoE：控制器 v5.1"),
    "15347807": ("E16", "开关单测（worker 单侧版）：引擎断言崩溃，取消"),
    "15347808": ("E13", "代码：开关版控制器（worker 单侧版），取消"),
    "15347809": ("E14", "中文：开关版控制器（worker 单侧版），未运行即取消"),
    "15347872": ("E16", "开关单测（worker 单侧版）：关后 −46%，取消"),
    "15347873": ("E13", "代码：开关版控制器（worker 单侧开关；本轮控制器从未关投机）"),
    "15347874": ("E14", "中文：开关版控制器（worker 单侧版），取消"),
    "15348478": ("E16", "开关单测（调度器级）：代码 + 中文"),
    "15348479": ("E14", "中文：开关版控制器（调度器级开关）"),
    "15349223": ("E16", "开关单测（代码，hook 显式缓存 next tokens 后重跑）"),
    "15363115": ("E17", "TP=4 服务热替换（gpu 分区）"),
    "15391476": ("E17", "TP=2 逐 token 精确性探针（gpu 分区）"),
}
for _j in ["15343744", "15344905", "15344917", "15345527", "15345563", "15345655", "15345944", "15346282",
           "15346391", "15346453", "15347818", "15347875", "15348491", "15348503", "15349056", "15349224"]:
    JOBS[_j] = ("E18", "vLLM 主干分支 feat/spec-live-tap：实时导出 / 开关的单测与 e2e（调试轮）")
JOBS["15345163"] = ("E18", "主干单测挂起（queue.join 没有 task_done），取消")
JOBS["15349325"] = ("E18", "主干最终验证（commit ddab4c4）：单测 3 passed、导出 e2e 2 passed、开关 e2e 1 passed")
JOBS["15772661"] = ("E19", "消融矩阵冒烟（nB=600）：vanilla / static / simple / fixed800")
JOBS["15772662"] = ("E19", "消融矩阵冒烟（nB=600）：nospec / switch / fixed / v6")
JOBS["15864030"] = ("E22", "TIDE 式冒烟：共享环境被 scratch 清理删坏（缺 vllm.version），12 s 失败")
JOBS["15864325"] = ("E22", "TIDE 式冒烟（重建环境 ss312，nB=600）")
JOBS["15864323"] = ("E22", "环境重建核对：冻结矩阵的 3 次运行在 ss312 重跑")
for _j in ["15864708", "15864709", "15864711"]:
    JOBS[_j] = ("E22", "TIDE 式基线工人（3 场景 × 3 次，abl_runner_any.py）")
JOBS["15865711"] = ("E22", "完整控制器 8B 代码 3 次在重建环境重跑（同环境对照）")
JOBS["15866165"] = ("E22", "TIDE 式 MoE 第 2 次换节点重跑（原运行被干扰扫描标记）")
for _j in ["15866259", "15866260"]:
    JOBS[_j] = ("E22", "冻结矩阵被标记运行的换节点重跑（本想每个作业跑 2 次，sbatch --export 按逗号拆开 ONLY，只跑了第 1 次）")
for _j in ["15866596", "15866597"]:
    JOBS[_j] = ("E22", "冻结矩阵被标记运行的换节点重跑（补上被逗号拆掉的另外 2 次）")
for _j in ["15884860", "15884861", "15884862"]:
    JOBS[_j] = ("E22", "TIDE 式参数敏感性工人（每轮 800 条、漂移阈值 0.3，3 场景 × 3 次）")
SGL_JOBS = {"ss-sgl": "SGLang 冒烟（未打补丁）", "ss-sglchk": "SGLang 移植功能 / 速度检查", "ss-sglc1": "SGLang 校准：负载扫描与训练干扰",
            "ss-sglfr": "SGLang 校准：前沿", "ss-sglsm": "SGLang 矩阵冒烟（nB=600）", "ss-sglm": "SGLang 矩阵工人（abl_runner_any.py）"}
E24_JOBS = {"ss-c2mix": "混合域学习曲线（c2）", "ss-frmix": "混合域前沿", "ss-ablsm": "消融冒烟", "ss-ablc": "消融矩阵工人", "ss-mixsm": "混合流量冒烟",
            "ss-mixm": "混合流量矩阵工人"}
SGL_JOBS = {**SGL_JOBS, **E24_JOBS}
NEW_JOBS = {"ss-v7sm": ("E25", "v7.0 无标签自校准控制器冒烟"), "ss-v7m": ("E25", "v7.0 矩阵工人（3 单域 + 2 混合场景 × 3 次）"),
            "ss-bud": ("E25", "Switch+Budget 校准固定预算基线（5 场景 × 3 次）"), "ss-v71": ("E25", "v7.1 矩阵工人（配对测量 / 关投机探测 / 多轮前瞻）"),
            "ss-v71r": ("E25", "v7.1 被干扰扫描标记运行的换节点重跑"),
            "ss-seg": ("E26", "段长 × 预算工人（3 场景 × 3 段长 × 5 策略 × 3 次）"), "ss-sv70": ("E26", "段长测试里 v7.0 补跑（排队时代码已升到 v7.1，另加 v70 策略）"),
            "ss-wait": ("E27", "实验 E 第一轮：MoE 代码 + 8B 代码+对话（4 策略 × 4 段长 × 3 次）"), "ss-wait2": ("E27", "实验 E 第二轮：8B 代码"),
            "ss-wait3": ("E27", "实验 E 第三轮：Δ̂ 偏差变体 wait_d05 / wait_d2（MoE）"),
            "ss-waitr": ("E27", "实验 E 重跑：干扰扫描标记的 8B 代码 w4 仅开关 第 3 次（c0820）"),
            "ss-waitr2": ("E27", "实验 E 重跑：训练器慢（c0820）与训练器崩溃（cuDNN MHA）各 1 次"),
            "ss-waitr3": ("E27", "实验 E 重跑：w1 里行为相同的 switch/wait 对相差 >3% 的 3 次"),
            "ss-waitr4": ("E27", "实验 E 重跑：规则 (b) 重算中位数后再命中的 1 次（mixcc w3 train_now r2）"),
            "ss-arm-smoke": ("E28", "E28 冒烟（round300 / budget3k 闭环 + 两次开环）"),
            "ss-arm": ("E28", "E28a 矩阵工人（两条链各 16 个作业；链尾队列已空的作业 1–9 秒退出）"),
            "ss-diag": ("E28", "开环显存诊断（6 次，不进对比）")}
JOBS["17132895"] = ("E28", "E28a 预注册规则 (b) 命中的 1 次重跑（m1_zh round800 种子 1，c0802）")
STAGE = {e: s for s, es in [("Rung 0", ["E01", "E02", "E03", "E04", "E05"]),
                            ("Rung 1", ["E06", "E07", "E08", "E09", "E10", "E11"]),
                            ("demo 冲刺", ["E12", "E13", "E14", "E15", "E16", "E17", "E18"]),
                            ("论文补充", ["E19", "E20", "E21", "E22", "E23", "E24"]),
                            ("方法深化", ["E25", "E26", "E27"]),
                            ("同起点对照", ["E28", "E29"])] for e in es}


def load_jobs():
    rows = []
    for line in open(f"{ROOT}/data/osc_jobs_sacct.psv"):
        f = line.rstrip("\n").split("|")
        jid, name, _acct, part, _sub, start, _end, el, elraw, tres, state, _ec = f
        g = re.search(r"gres/gpu=(\d+)", tres)
        ngpu = int(g.group(1)) if g else 0
        st = datetime.fromisoformat(start).replace(tzinfo=NY).astimezone(SYD) if start.startswith("2026") else None
        exp, what = JOBS.get(jid, NEW_JOBS[name] if name in NEW_JOBS
                             else ("E19", "消融 × 重复矩阵工人（abl_runner.py 从 manifest 领取运行）") if name.startswith("ss-abl")
                             else (("E24" if name in E24_JOBS else "E23", SGL_JOBS[name]) if name in SGL_JOBS else ("?", "")))
        rows.append(dict(id=jid, name=name, part=part, start=st, el=el, secs=int(elraw), ngpu=ngpu,
                         gh=ngpu * int(elraw) / 3600,
                         state="完成" if state.startswith("COMPLETED") else ("取消" if "CANCEL" in state else state),
                         exp=exp, what=what))
    unknown = [r["id"] for r in rows if r["exp"] == "?"]
    assert not unknown, f"jobs without an experiment: {unknown}"
    return rows


def meta(ctx, eid, **kw):
    js = [j for j in ctx["jobs"] if j["exp"] == eid]
    ran = [j for j in js if j["secs"] > 0 and j["start"]]
    if ran:
        d0, d1 = min(j["start"] for j in ran).strftime("%m-%d"), max(j["start"] for j in ran).strftime("%m-%d")
        date = d0 if d0 == d1 else f"{d0} ~ {d1}"
    else:
        date = "10-05（本机）"
    kw.setdefault("date", date)
    kw.update(id=eid, stage=STAGE[eid], gh=sum(j["gh"] for j in js), njobs=len(js), sheet=f"{eid} {kw['short']}")
    return kw


# =============================================================================================== experiments
def e01(ctx):
    s1, s3 = c1("m1_short", "S1_smoke"), c1("m1_short", "S3_d1_smoke")
    t1, t3 = s1[1]["median_tpot_ms"], s3[1]["median_tpot_ms"]
    lead = ("全流程在小规模上跑通。EAGLE-3 头读取目标模型的层号定为 M1 (2,18,33)、M2 (2,24,45)。同卡训练进程的峰值显存 "
            "M1 约 10.5 GB、M2 约 5.4 GB，正式实验分别预留 12 GB、6 GB。冒烟里同卡训练让投机的 TPOT 翻倍 "
            f"（C=1：{t1:.1f} → {t3:.1f} ms），因此追加了占空比扫描和 MPS（E03）。")

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], [22, 14, 14, 14, 14, 14, 14, 14])
        s.title_(f"{e['id']} · {e['name']}", lead)
        s.section("实验配置")
        s.kv([("目的", "正式实验之前，把\"生成流量 → 校准 → 缓存 → 同卡训练 → C1 → 在线更新 → C2 评测\"在小规模上跑通，"
                      "并定下 EAGLE-3 头读取目标模型哪几层、训练进程要预留多少显存。"),
              ("作业与日期", "10-04（悉尼）。冒烟 15248915（之前 3 次是环境调试，手动取消）；M1 流量生成 15249057；M2 准备 15249094。"),
              ("冒烟步骤", "① 生成对话、SQL 各 64 条流量；② aux 层校准（32 条对话请求）；③ 缓存 48 条对话请求的 hidden states；"
                         "④ 训练进程单独跑 45 s；⑤ C1 只跑 S1、S3 两个条件、并发 1 和 16；⑥ SQL 在线更新 32/64 条；⑦ C2 评测 N0、N64。"),
              ("aux 层校准方法", "原装 AngelSlim 头在 teacher forcing 下预测下一个 token 的 top-1 命中率，比较两组层号。vLLM 0.27.1 "
                             "里注册号 L = 第 L−1 层的输出 = HF hidden_states[L]，读源码确认后再用命中率二次确认。"),
              ("流量生成", "目标模型对每个域的提示贪心生成（非思考模式、≤512 token、不开前缀缓存），每域 3200 train + 300 eval + 300 reserve；"
                       "输出 token id 落盘，既当\"线上流量\"，也当逐 token 一致性的参照（统计见「通用配置」）。")])
        s.section("结果")
        s.table([("m", "模型", None), ("aux", "候选层号（vLLM 注册号）", None), ("hit", "top-1 命中率", F3), ("use", "采用", None)], [
            ("m1a", {"m": "M1 Qwen3-8B", "aux": "(2, 18, 33)", "hit": 0.588, "use": V("采用", status="通过")}),
            ("m1b", {"m": "M1 Qwen3-8B", "aux": "(3, 19, 34)", "hit": 0.544, "use": "—"}),
            ("m2a", {"m": "M2 Qwen3-30B-A3B", "aux": "(2, 24, 45)", "hit": 0.617, "use": V("采用", status="通过")}),
            ("m2b", {"m": "M2 Qwen3-30B-A3B", "aux": "(3, 25, 46)", "hit": 0.591, "use": "—"}),
        ], caption="表 1　aux 层校准（原装头，teacher forcing，32 条对话请求）",
            note="命中率数字转录自预注册文档的变更记录（10-04 冒烟之后写入）；冒烟日志在 OSC，未镜像到本地。")
        rows = []
        for C in (1, 16):
            a, b = s1[C], s3[C]
            rows.append((f"C{C}", {"c": f"C={C}", "s1": a["output_throughput"], "s3": b["output_throughput"],
                                   "ratio": F("={s3}{r}/{s1}{r}", b["output_throughput"] / a["output_throughput"], X2),
                                   "t1": a["median_tpot_ms"], "t3": b["median_tpot_ms"],
                                   "tr": F("={t3}{r}/{t1}{r}", b["median_tpot_ms"] / a["median_tpot_ms"], X2),
                                   "trn": b.get("trainer_tok_s", 0) / 1000}))
        s.table([("c", "并发", None), ("s1", "S1 投机\n吞吐 tok/s", INT), ("s3", "S3 投机+同卡训练\n吞吐 tok/s", INT),
                 ("ratio", "吞吐保留\nS3÷S1", X2), ("t1", "S1 TPOT\np50 ms", F2), ("t3", "S3 TPOT\np50 ms", F2),
                 ("tr", "TPOT 倍数\nS3÷S1", X2), ("trn", "训练速率\nk tok/s", F1)], rows,
                caption="表 2　冒烟里的 C1（M1 对话，投机开，训练占空比 1.0）",
                note=f"接受长度两种条件都约 {s1[1]['acc_len']:.2f}（训练进程不影响草稿质量，只抢 GPU 时间）。")
        s.table([("m", "模型", None), ("peak", "训练进程峰值显存 GB", F1), ("res", "正式实验预留 GB", INT),
                 ("util", "S2/S3 的 gpu_memory_utilization", F3)], [
            ("m1", {"m": "M1 Qwen3-8B", "peak": c1("m1_short", "S3_d1")[1]["trainer_peak_mem_gb"], "res": 12,
                    "util": c1("m1_short", "S3_d1")[1]["util"]}),
            ("m2", {"m": "M2 Qwen3-30B-A3B", "peak": c1("m2_short", "S3_d1")[1]["trainer_peak_mem_gb"], "res": 6,
                    "util": c1("m2_short", "S3_d1")[1]["util"]}),
        ], caption="表 3　同卡训练进程的显存（load 模式：只含头的前向反向）",
            note="util = 0.90 − 预留 ÷ 显存总量（93.6 GB）；KV 缓存因此变小，这部分损失算进共置代价。")
        s.section("怎么读")
        s.bullets(["(2,18,33) 和 (2,24,45) 命中率更高，之后所有实验都用这组层号。",
                   "冒烟里 S3 的 TPOT 约是 S1 的 2 倍，说明不限速的同卡训练代价很大 → 预注册变更里追加了两组探索性条件：占空比扫描、MPS（E03）。",
                   "冒烟也定下了 C2 饱和吞吐要\"测三次取中位数\"、长输出每档请求数改为约 3×并发（见预注册变更记录）。"])
        s.source("data/runs/c1/m1_short/{S1_smoke,S3_d1_smoke}/summary.json；data/runs/c1/*/S3_d1/summary.json；"
                 "ideas/2026-10-04_Rung0_characterization_预注册.md（变更记录）")
    ctx["e01"] = dict(t1=t1, t3=t3)
    return meta(ctx, "E01", short="冒烟与校准", name="准备：冒烟、流量生成、aux 层校准",
                q="正式实验前全流程能不能跑通？EAGLE-3 头该读目标模型哪几层？训练进程要留多少显存？",
                cfg="M1/M2 小规模：生成流量 → 校准 → 缓存 → 同卡训练 → C1 两档 → 在线更新 → C2 评测",
                res=f"层号 M1 (2,18,33) 命中 0.588 > 0.544；M2 (2,24,45) 0.617 > 0.591；训练进程显存 10.5 / 5.4 GB；"
                    f"同卡训练让 C=1 的 TPOT {t1:.1f} → {t3:.1f} ms",
                verdict="通过；据此追加 E03 两组探索性条件", word="通过", status="有效", build=build)


def e02(ctx):
    regimes = [("m1_short", "M1 Qwen3-8B 短输出（对话）"), ("m1_long", "M1 Qwen3-8B 长输出（思考数学）"),
               ("m2_short", "M2 Qwen3-30B-A3B MoE 短输出（对话）")]
    data = {}
    for g, _ in regimes:
        d = dict(S0=c1(g, "S0"), S1=c1(g, "S1"), S2=c1(g, "S2_d1"), S3=c1(g, "S3_d1"))
        if g == "m1_long":
            hi = c1(g, "S3_d1_hi")
            if 96 in hi:
                d["S3"][96] = hi[96]
        data[g] = d
    gain = {g: {C: d["S1"][C]["output_throughput"] / d["S0"][C]["output_throughput"] - 1 for C in d["S0"] if C in d["S1"]}
            for g, d in data.items()}
    cost = [1 - d["S2"][C]["output_throughput"] / d["S0"][C]["output_throughput"] for d in data.values() for C in d["S0"] if C in d["S2"]]
    p99 = [d["S2"][C]["p99_tpot_ms"] / d["S0"][C]["p99_tpot_ms"] - 1 for d in data.values() for C in d["S0"] if C in d["S2"]]
    inter = {g: {C: (d["S3"][C]["output_throughput"] / d["S2"][C]["output_throughput"]) /
                    (d["S1"][C]["output_throughput"] / d["S0"][C]["output_throughput"]) - 1
                 for C in d["S0"] if C in d["S1"] and C in d["S2"] and C in d["S3"]} for g, d in data.items()}
    hi_int = [v for g in inter for C, v in inter[g].items() if C >= 64]
    lo_int = [v for g in inter for C, v in inter[g].items() if C <= 32]
    gs, gl, gm = gain["m1_short"], gain["m1_long"], gain["m2_short"]
    ctx["e02"] = dict(gain=gain, cost=(min(cost), max(cost)), p99=(min(p99), max(p99)), hi_int=(min(hi_int), max(hi_int)),
                      lo_int=(min(lo_int), max(lo_int)), data=data)
    lead = (f"投机收益随负载的走向分三种：稠密短输出从 C=1 的 {gs[1]:+.0%} 一路降到 C=256 的 {gs[256]:+.0%}；长输出到 C=96 仍有 "
            f"{gl[96]:+.0%}，没有拐点；MoE 反过来随负载上升（C=1 {gm[1]:+.0%}，C=64 {gm[64]:+.0%}，C=256 仍 {gm[256]:+.0%}）。"
            f"同卡训练（占空比 100%）在任何负载下都吃掉 {min(cost):.0%}–{max(cost):.0%} 的吞吐，P99 TPOT 涨 {min(p99):.0%}–{max(p99):.0%}。"
            f"两者只在 C≥64 时互相争：共置让投机的相对收益再少 {-max(hi_int):.0%}–{-min(hi_int):.0%}；C≤32 时共置不削弱投机的相对收益"
            f"（交互 {min(lo_int):+.0%}~{max(lo_int):+.0%}）。")

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], [12, 11, 11, 11, 11, 10, 11, 11, 10, 10, 11, 11, 11, 11])
        s.title_(f"{e['id']} · {e['name']}", lead)
        s.section("实验配置")
        s.kv([("要回答", "同一张卡上，投机的即时收益和同卡训练的代价随负载怎么变？有没有一个负载带，训练代价已经不可忽略而投机仍有收益？"
                       "两者是否争同一份余量？（预注册 H1–H3）"),
              ("作业与日期", "10-04（悉尼）。M1 短输出 15249058（S0）+ 15249151（S1–S3）；M1 长输出 15249152（S0、S1）+ 15249153（S2、S3）"
                          "+ 15250579（补 S3 的 C=64/96）；M2 15249154（S0、S1）+ 15249155（S2、S3）。"),
              ("三种情形", "① M1 短输出：Qwen3-8B，对话 eval 300 条，非思考模式，输出 ≤512，max_model_len 4096，C ∈ {1,2,4,8,16,32,64,128,256}，"
                        "每档请求数 = min(max(32, 4C), 1024)。② M1 长输出：Qwen3-8B，MATH-500 思考模式，输出 ≤4096，max_model_len 6144，"
                        "C ∈ {1,4,16,64,96}，每档请求数 C=1 为 6、C=4 为 12、其余 3C。③ M2 短输出：Qwen3-30B-A3B（MoE），其余同 ①。"),
              ("四个条件", "S0 不投机；S1 投机（EAGLE-3 原装头，k=3）；S2 不投机 + 同卡训练进程；S3 投机 + 同卡训练进程。"
                        "训练进程占空比 100%（一直训），两个进程直接分时共享 GPU，不开 MPS；它在 48 条对话请求预先算好的 hidden "
                        "states 上反复训头（只含头的前向反向，与真实系统里\"hidden states 由服务顺手给出\"的开销一致）。"),
              ("显存", "S0/S1 的 gpu_memory_utilization = 0.90；S2/S3 = 0.90 − 训练进程预留（M1 12 GB、M2 6 GB）÷ 显存总量，"
                     "KV 容量的损失算作共置成本。"),
              ("负载与测量", "vllm serve + vllm bench serve 闭环并发（--max-concurrency C，--request-rate inf），贪心；每个条件先热身 8 条（C=4）；"
                         "接受长度 = 1 + 接受的草稿 token 数 ÷ 草稿次数（/metrics 计数器差分）；训练速率按每档测量窗口统计。")])
        s.section("结果")
        cols = [("c", "并发 C", None), ("s0", "S0 不投机\ntok/s", INT), ("s1", "S1 投机\ntok/s", INT),
                ("s2", "S2 不投机\n+训练 tok/s", INT), ("s3", "S3 投机\n+训练 tok/s", INT), ("acc", "接受长度\n(S1)", F2),
                ("g", "投机收益\nS1÷S0−1", PCT), ("cost", "训练代价\n1−S2÷S0", PCTP), ("p0", "P99 TPOT\nS0 ms", F1),
                ("p2", "P99 TPOT\nS2 ms", F1), ("pu", "P99 涨幅\nS2÷S0−1", PCT), ("sx", "共置时投机\n倍数 S3÷S2", X2),
                ("ix", "交互\n共置倍数÷单独倍数−1", PCT), ("trn", "训练速率\nS2 k tok/s", F1)]
        for g, label in regimes:
            d = data[g]
            rows = []
            for C in sorted(d["S0"]):
                a0, a1, a2, a3 = (d[k].get(C) for k in ("S0", "S1", "S2", "S3"))
                row = {"c": f"C={C}", "s0": a0["output_throughput"]}
                if a1:
                    row.update(s1=a1["output_throughput"], acc=a1.get("acc_len"),
                               g=F("={s1}{r}/{s0}{r}-1", a1["output_throughput"] / a0["output_throughput"] - 1))
                if a2:
                    row.update(s2=a2["output_throughput"], p0=a0["p99_tpot_ms"], p2=a2["p99_tpot_ms"],
                               cost=F("=1-{s2}{r}/{s0}{r}", 1 - a2["output_throughput"] / a0["output_throughput"]),
                               pu=F("={p2}{r}/{p0}{r}-1", a2["p99_tpot_ms"] / a0["p99_tpot_ms"] - 1),
                               trn=(a2.get("trainer_tok_s") or 0) / 1000)
                if a3 and a2:
                    row.update(s3=a3["output_throughput"],
                               sx=F("={s3}{r}/{s2}{r}", a3["output_throughput"] / a2["output_throughput"]))
                    if a1:
                        row["ix"] = F("=({s3}{r}/{s2}{r})/({s1}{r}/{s0}{r})-1", inter[g][C])
                rows.append((f"{g}{C}", row))
            s.table(cols, rows, caption=f"{label}")
        hi64, main64 = c1("m1_long", "S3_d1_hi")[64]["output_throughput"], data["m1_long"]["S3"][64]["output_throughput"]
        s.note(f"M1 长输出 C=96 的 S3 来自补跑作业 15250579（S3_d1_hi）；同一作业也重测了 C=64：{hi64:,.0f} tok/s，与主表 {main64:,.0f} 相差 "
               f"{hi64 / main64 - 1:+.1%}。")
        s.table([("h", "假设", None), ("txt", "内容（预注册）", None, 2), ("rule", "判为成立的条件", None, 4), ("res", "实测", None, 5),
                 ("v", "判定", None, 2)], [
            ("H1", {"h": "H1", "txt": "低负载共置不是免费的", "rule": "M1 短输出、L≤30% 饱和时，S2 相对 S0 的 P99 TPOT 涨 ≥20%",
                    "res": f"P99 TPOT 涨 {min(p99):.0%}–{max(p99):.0%}，且与负载无关", "v": V("成立（远超）", status="成立")}),
            ("H2", {"h": "H2", "txt": "投机收益有负载拐点且依情形而异", "rule": "M1 短输出在饱和前转为 ≤0；长输出或 MoE 到 95% 饱和仍 >0",
                    "res": f"短输出 128→256 之间转负（{gs[256]:+.0%}）；长输出、MoE 到最高负载仍 {gl[96]:+.0%}、{gm[256]:+.0%}",
                    "v": V("成立", status="成立")}),
            ("H3", {"h": "H3", "txt": "投机和更新争同一份余量", "rule": "至少一个负载档上 |交互| ≥ 训练代价的 20%（加性口径）",
                    "res": f"加性口径\"成立\"但受尺度混淆；乘法口径：C≤32 共置不削弱投机的相对收益（交互 {min(lo_int):+.0%}~{max(lo_int):+.0%}），"
                           f"C≥64 被吃掉 {-max(hi_int):.0%}–{-min(hi_int):.0%}", "v": V("部分成立", status="部分成立")}),
        ], caption="预注册假设判定（H4 见 E04）")
        s.section("怎么读")
        s.bullets(["稠密短输出：饱和时算力成为瓶颈，验证多出来的 token 不再免费，投机在 128→256 之间转负。",
                   "长输出：KV 容量先把并发卡在 96 左右，系统到不了算力瓶颈，所以投机一直有 +70% 左右的收益。",
                   "MoE：小 batch 时多验证几个 token 会激活更多专家；batch 越大专家越接近全部加载，验证越接近免费，所以收益随负载上升"
                   "（与 MoESD 的预测一致）。",
                   "接受长度在各负载下几乎不变（M1 对话 1.88–1.91、数学 2.08–2.13、M2 对话 2.02–2.05），收益的变化全来自系统侧。",
                   "训练代价是\"两个进程直接分时共享、没做任何机制优化\"时的上界；E03 说明它能用占空比线性地调小。",
                   "预注册的 goodput（TPOT ≤ 并发 1 时 S0 中位数的 2 倍）对 MoE 过苛，表里只用吞吐和 P99 TPOT。"])
        s.source("data/runs/c1/{m1_short,m1_long,m2_short}/{S0,S1,S2_d1,S3_d1,S3_d1_hi}/summary.json（每档另有 C*.json 原始 bench 结果）；"
                 "图 results/figures/c1_*.png；结论文档 results/summaries/2026-10-04_Rung0_结论.md §1、§3")
    return meta(ctx, "E02", short="负载×条件", name="C1 负载 × 条件：投机收益与同卡训练代价",
                q="同一张卡上，投机的收益和同卡训练的代价随负载怎么变？两者争不争同一份余量？（H1–H3）",
                cfg="3 种情形（8B 短输出 / 8B 长输出 / MoE 短输出）× 4 个条件（S0–S3）× 并发 1–256；训练占空比 100%，分时共享",
                res=f"投机收益：8B 短 {gs[1]:+.0%} → {gs[256]:+.0%}（C=1→256）；8B 长 C=96 仍 {gl[96]:+.0%}；MoE {gm[1]:+.0%} → "
                    f"{gm[64]:+.0%}（C=64）。同卡训练吃掉 {min(cost):.0%}–{max(cost):.0%} 吞吐、与负载无关；只在 C≥64 争余量",
                verdict="H1 成立（远超）、H2 成立、H3 部分成立（只在中高负载争）", word="成立", status="有效", build=build)


def e03(ctx):
    Cs = [1, 4, 16, 64]
    base = c1("m1_short", "S1")
    duty = {d: c1("m1_short", f"S3_d{d:g}_sweep") for d in (0.1, 0.25, 0.5)}
    duty[1.0] = c1("m1_short", "S3_d1")
    mps_base, mps_full, mps30 = c1("m1_short", "S1_mps"), c1("m1_short", "S3_d1_mps"), c1("m1_short", "S3_d1_mps30")
    kept = {d: [duty[d][C]["output_throughput"] / base[C]["output_throughput"] for C in Cs] for d in duty}
    mfull = [mps_full[C]["output_throughput"] / mps_base[C]["output_throughput"] for C in Cs]
    m30 = [mps30[C]["output_throughput"] / mps_base[C]["output_throughput"] for C in Cs]
    model = jload(f"{ROOT}/code/specsched/sched/model.json")
    kappa = model["kappa"]
    loss = {d: (1 - max(v), 1 - min(v)) for d, v in kept.items()}
    ctx["e03"] = dict(kept=kept, mfull=mfull, m30=m30, loss=loss)
    lead = (f"服务吞吐的损失随占空比近似线性增长，各负载相近：占空比 0.1 只损失 {loss[0.1][0]:.0%}–{loss[0.1][1]:.0%}，0.5 损失 "
            f"{loss[0.5][0]:.0%}–{loss[0.5][1]:.0%}，1.0 损失 {loss[1.0][0]:.0%}–{loss[1.0][1]:.0%}（低占空比时比线性还便宜）——代价模型就是一个旋钮。"
            f"MPS 空间共享不限额时更糟（服务只剩 {min(mfull):.0%}–{max(mfull):.0%}）；训练进程限 30% SM 后（保留 {min(m30):.0%}–{max(m30):.0%}）"
            "回到与\"分时 50%\"同一条权衡线。共享机制不是关键，限额才是。")

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], [24, 12, 12, 12, 12, 30])
        s.title_(f"{e['id']} · {e['name']}", lead)
        s.section("实验配置")
        s.kv([("要回答", "把同卡训练限速（占空比），或换成 MPS 空间共享，能不能降低训练对服务的代价？代价和占空比是什么关系？"),
              ("性质", "探索性条件（冒烟之后、正式主表之前写进预注册变更记录），不参与 H1–H4 判定。"),
              ("作业与日期", "10-04（悉尼）。占空比扫描 15249453；MPS 不限 SM + MPS 下的 S1 基线 15249454；MPS 限 30% SM 15249455。"),
              ("情形", "M1 短输出（对话，同 E02），投机开（S3），C ∈ {1, 4, 16, 64}。占空比 1.0 一行取自 E02 的 S3。"),
              ("占空比怎么实现", "训练进程每训完一步就 sleep dt·(1−d)/d（dt 为这一步的耗时），即按时间片让出 GPU；d ∈ {0.1, 0.25, 0.5, 1.0}。"),
              ("MPS", "两个进程都作为同一个 MPS 服务的客户端（空间共享）。\"限 30%\" = 训练进程设 CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=30。"
                      "MPS 各行的基线是 MPS 下的 S1（MPS 本身让 S1 慢 1–3%）。")])
        s.section("结果")
        cols = [("k", "条件", None)] + [(f"c{C}", f"C={C}", INT) for C in Cs]
        raw = [("S1", {"k": V("S1 投机（基线）", bold=True), **{f"c{C}": base[C]["output_throughput"] for C in Cs}})]
        raw += [(f"d{d:g}", {"k": f"S3 占空比 {d:g}", **{f"c{C}": duty[d][C]["output_throughput"] for C in Cs}}) for d in duty]
        pa = s.table(cols, raw, caption="表 1　吞吐 tok/s（M1 短输出，投机开）")
        kept_rows = []
        for d in duty:
            key = f"d{d:g}"
            row = {"k": f"占空比 {d:g}"}
            for C in Cs:
                row[f"c{C}"] = F(f"={pa[(key, f'c{C}')]}/{pa[('S1', f'c{C}')]}",
                                 duty[d][C]["output_throughput"] / base[C]["output_throughput"], PCTP)
            row["lin"] = F(f"=(1-AVERAGE({{c1}}{{r}}:{{c64}}{{r}}))/{d:g}", (1 - np.mean(kept[d])) / d, F2)
            kept_rows.append((key, row))
        s.table([("k", "占空比", None)] + [(f"c{C}", f"C={C}", PCTP) for C in Cs] + [("lin", "损失÷占空比\n（四档平均）", F2)],
                kept_rows, caption="表 2　吞吐保留率 = S3 ÷ S1（公式）",
                note="最后一列 = 损失 ÷ 占空比：占空比 0.5–1 时约 0.5（损失 ≈ 占空比 × 50%）；占空比低时更小，低占空比比线性还便宜。"
                     "模拟器里用的 κ(C) 见表 6。")
        rawp = [("S1", {"k": V("S1 投机（基线）", bold=True), **{f"c{C}": base[C]["p99_tpot_ms"] for C in Cs}})]
        rawp += [(f"d{d:g}", {"k": f"S3 占空比 {d:g}", **{f"c{C}": duty[d][C]["p99_tpot_ms"] for C in Cs}}) for d in duty]
        pp = s.table([("k", "条件", None)] + [(f"c{C}", f"C={C}", F1) for C in Cs], rawp, caption="表 3　P99 TPOT（ms）")
        s.table([("k", "占空比", None)] + [(f"c{C}", f"C={C}", PCT) for C in Cs],
                [(f"d{d:g}", {"k": f"占空比 {d:g}", **{f"c{C}": F(f"={pp[(f'd{d:g}', f'c{C}')]}/{pp[('S1', f'c{C}')]}-1",
                                                               duty[d][C]["p99_tpot_ms"] / base[C]["p99_tpot_ms"] - 1, PCT)
                                                  for C in Cs}}) for d in duty],
                caption="表 4　P99 TPOT 涨幅 = S3 ÷ S1 − 1（公式）")
        s.table([("k", "占空比", None)] + [(f"c{C}", f"C={C}", F1) for C in Cs],
                [(f"d{d:g}", {"k": f"占空比 {d:g}", **{f"c{C}": (duty[d][C].get("trainer_tok_s") or 0) / 1000 for C in Cs}})
                 for d in duty], caption="表 5　训练速率（k token/s）")
        s.table([("k", "C", None), ("kap", "κ(C)", F2)],
                [(f"k{c}", {"k": f"C={c}", "kap": kappa[c]}) for c in sorted(kappa, key=int)],
                caption="表 6　模拟器用的代价系数 κ(C)：服务吞吐损失 = κ(C) × 占空比（由 E02/E03 拟合，sched/model.json）")
        mrows = [("S1", {"k": V("S1 分时（无 MPS）", bold=True), **{f"c{C}": base[C]["output_throughput"] for C in Cs}}),
                 ("S1m", {"k": V("S1 在 MPS 下", bold=True), **{f"c{C}": mps_base[C]["output_throughput"] for C in Cs}}),
                 ("t1", {"k": "S3 分时，占空比 1.0", **{f"c{C}": duty[1.0][C]["output_throughput"] for C in Cs}}),
                 ("t05", {"k": "S3 分时，占空比 0.5", **{f"c{C}": duty[0.5][C]["output_throughput"] for C in Cs}}),
                 ("mf", {"k": "S3 MPS，训练不限 SM", **{f"c{C}": mps_full[C]["output_throughput"] for C in Cs}}),
                 ("m30", {"k": "S3 MPS，训练限 30% SM", **{f"c{C}": mps30[C]["output_throughput"] for C in Cs}})]
        pm = s.table(cols, mrows, caption="表 7　共享机制对比：吞吐 tok/s")
        ratio_rows = []
        for key, label, b in [("t1", "分时，占空比 1.0（÷ S1 分时）", "S1"), ("t05", "分时，占空比 0.5（÷ S1 分时）", "S1"),
                              ("mf", "MPS 不限 SM（÷ S1 在 MPS 下）", "S1m"), ("m30", "MPS 限 30% SM（÷ S1 在 MPS 下）", "S1m")]:
            src = {"t1": duty[1.0], "t05": duty[0.5], "mf": mps_full, "m30": mps30}[key]
            den = base if b == "S1" else mps_base
            ratio_rows.append((key, {"k": label, **{f"c{C}": F(f"={pm[(key, f'c{C}')]}/{pm[(b, f'c{C}')]}",
                                                                src[C]["output_throughput"] / den[C]["output_throughput"], PCTP)
                                                    for C in Cs}}))
        s.table([("k", "条件（分母）", None)] + [(f"c{C}", f"C={C}", PCTP) for C in Cs], ratio_rows,
                caption="表 8　服务吞吐保留率（公式）")
        s.table([("k", "条件", None)] + [(f"c{C}", f"C={C}", F1) for C in Cs], [
            ("t1", {"k": "分时，占空比 1.0", **{f"c{C}": (duty[1.0][C].get("trainer_tok_s") or 0) / 1000 for C in Cs}}),
            ("t05", {"k": "分时，占空比 0.5", **{f"c{C}": (duty[0.5][C].get("trainer_tok_s") or 0) / 1000 for C in Cs}}),
            ("mf", {"k": "MPS 不限 SM", **{f"c{C}": (mps_full[C].get("trainer_tok_s") or 0) / 1000 for C in Cs}}),
            ("m30", {"k": "MPS 限 30% SM", **{f"c{C}": (mps30[C].get("trainer_tok_s") or 0) / 1000 for C in Cs}})],
            caption="表 9　训练速率（k token/s）")
        s.section("怎么读")
        s.bullets(["占空比是一个干净的旋钮：损失随占空比近似线性（0.5–1 时 ≈ 占空比 × 50%，低占空比更便宜），在 C=1–64 上差别不大。"
                   "调度器因此只需要决定\"给多少占空比\"。",
                   "按 SQL 域的数据量（3200 条请求约 31 万 token），占空比 10% 时约 1.5 分钟训完一轮完整适应，服务几乎无感。",
                   "MPS 不限额时，训练进程把显存带宽吃满，服务被饿到只剩约 1/5——解码和头训练都卡在带宽上，\"更细粒度的共享\"本身不产生余量。",
                   "MPS 限 30% SM 后与\"分时 50%\"落在同一条权衡线（低负载略好、高负载略差）。"])
        s.source("data/runs/c1/m1_short/{S1,S3_d0.1_sweep,S3_d0.25_sweep,S3_d0.5_sweep,S3_d1,S1_mps,S3_d1_mps,S3_d1_mps30}/summary.json；"
                 "code/specsched/sched/model.json（kappa）；结论文档 2026-10-04_Rung0_结论.md §1.4")
    return meta(ctx, "E03", short="占空比与MPS", name="C1 探索：训练占空比扫描与 MPS 空间共享",
                q="训练限速（占空比）和共享机制（MPS）能不能降低同卡训练的代价？",
                cfg="8B 短输出，投机开；占空比 0.1 / 0.25 / 0.5 / 1.0 × 并发 1 / 4 / 16 / 64；MPS 不限 SM / 限 30% SM",
                res=f"损失随占空比近似线性：0.1 → {loss[0.1][0]:.0%}–{loss[0.1][1]:.0%}，0.5 → {loss[0.5][0]:.0%}–{loss[0.5][1]:.0%}，"
                    f"1.0 → {loss[1.0][0]:.0%}–{loss[1.0][1]:.0%}；MPS 不限额服务只剩 {min(mfull):.0%}–{max(mfull):.0%}；限 30% SM ≈ 分时 50%",
                verdict="探索性：代价可以用一个旋钮线性控制；共享机制不是关键，限额才是", word="有效", status="有效", build=build)


def e04(ctx):
    doms = ["sql", "code", "zh", "fc"]
    dfs, nones = {}, {}
    for d in doms:
        df, none = analyze_c2.load_domain(f"{RUNS}/c2/m1/{d}")
        dfs[d], nones[d] = df.sort_values("N").reset_index(drop=True), none
    Ns = [0, 100, 200, 400, 800, 1600, 3200]

    def at(d, N, col):
        r = dfs[d][dfs[d].N == N]
        return float(r[col].iloc[0]) if len(r) and finite(float(r[col].iloc[0])) else None

    acc0 = {d: at(d, 0, "acc_len") for d in doms}
    acc3 = {d: at(d, 3200, "acc_len") for d in doms}
    tbe = {d: [at(d, N, "Tbe_min_accmodel") for N in Ns[1:]] for d in doms}
    tmax = max(max(x for x in v if x is not None) for v in tbe.values())
    fc_peak = max((at("fc", N, "acc_len"), N) for N in Ns)
    lat = {d: at(d, 3200, "b1_ms_tok") / at(d, 0, "b1_ms_tok") - 1 for d in doms}
    c100 = [at(d, 100, "C_train_gpu_s") for d in doms]
    t200 = [at(d, 200, "Tbe_min_accmodel") for d in doms]
    r200 = max(t200) / min(t200)
    chat_d = {d: at(d, 3200, "acc_len_chat") - at(d, 0, "acc_len_chat") for d in doms}
    sat0 = {d: at(d, 0, "spec_speedup_sat") for d in doms}
    sat3 = {d: at(d, 3200, "spec_speedup_sat") for d in doms}
    ctx["e04"] = dict(acc0=acc0, acc3=acc3, tmax=tmax, dfs=dfs, nones=nones, lat=lat, c100=c100, r200=r200, chat_d=chat_d)
    pos_chat = [chat_d[d] for d in ("sql", "code", "zh")]
    lead = (f"在线更新又便宜又有效：100 条请求只要 {min(c100):.1f}–{max(c100):.1f} GPU 秒，接受长度就明显上升；3200 条后代码 "
            f"{acc0['code']:.2f}→{acc3['code']:.2f}、SQL {acc0['sql']:.2f}→{acc3['sql']:.2f}、中文 {acc0['zh']:.2f}→{acc3['zh']:.2f}"
            f"（被草稿词表卡住）；函数调用先升后落（峰值 {fc_peak[0]:.2f}@N{fc_peak[1]}，N3200 为 {acc3['fc']:.2f}）。在 SQL、代码、中文上学，"
            f"对话域接受长度都上升（+{min(pos_chat):.2f}~+{max(pos_chat):.2f}）；在函数调用上学到 N3200 时对话略降（{chat_d['fc']:+.2f}）。"
            f"\"投机一直开\"口径下回本全在 {tmax:.1f} 分钟内，N200 时域间最大/最小只差 {r200:.1f} 倍（H4 不成立）；\"按负载开关投机\"口径下"
            "SQL、中文在饱和时永不回本，代码按固定长度口径也永不（E05）。")

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], [16, 10, 10, 10, 10, 10, 10, 10, 11, 12])
        s.title_(f"{e['id']} · {e['name']}", lead)
        s.section("实验配置")
        s.kv([("要回答", "在线更新草稿头多久能把花掉的训练算力赚回来？回本时间在不同 workload 之间差多少？（预注册 H4）"),
              ("作业与日期", "10-04（悉尼）。SQL 15249060、代码 15249061、中文 15249062、函数调用 15249063。"),
              ("模型与流量", "M1 Qwen3-8B + 原装 AngelSlim EAGLE-3 头，k=3，贪心。四个域（SQL、代码、中文、函数调用）的流量都是目标模型"
                          "自己生成的（见「通用配置」）：3200 条 train 当作\"已观测的线上流量\"，300 条 eval、300 条 reserve 留出。"),
              ("更新方式（E1 规则）", "单遍在线：按到达顺序，每条 train 请求只学一次，打包成约 4096 token 一步；目标 = 目标模型实际输出的下一个"
                                  " token（草稿词表上的交叉熵）；AdamW lr 5e-5。消费 N ∈ {100,200,400,800,1600,3200} 条后各导出一个头；"
                                  "记录只含头的 GPU 时间 C_train(N)。"),
              ("评测（vLLM 离线引擎，一个头一个进程）", "① 接受长度：本域 eval 300 条，另测对话 eval 300 条看迁移/遗忘；② 饱和：eval+reserve "
                                               "600 条一次提交，测 3 次取中位数 → 每请求 GPU 秒；③ 单请求延迟：12 条逐条跑，ms/token；"
                                               "④ 无损检查：输出与参照流量逐 token 比对。"),
              ("回本怎么算", "T_be(N) = C_train(N) ÷ s(N)，s(N) 是饱和时每请求省下的 GPU 秒，用接受长度模型 s = g(N0)·(1 − a(N0)/a(N))"
                          "（比两次饱和测量相减稳得多）；换算成\"饱和到达率下的分钟数\"。\"开关\"口径：只有新头比\"不投机\"和\"原头投机\""
                          "中较好者还快，才算收益。")])
        s.section("结果")
        cols = [("d", "域", None)] + [(f"n{N}", f"N{N}", F2) for N in Ns] + [("dlt", "增幅\nN3200−N0", "+0.00;-0.00"),
                                                                             ("cov", "草稿词表\n覆盖率", PCTP)]
        rows = []
        for d in doms:
            row = {"d": DOM_ZH[d], **{f"n{N}": at(d, N, "acc_len") for N in Ns}, "cov": COVER[d]}
            row["dlt"] = F("={n3200}{r}-{n0}{r}", acc3[d] - acc0[d])
            rows.append((d, row))
        s.table(cols, rows, caption="表 1　本域接受长度（N = 已在线学习的请求数；N0 = 原装头）",
                note="草稿词表覆盖率 = 目标模型输出 token 落在 32k 草稿词表里的比例（预注册变更记录）。中文只有 42.6%，头永远提不出其余 57%。")
        rows = []
        for d in doms:
            name = DOM_ZH[d]
            row = {"d": f"在 {name} 上学" if name.isascii() else f"在{name}上学", **{f"n{N}": at(d, N, "acc_len_chat") for N in Ns}}
            row["dlt"] = F("={n3200}{r}-{n0}{r}", at(d, 3200, "acc_len_chat") - at(d, 0, "acc_len_chat"))
            rows.append((d, row))
        s.table(cols[:-1], rows, caption="表 2　对话域接受长度（迁移 / 遗忘检查）：在别的域上学，对话变好还是变差")
        rows = []
        for d in doms:
            none = nones[d]
            rows.append((d, {"d": DOM_ZH[d], "b": none["batch1_ms_per_token"], "n0": at(d, 0, "b1_ms_tok"),
                             "n32": at(d, 3200, "b1_ms_tok"),
                             "v0": F("={n32}{r}/{n0}{r}-1", lat[d]),
                             "vb": F("={n32}{r}/{b}{r}-1", at(d, 3200, "b1_ms_tok") / none["batch1_ms_per_token"] - 1)}))
        s.table([("d", "域", None), ("b", "不投机\nms/token", F2), ("n0", "N0\nms/token", F2), ("n32", "N3200\nms/token", F2),
                 ("v0", "N3200 比 N0", PCT), ("vb", "N3200 比不投机", PCT)], rows,
                caption="表 3　单请求延迟（一次只跑一条，低负载口径）")
        rows = []
        for d in doms:
            g_none = nones[d]["saturation"]["gpu_s_per_req"]
            g0, g3 = at(d, 0, "sat_gpu_s_per_req"), at(d, 3200, "sat_gpu_s_per_req")
            rows.append((d, {"d": DOM_ZH[d], "gn": g_none * 1000, "g0": g0 * 1000, "g3": g3 * 1000,
                             "x0": F("={gn}{r}/{g0}{r}", g_none / g0, X2), "x3": F("={gn}{r}/{g3}{r}", g_none / g3, X2),
                             "ct": at(d, 3200, "C_train_gpu_s"),
                             "per": F("={ct}{r}/3200*1000", at(d, 3200, "C_train_gpu_s") / 3200 * 1000, F2)}))
        s.table([("d", "域", None), ("gn", "不投机\nGPU 毫秒/请求", F2), ("g0", "N0\nGPU 毫秒/请求", F2), ("g3", "N3200\nGPU 毫秒/请求", F2),
                 ("x0", "N0 相对\n不投机", X2), ("x3", "N3200 相对\n不投机", X2), ("ct", "头训练 GPU 秒\n（3200 条）", F1),
                 ("per", "每条请求\n训练 GPU 毫秒", F2)], rows,
                caption="表 4　饱和吞吐（600 条一次提交，变长口径）与训练花费",
                note="变长口径含排空段（后半段并发很低、投机占优），偏向投机，几个百分点的差别不可解读；固定长度复测见 E05。>1 表示比不投机快。")
        rows = [(d, {"d": DOM_ZH[d], **{f"n{N}": at(d, N, "Tbe_min_accmodel") for N in Ns[1:]}}) for d in doms]
        ratio = {}
        for N in Ns[1:]:
            vals = [at(d, N, "Tbe_min_accmodel") for d in doms]
            ratio[f"n{N}"] = F(f"=MAX({{@sql|n{N}}}:{{@fc|n{N}}})/MIN({{@sql|n{N}}}:{{@fc|n{N}}})", max(vals) / min(vals), X2, bold=True)
        rows.append(("ratio", {"d": V("最大 ÷ 最小", bold=True), **ratio}))
        s.table([("d", "域", None)] + [(f"n{N}", f"N{N}", F2) for N in Ns[1:]], rows,
                caption="表 5　回本时间（分钟，饱和到达率下）：\"投机一直开\"口径",
                note="s(N) 用接受长度模型 g(N0)·(1 − a(N0)/a(N))。预注册 H4 看 N200（K=200）这一列的最大 ÷ 最小。")
        fx3 = ctx["e05"]["fx"]["N3200"]
        concl = {"sql": f"永不：饱和时 N3200 也只追平不投机（{sat3['sql']:.3f}×，在噪声内）",
                 "code": f"永不：固定长度口径 N3200 仍只有 {fx3:.2f}×（E05）；变长口径的 {sat3['code']:.2f}× 来自排空段",
                 "zh": f"永不：词表天花板，饱和时 N3200 只有 {sat3['zh']:.2f}×",
                 "fc": "变长口径 N100–N1600 回本几分钟，但只快 1.01–1.04×（排空偏向内），读作打平"}
        rows = [(d, {"d": DOM_ZH[d], **{f"n{N}": at(d, N, "Tbe_min_policy") for N in Ns[1:]}, "c": concl[d]}) for d in doms]
        s.table([("d", "域", None)] + [(f"n{N}", f"N{N}", F1) for N in Ns[1:]] + [("c", "结论", None, 3)], rows,
                caption="表 6　回本时间（分钟）：\"按负载开关投机\"口径（空格 = 永不：新头比\"不投机\"和\"原头投机\"都不快）",
                note="这一口径下只有\"新头比不投机和原头里较好的那个还快\"才算收益；数字用变长口径的饱和测量，偏向投机。")
        rows = []
        for d in doms:
            rows.append((d, {"d": DOM_ZH[d], "none": nones[d]["sets"][d]["identical_frac"], "n0": at(d, 0, "identical"),
                             "n32": at(d, 3200, "identical")}))
        s.table([("d", "域", None), ("none", "不投机", PCTP), ("n0", "N0", PCTP), ("n32", "N3200", PCTP)], rows,
                caption="表 7　输出与参照流量逐 token 一致的比例（无损检查）",
                note="参照流量是离线引擎单独生成的；长输出域里连\"不投机\"也只有 43–73% 一致，是近似平局位置上浮点非确定性翻转 argmax"
                     "（批形状不同、多 token 验证走不同内核），一致率随头和批形状变化、并不单调（如中文 N3200 为 93%），不是投机改了输出；"
                     "未做 top-2 平局核验。")
        s.table([("h", "项目", None, 2), ("rule", "预注册条件", None, 3), ("res", "实测", None, 4), ("v", "判定", None)], [
            ("H4", {"h": "H4 回本时间跨 workload 差异大", "rule": "4 个 B 域的 T_be(K=200) 最大/最小 ≥10×",
                    "res": f"\"一直开\"口径 N200 时最大/最小 {r200:.1f}×；\"开关\"口径差异无穷大，但那是\"要不要开投机\"之别，不是摊销快慢之别",
                    "v": V("不成立", status="不成立")}),
            ("g1", {"h": "图 1 通过线", "rule": "存在负载带：训练代价不可忽略且投机收益 >0，≥2 种情形",
                    "res": "所有负载带都满足，三种情形（E02）", "v": V("通过", status="通过")}),
            ("g2", {"h": "图 2 通过线", "rule": "H4 成立，且一个域饱和时 T_be >10 分钟、另一个 <2 分钟",
                    "res": "开关口径下 SQL/中文\"永不\"、函数调用 0.5–3.5 分钟，字面满足；一直开口径不满足",
                    "v": V("半过", status="半过")}),
            ("rule", {"h": "按预注册规则", "rule": "两图都不过 → 停；只过一个 → 回报用户再定",
                      "res": "回报后进入调度器阶段（Rung 1），但目标函数重写：投机开关按情形+负载、训练占空比按线性代价、结构天花板判别",
                      "v": V("部分成立", status="部分成立")}),
        ], caption="表 8　预注册判定")
        s.section("怎么读")
        s.bullets([f"更新又便宜又有效：100 条请求、不到 1 GPU 秒，接受长度就上去；3200 条后低负载延迟降 "
                   f"{-max(lat[d] for d in ('sql', 'code', 'zh')):.0%}–{-min(lat[d] for d in ('sql', 'code', 'zh')):.0%}（函数调用只降 {-lat['fc']:.0%}）。",
                   "正迁移、几乎无遗忘：在 SQL、代码、中文上学完，对话域接受长度都上升——原装头对这个目标模型的非思考输出本来就没对齐，任何线上数据"
                   "都在帮它对齐，所以\"分布漂移\"里有一部分其实是\"出厂头没对齐\"。函数调用 N200 时对话也 +0.15，学到 N1600 之后才略降。",
                   f"饱和时稠密短输出的投机最多打平：原装头四个域都低于不投机（{min(sat0.values()):.2f}–{max(sat0.values()):.2f}×），更新到 N3200 后 "
                   f"SQL {sat3['sql']:.2f}×、代码 {sat3['code']:.2f}×、函数调用 {sat3['fc']:.2f}×（变长口径，偏向投机），中文 {sat3['zh']:.2f}×。",
                   "中文是词表天花板，不是漂移：接受率低不等于值得训，这是后来调度器\"结构天花板判别\"的来源。",
                   "回本快到构不成权衡：训练的价值几乎全部兑现在\"投机开着\"的时候，所以真正的决策变量是\"未来投机开不开\"。",
                   "函数调用 N200 之后接受长度回落（2.79 → 2.52），是学习率不衰减的单遍更新在短输出（约 30 token）上过拟合近期请求。"])
        s.source("data/runs/c2/m1/{sql,code,zh,fc}/{none,N0..N3200}/{eval,train_stats}.json；回本按 code/specsched/analyze_c2.py 计算；"
                 "图 results/figures/c2_m1.png；结论文档 2026-10-04_Rung0_结论.md §2、§3")
    return meta(ctx, "E04", short="在线更新回本", name="C2 在线更新多久回本（8B，四个域）",
                q="在线更新多久回本？回本时间在不同 workload 之间差多少？（H4）",
                cfg="8B，SQL / 代码 / 中文 / 函数调用，单遍在线更新 N=100–3200 条；离线引擎测接受长度、饱和吞吐、单请求延迟",
                res=f"100 条 ≤1 GPU 秒就见效；代码 {acc0['code']:.2f}→{acc3['code']:.2f}、SQL {acc0['sql']:.2f}→{acc3['sql']:.2f}、"
                    f"中文 {acc0['zh']:.2f}→{acc3['zh']:.2f}（词表天花板）；对话域基本正迁移；\"一直开\"口径回本 ≤{tmax:.1f} 分钟、"
                    f"N200 时域间 {r200:.1f}×",
                verdict=f"H4 不成立（{r200:.1f}×，<10×）；图 1 过、图 2 半过 → 进入调度器阶段，目标函数重写", word="不成立",
                status="有效", build=build)


def e05(ctx):
    reps = {}
    for d, h in [("code", "none"), ("code", "N400"), ("sql", "none"), ("sql", "N3200")]:
        base = f"{RUNS}/c2/m1/{d}/{h}"
        ws = [json.load(open(p))["saturation"]["wall_s"] for p in sorted(glob.glob(f"{base}/eval_rep*.json"))]
        reps[(d, h)] = (ws, json.load(open(f"{base}/eval.json"))["saturation"]["wall_s"])
    fixed = {}
    for d in ["code", "sql", "fc"]:
        for h in ["none", "N0", "N400", "N800", "N3200"]:
            p = f"{RUNS}/c2/m1/{d}/{h}/eval_fixed.json"
            if os.path.exists(p):
                fixed[(d, h)] = json.load(open(p))["saturation_fixed"]
    spread = {k: (max(v[0]) - min(v[0])) / np.mean(v[0]) for k, v in reps.items()}
    cv = {k: float(np.std(v[0], ddof=1) / np.mean(v[0])) for k, v in reps.items()}
    cv_dom = {d: max(cv[k] for k in cv if k[0] == d) for d in ("code", "sql")}
    fx = {h: fixed[("code", "none")]["wall_s"] / fixed[("code", h)]["wall_s"] for h in ["N0", "N400", "N800", "N3200"]}
    ctx["e05"] = dict(spread=spread, fx=fx, cv=cv, cv_max=max(cv.values()), rng_max=max(spread.values()))
    lead = (f"饱和吞吐的跨实例噪声很小：同一个头在 3 个独立实例里，用时的变异系数代码 ≤{cv_dom['code']:.2%}、SQL ≤{cv_dom['sql']:.2%}"
            f"（极差 ≤{max(spread.values()):.1%}）。但\"600 条一次提交\"的变长口径含排空段、偏向投机："
            f"换成固定长度（每条 256 token）后，代码域 N0 只有不投机的 {fx['N0']:.2f}×，学到 N3200 也只有 {fx['N3200']:.2f}×。"
            "稠密短输出在饱和时，在线学习够得着的接受长度内，投机不可能反超不投机。")

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], [22, 12, 12, 12, 12, 12, 12, 14])
        s.title_(f"{e['id']} · {e['name']}", lead)
        s.section("实验配置")
        s.kv([("要回答", "E04 的饱和吞吐有多少是测量噪声？\"600 条真实请求一次提交\"这个口径有没有系统性偏向？"),
              ("作业与日期", "10-04（悉尼）。噪声标定 15250120；固定长度复测 15251522。"),
              ("噪声标定", "同一个头在 3 个独立的 vLLM 实例里各测一次饱和（每次本身也是 3 次取中位数），与 E04 原评测一起比较。"
                        "代码：不投机、N400；SQL：不投机、N3200。"),
              ("固定长度口径", "ignore_eos=True，每条强制生成 256 token；600 条一次提交，测 3 次取中位数。这样所有头的 token 数和排空形状完全一致。"
                           "只对真实输出够长的代码域有效；SQL/函数调用真实输出只有约 30 token，256 token 里 90% 是 EOS 之后的重复文本。")])
        s.section("结果")
        rows = []
        for (d, h), (ws_, orig) in reps.items():
            rows.append((f"{d}{h}", {"k": f"{DOM_ZH[d]}，{'不投机' if h == 'none' else h}", "r1": ws_[0], "r2": ws_[1], "r3": ws_[2],
                                     "o": orig, "m": F("=AVERAGE({r1}{r}:{r3}{r})", np.mean(ws_), F3),
                                     "cv": F("=STDEV({r1}{r}:{r3}{r})/AVERAGE({r1}{r}:{r3}{r})", cv[(d, h)], "0.00%"),
                                     "sp": F("=(MAX({r1}{r}:{r3}{r})-MIN({r1}{r}:{r3}{r}))/AVERAGE({r1}{r}:{r3}{r})", spread[(d, h)], "0.00%")}))
        s.table([("k", "域与头", None), ("r1", "实例 1\n秒", F3), ("r2", "实例 2\n秒", F3), ("r3", "实例 3\n秒", F3),
                 ("o", "E04 原评测\n秒", F3), ("m", "3 实例均值", F3), ("cv", "变异系数\n标准差÷均值", "0.00%"),
                 ("sp", "极差 ÷ 均值", "0.00%")], rows,
                caption="表 1　600 条一次提交的用时（秒，越小越快）：跨实例噪声")
        rows = []
        for h in ["none", "N0", "N400", "N800", "N3200"]:
            f_ = fixed[("code", h)]
            rows.append((h, {"k": "不投机" if h == "none" else h, "w": f_["wall_s"], "g": f_["gpu_s_per_req"] * 1000,
                             "x": F("={@none|w}/{w}{r}", fixed[("code", "none")]["wall_s"] / f_["wall_s"], X2) if h != "none" else V(1.0, fmt=X2),
                             "acc": f_.get("acc_len")}))
        s.table([("k", "代码域的头", None), ("w", "600 条用时 秒", F2), ("g", "GPU 毫秒/请求", F2), ("x", "相对不投机", X2),
                 ("acc", "接受长度\n（EOS 后文本）", F2)], rows,
                caption="表 2　固定长度口径（每条 256 token）：代码域",
                note="这一口径下的接受长度是在 EOS 之后的重复文本上测的，不能当接受长度指标，只作参考。")
        e4 = ctx["e04"]["dfs"]["code"]
        none_g = ctx["e04"]["nones"]["code"]["saturation"]["gpu_s_per_req"]
        rows = []
        for h, N in [("N0", 0), ("N400", 400), ("N800", 800), ("N3200", 3200)]:
            var = none_g / float(e4[e4.N == N]["sat_gpu_s_per_req"].iloc[0])
            rows.append((h, {"k": h, "var": var, "fix": fx[h], "d": F("={fix}{r}-{var}{r}", fx[h] - var, "+0.00;-0.00")}))
        s.table([("k", "代码域的头", None), ("var", "变长口径\n相对不投机", X2), ("fix", "固定长度\n相对不投机", X2), ("d", "差", "+0.00;-0.00")],
                rows, caption="表 3　两种口径对比（代码域）",
                note="变长口径里 N800 比 N400 慢（\"接受长度更高但饱和时间 +11%\"），固定长度下消失：是输出集效应（贪心输出在近似平局处翻转后个别请求变长，"
                     "拖长了排空尾巴），不是头变差。")
        rows = []
        for d in ["sql", "fc"]:
            for h in ["none", "N0", "N3200"]:
                if (d, h) in fixed:
                    f_ = fixed[(d, h)]
                    rows.append((f"{d}{h}", {"k": f"{DOM_ZH[d]}，{'不投机' if h == 'none' else h}", "w": f_["wall_s"],
                                             "x": (fixed[(d, "none")]["wall_s"] / f_["wall_s"]) if h != "none" else 1.0}))
        s.table([("k", "域与头", None), ("w", "600 条用时 秒", F2), ("x", "相对不投机", X2)], rows,
                caption="表 4　固定长度口径在 SQL / 函数调用上（不采用）",
                note="这两个域的真实输出约 30 token，256 token 里绝大部分是 EOS 之后的文本，头是在真实输出上学的，所以更新后反而略差；数字不采用。")
        s.section("怎么读")
        s.bullets([f"跨实例噪声：变异系数 ≤{max(cv.values()):.1%}、极差 ≤{max(spread.values()):.1%}；E04 里几个百分点以内的\"追平/反超\"一律读作打平。",
                   "N0 相对不投机慢 13%–54%（变长口径），远超噪声，\"饱和时原装头投机是负收益\"不受影响。",
                   "两种口径互补：变长口径偏向投机；固定长度口径只对长输出域有效。报告时并列。"])
        s.source("data/runs/c2/m1/{code,sql}/{none,N400,N3200}/eval_rep{1,2,3}.json；data/runs/c2/m1/{code,sql,fc}/*/eval_fixed.json；"
                 "结论文档 2026-10-04_Rung0_结论.md §2.1b、§4")
    return meta(ctx, "E05", short="口径校验", name="C2 口径校验：跨实例噪声与固定长度复测",
                q="C2 的饱和吞吐有多少是噪声？\"600 条一次提交\"的口径有没有偏向？",
                cfg="同一头 3 个独立实例重复测（代码 none/N400、SQL none/N3200）；固定长度 ignore_eos 256 token 复测",
                res=f"跨实例噪声：变异系数 ≤{max(cv.values()):.1%}（极差 ≤{max(spread.values()):.1%}）；固定长度下代码 N0 {fx['N0']:.2f}× → "
                    f"N3200 {fx['N3200']:.2f}×（变长口径的\"追平\"来自排空段）",
                verdict="变长口径偏向投机；稠密短输出饱和时投机在可学到的接受长度内反超不了不投机", word="有效", status="有效",
                build=build)


def e06(ctx):
    Cs = [1, 4, 16, 64, 128, 256]
    heads = ["S1_N0", "S1_N100", "S1_N400", "S1_N3200"]
    fr = {d: {h: frontier(f"m1_{d}", h) for h in ["S0"] + heads} for d in ["code", "sql", "zh"]}
    model = jload(f"{ROOT}/code/specsched/sched/model.json")
    fits = model["frontier"]
    sp = {d: {h: {C: fr[d][h][C]["output_throughput"] / fr[d]["S0"][C]["output_throughput"] for C in Cs if C in fr[d][h]}
              for h in heads if fr[d][h]} for d in fr}
    res_max = {d: max(f["max_abs_err"] for f in fits[d]["fit"].values()) for d in fits}
    step = model["spec_step"]
    ctx["e06"] = dict(sp=sp, res_max=res_max)
    c_sp = sp["code"]
    lead = (f"投机倍数对接受长度是线性的：gain = α(C)·a + β(C)，代码、中文各并发最大残差 ≤{max(res_max['code'], res_max['zh']):.2f}，"
            f"SQL 最大 {res_max['sql']:.2f}（C=16）。长输出域（代码、中文）贴近闭式 a·T_base/T_spec，短输出（SQL）斜率低得多。"
            f"代码域学 100 条就让饱和时投机由负转正（{c_sp['S1_N0'][256]:.2f}× → {c_sp['S1_N100'][256]:.2f}×），N3200 到 "
            f"{c_sp['S1_N3200'][256]:.2f}×；SQL 学到天花板饱和时也只有 {sp['sql']['S1_N3200'][256]:.2f}×；中文的天花板在 C≥64 "
            "永远到不了盈亏线。")

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], [22, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        s.title_(f"{e['id']} · {e['name']}", lead)
        s.section("实验配置")
        s.kv([("要回答", "投机比不投机快多少倍（gain），和头的接受长度 a、并发 C 是什么关系？能不能用闭式 a·T_base(C)/T_spec(C) 预测？"),
              ("作业与日期", "10-05（悉尼）。代码 15263570；SQL + 中文 15263571。第一轮 15263307/15263308 因 OSC 同节点端口碰撞作废重跑（见「作废与教训」）。"),
              ("设置", "M1 Qwen3-8B，vllm serve（同 E02 的服务配置）；头 = 不投机（S0）、E04 导出的 N0 / N100 / N400 / N3200（中文没跑 N3200）；"
                     "C ∈ {1, 4, 16, 64, 128, 256}，每档请求数 min(max(32, 4C), 1024)；用本域 eval 提示，自然停止，输出 ≤512。"),
              ("拟合", "每个并发 C 上，对各头的 (接受长度 a, 实测倍数 gain) 做线性拟合 gain = α(C)·a + β(C)；盈亏接受长度 a* = (1 − β) / α。"
                     "参数在 code/specsched/sched/model.json（fit_model.py 于 10-08 重拟合，含 N400 共 4 个头；结论文档 10-05 版的 SQL 拟合点更少，数值略有不同）。"),
              ("闭式", "T_base(C)、T_spec(C) = 不投机 / 投机时每步解码耗时，取自 E02 的对话 S0/S1；T_spec 与头无关（k 和头结构固定），"
                     "所以 gain(a, C) = a · T_base(C) / T_spec(C)，盈亏接受长度 a*(C) = T_spec / T_base。")])
        s.section("结果")
        cols = [("h", "头", None)] + [(f"c{C}", f"C={C}", INT) for C in Cs]
        hname = {"S0": "不投机", "S1_N0": "N0（原装头）", "S1_N100": "N100", "S1_N400": "N400", "S1_N3200": "N3200"}
        for d, dz in [("code", "代码"), ("sql", "SQL"), ("zh", "中文")]:
            hs = [h for h in ["S0"] + heads if fr[d][h]]
            pr = s.table(cols, [(h, {"h": V(hname[h], bold=(h == "S0")), **{f"c{C}": fr[d][h][C]["output_throughput"] for C in Cs}})
                                for h in hs], caption=f"{dz}域　吞吐 tok/s")
            rows = []
            for h in hs[1:]:
                accs = [fr[d][h][C]["acc_len"] for C in Cs if fr[d][h][C].get("acc_len")]
                rows.append((h, {"h": hname[h], "a": float(np.mean(accs)),
                                 **{f"c{C}": F(f"={pr[(h, f'c{C}')]}/{pr[('S0', f'c{C}')]}", sp[d][h][C], X2) for C in Cs}}))
            s.table([("h", "头", None), ("a", "接受长度\n（各 C 均值）", F2)] + [(f"c{C}", f"C={C}", X2) for C in Cs], rows,
                    caption=f"{dz}域　相对不投机的倍数（公式；<1 表示投机反而更慢）")
            frows = []
            for C in Cs:
                f_ = fits[d]["fit"][str(C)]
                frows.append((C, {"c": f"C={C}", "al": f_["alpha"], "be": f_["beta"], "n": f_["n"], "err": f_["max_abs_err"],
                                  "ast": F("=(1-{be}{r})/{al}{r}", (1 - f_["beta"]) / f_["alpha"], F2)}))
            s.table([("c", "并发", None), ("al", "α(C)\n斜率", F3), ("be", "β(C)\n截距", F3), ("n", "头数", INT),
                     ("err", "最大残差", F3), ("ast", "盈亏接受长度\na* =(1−β)/α", F2)], frows,
                    caption=f"{dz}域　线性拟合 gain = α(C)·a + β(C)")
        rows = []
        code_acc = {h: {C: fr["code"][h][C]["acc_len"] for C in Cs} for h in ["S1_N0", "S1_N3200"]}
        for C in Cs:
            st = step[str(C)]
            rows.append((C, {"c": f"C={C}", "tb": st["T_base"], "ts": st["T_spec"],
                             "ast": F("={ts}{r}/{tb}{r}", st["T_spec"] / st["T_base"], F2),
                             "a0": code_acc["S1_N0"][C],
                             "p0": F("={a0}{r}*{tb}{r}/{ts}{r}", code_acc["S1_N0"][C] * st["T_base"] / st["T_spec"], X2),
                             "m0": sp["code"]["S1_N0"][C],
                             "a3": code_acc["S1_N3200"][C],
                             "p3": F("={a3}{r}*{tb}{r}/{ts}{r}", code_acc["S1_N3200"][C] * st["T_base"] / st["T_spec"], X2),
                             "m3": sp["code"]["S1_N3200"][C],
                             "e3": F("={p3}{r}/{m3}{r}-1", code_acc["S1_N3200"][C] * st["T_base"] / st["T_spec"] / sp["code"]["S1_N3200"][C] - 1)}))
        s.table([("c", "并发", None), ("tb", "T_base\nms/步", F2), ("ts", "T_spec\nms/步", F2), ("ast", "闭式 a*\nT_spec÷T_base", F2),
                 ("a0", "代码 N0\n接受长度", F2), ("p0", "N0 预测\n倍数", X2), ("m0", "N0 实测\n倍数", X2), ("a3", "代码 N3200\n接受长度", F2),
                 ("p3", "N3200 预测\n倍数", X2), ("m3", "N3200 实测\n倍数", X2), ("e3", "N3200\n预测误差", PCT)], rows,
                caption="闭式预测 vs 实测（步时取自 E02 的对话 S0/S1；预测 = a × T_base ÷ T_spec）")
        s.section("怎么读")
        s.bullets(["线性成立：三个域、各并发对接受长度都是直线；但 T 曲线要按负载形状（提示/输出长度）标定，不能只按模型。",
                   "代码（长输出）：比例形式（β≈0）在 a∈[2.1, 2.7] 内误差 ≤5%，高负载端闭式略偏乐观（接受长度高时每步 KV 增长更快）。",
                   "SQL（27 token 输出）：步时被 prefill 和固定开销主导，接受长度的作用被稀释，中高负载下 +1 接受长度只值 +15–25% 吞吐（代码是 +40–57%）。"
                   "饱和时投机不划算，唯一的杠杆是关掉。",
                   "中文：斜率与比例模型一致，问题不在 frontier 而在头——天花板 1.45 在 C≥64 永远到不了 a*，原装头从并发 4 起就是净负。",
                   "这张表就是调度器规则 1（按 frontier 开关投机）和规则 0（结构天花板判别）的依据。"])
        s.source("data/runs/frontier/m1_{code,sql,zh}/{S0,S1_N*}/summary.json；code/specsched/sched/model.json（frontier、spec_step）；"
                 "图 results/figures/frontier_m1.png；结论文档 2026-10-05_Rung1_调度器.md §1")
    return meta(ctx, "E06", short="frontier", name="frontier：负载 × 头的投机倍数（8B，三个域）",
                q="投机倍数和接受长度、并发是什么关系？能不能用闭式预测？",
                cfg="8B，代码 / SQL / 中文 × 头 N0 / N100 / N400 / N3200 × 并发 1–256，vllm serve 实测",
                res=f"gain = α(C)·a + β(C) 线性（残差 ≤{max(res_max['code'], res_max['zh']):.2f}，SQL C=16 为 {res_max['sql']:.2f}）；"
                    f"代码学 100 条饱和由 {c_sp['S1_N0'][256]:.2f}× 转正到 {c_sp['S1_N100'][256]:.2f}×；SQL 到天花板也只 "
                    f"{sp['sql']['S1_N3200'][256]:.2f}×；中文到不了盈亏线",
                verdict="成立：线性、可标定，但要按负载形状分别标定", word="成立", status="有效（含作废轮次）", build=build)


def e07(ctx):
    heads = ["N0", "N100", "N400", "N3200"]
    ev = {h: jload(f"{RUNS}/c2/m1/chat/{h}/eval.json") for h in heads}
    none = jload(f"{RUNS}/c2/m1/chat/none/eval.json")
    doms = ["chat", "sql", "code", "zh", "fc"]
    acc = {h: {d: ev[h]["sets"][d]["acc_len"] for d in doms} for h in heads}
    ctx["e07"] = acc
    lead = (f"对话域在线学 3200 条：{acc['N0']['chat']:.2f} → {acc['N3200']['chat']:.2f}。学完的对话头在别的域上不降反升"
            f"（SQL {acc['N0']['sql']:.2f}→{acc['N3200']['sql']:.2f}、代码 {acc['N0']['code']:.2f}→{acc['N3200']['code']:.2f}、"
            f"中文 {acc['N0']['zh']:.2f}→{acc['N3200']['zh']:.2f}），只有函数调用微降（{acc['N0']['fc']:.2f}→{acc['N3200']['fc']:.2f}）。"
            "这给漂移实验提供了\"A 已适应、再切到 B\"的起点。")

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], [22, 12, 12, 12, 12, 12, 13])
        s.title_(f"{e['id']} · {e['name']}", lead)
        s.section("实验配置")
        s.kv([("要回答", "先在对话域上在线适应，再看接受长度在别的域会不会掉——漂移实验需要\"源域已适应、目标域掉多少\"这组数。"),
              ("作业与日期", "10-05（悉尼），15263309。"),
              ("设置", "M1；对话域 train 3200 条单遍在线更新（同 E04 的 E1 规则），导出 N100 / N400 / N3200；"
                     "每个头评测对话 + SQL / 代码 / 中文 / 函数调用各 300 条 eval；单请求延迟 6 条。")])
        s.section("结果")
        rows = [(h, {"h": "N0（原装头）" if h == "N0" else h, **{d: acc[h][d] for d in doms},
                     "b1": ev[h]["batch1_ms_per_token"]}) for h in heads]
        rows.append(("dlt", {"h": V("N3200 − N0", bold=True), **{d: F(f"={{@N3200|{d}}}-{{@N0|{d}}}", acc["N3200"][d] - acc["N0"][d],
                                                                   "+0.00;-0.00") for d in doms}}))
        s.table([("h", "在对话上学了多少", None)] + [(d, f"在{DOM_ZH[d]}上\n接受长度", F2) for d in doms] + [("b1", "对话单请求\nms/token", F2)],
                rows, caption="表 1　各头在五个域上的接受长度",
                note=f"对话域不投机的单请求延迟为 {none['batch1_ms_per_token']:.2f} ms/token。")
        s.section("怎么读")
        s.bullets(["正迁移：对话头学完后在 SQL、代码、中文上都更好，说明原装头对这个目标模型的非思考输出本来就欠对齐。",
                   "函数调用微降，是唯一的负迁移，幅度很小。",
                   "对调度器的含义：想测\"训还是不训\"的决策，必须先把头对齐（消掉出厂欠对齐），否则任何策略只要训就赢，区分不出调度好坏。"])
        s.source("data/runs/c2/m1/chat/{none,N0,N100,N400,N3200}/eval.json；sched/model.json（domains.*.chat_curve）")
    return meta(ctx, "E07", short="对话域适应", name="C2 补充：对话域在线适应与跨域迁移",
                q="在对话域适应后，接受长度在其它域会不会掉？（漂移实验的起点）",
                cfg="8B，对话域单遍在线 N=100 / 400 / 3200，五个域各 300 条评测",
                res=f"对话 {acc['N0']['chat']:.2f}→{acc['N3200']['chat']:.2f}；在 SQL、代码、中文上不降反升，函数调用微降"
                    f"（{acc['N0']['fc']:.2f}→{acc['N3200']['fc']:.2f}）",
                verdict="正迁移、几乎无遗忘", word="有效", status="有效", build=build)


TRACE = [  # results/summaries/2026-10-05_Rung1_调度器.md §2.4 (simulator, 10-05; no CSV saved)
    ("static_off", "不投机、不训", 228.8, 3970, 0.0, 241.1, 1340, 0.0),
    ("spec_only", "投机常开（原装头）", 215.6, 5630, 0.0, 207.7, 4280, 0.0),
    ("osd_idle", "投机常开，并发 <16 就全速训（OSD 式\"空闲就训\"）", 251.1, 2420, 215.7, 257.2, 50, 121.3),
    ("accdrop", "接受长度低于天花板 95% 就全速训", 243.1, 2770, 0.07, 256.7, 90, 0.05),
    ("costmodel", "cost-model 调度器（persist 预测器）", 253.4, 1970, 1.15, 257.2, 50, 1.24),
]


def e08(ctx):
    t = {r[0]: r for r in TRACE}
    cm, so, sp, osd = t["costmodel"], t["static_off"], t["spec_only"], t["osd_idle"]
    ctx["e08"] = dict(vs_off=cm[2] / so[2] - 1, vs_spec=cm[2] / sp[2] - 1, vs_osd=cm[2] / osd[2] - 1, ratio=osd[4] / cm[4])
    e = ctx["e08"]
    lead = (f"24 小时真实 trace 回放（模拟）：cost-model 调度器在对话日志上 goodput 最高（{cm[2]}M token，比不投机 {e['vs_off']:+.1%}、"
            f"比投机常开 {e['vs_spec']:+.1%}、比 OSD 式\"空闲就训\" {e['vs_osd']:+.1%}），训练量只有 OSD 的 1/{e['ratio']:.0f}。"
            "原装头投机常开反而比不投机差（峰值时 1.91 < a*(256)=2.20）。但对 OSD 只多 0.9%，说明\"什么时候训\"大体是退化问题。")

    def build(wb, ctx, e_):
        s = Sheet(wb, e_["sheet"], [20, 30, 12, 10, 10, 10, 12, 10, 10, 10])
        s.title_(f"{e_['id']} · {e_['name']}", lead)
        s.section("实验配置")
        s.kv([("要回答", "用 Rung 0 实测曲线搭的最小调度器，在 24 小时真实负载曲线上，比简单策略好多少？"),
              ("性质", "本机模拟（不占 GPU），10-05。所有参数来自实测：T_base/T_spec（E02）、κ 与训练速率 52.8k token/s/占空比（E02/E03）、"
                     "各域学习曲线与迁移（E04/E07）、按域 frontier（E06）。"),
              ("模拟器", "code/specsched/sched/sim.py：单卡流体模型。到达率 λ(t) → 解出平衡并发 C；thr(C) = 不投机吞吐 × [投机时 gain(a,C)] × "
                      "(1 − κ(C)·占空比)；需求超过 C=256 的容量按 token 积压；服务过的请求进入 on-policy 缓冲（后进先出），训练按 "
                      "占空比 × 52.8k token/s 消耗缓冲，头的接受长度按学习曲线更新。"),
              ("目标与决策", "goodput = TPOT ≤ 16.7 ms（= 2 × T_base(1)）且排队 ≤1 s 的窗口里服务的 token。每秒决策：投机开/关、"
                          "占空比 ∈ {0, 0.1, 0.25, 0.5, 1}。"),
              ("trace", "BurstGPT：对话日志用第 1–8 天、API 日志用第 9–10 天，缩放到 p95 = 单卡容量的 80%，对话域，24 h，10 s 步长。"),
              ("调度器三条规则", "① 投机开关：gain(a,C) > 1 才开；② SLO 内训练免费（只付延迟），取仍在 SLO 内的最大占空比；"
                              "③ 已违约时按\"连训 10 秒的 goodput 增量 − 吞吐损失\"选占空比；另有结构天花板判别（天花板到不了 a* 就不训）。")])
        s.section("结果")
        rows = []
        for k, desc, g1, v1, t1, g2, v2, t2 in TRACE:
            rows.append((k, {"k": V(k, bold=(k == "costmodel")), "d": desc, "g1": g1, "x1": F("={g1}{r}/{@static_off|g1}-1", g1 / so[2] - 1, PCT),
                             "v1": v1, "t1": t1, "g2": g2, "x2": F("={g2}{r}/{@static_off|g2}-1", g2 / so[5] - 1, PCT),
                             "v2": v2, "t2": t2}))
        s.table([("k", "策略", None), ("d", "做法", None), ("g1", "对话日志\ngoodput M token", F1), ("x1", "相对\n不投机", PCT),
                 ("v1", "违约秒", INT), ("t1", "训练 token\nM", "0.00"), ("g2", "API 日志\ngoodput M token", F1), ("x2", "相对\n不投机", PCT),
                 ("v2", "违约秒", INT), ("t2", "训练 token\nM", "0.00")], rows,
                caption="表 1　24 小时 trace 回放（模拟器）",
                note="数字转录自结论文档 2026-10-05_Rung1_调度器.md §2.4：当时模拟器没有存 CSV；sched/model.json 在 10-08 重拟合过，"
                     "现在重跑会有小幅差别。图 results/figures/sim_trace_timeline.png。")
        s.section("怎么读")
        s.bullets(["投机常开反而比不投机差：峰值时原装头的接受长度 1.91 低于 a*(256) = 2.20，这就是\"原装头直接部署\"的真实风险。",
                   "cost-model ≥ OSD，训练量少两个数量级；accdrop 略差是因为\"95% 天花板\"门槛停得早（2.29）。",
                   "差异几乎全部来自峰值时段；非峰值时所有策略都满足 SLO，分不出高下。",
                   "对 OSD 只多 0.9%（低于 2026-09-29 定的 K1 线 5%）：用实测学习曲线时\"什么时候训\"大体退化——on-policy 适应 100 条请求"
                   "只要 0.2–1 GPU 秒。调度器真正起作用的是三件事：按 frontier 开关投机、SLO 内把训练当免费资源用满、识别结构天花板不白训。"])
        s.source("结论文档 results/summaries/2026-10-05_Rung1_调度器.md §2；code/specsched/sched/{sim.py,plot_sim.py,model.json}；"
                 "data/traces/BurstGPT_without_fails_1.csv")
    return meta(ctx, "E08", short="trace回放(模拟)", name="模拟器：24 小时真实 trace 回放", date="10-05（本机）",
                q="用实测曲线搭的最小调度器，在 24 小时真实负载上比简单策略好多少？",
                cfg="本机流体模拟器；BurstGPT 对话 / API 日志 24 h，缩放到 p95 = 单卡容量 80%，10 s 步长；5 种策略",
                res=f"cost-model {cm[2]}M vs 不投机 {so[2]}M（{e['vs_off']:+.1%}）、投机常开 {sp[2]}M、OSD {osd[2]}M（{e['vs_osd']:+.1%}）；"
                    f"训练量为 OSD 的 1/{e['ratio']:.0f}",
                verdict="调度器最好，但只比 OSD 高 0.9%（<5%）：\"什么时候训\"大体退化，价值在三个判别", word="部分成立",
                status="有效（模拟）", build=build)


POL_ZH = {"static_off": "不投机不训", "spec_only": "投机常开", "osd_idle": "空闲就训（OSD）", "accdrop": "掉了就训",
          "costmodel_oracle": "调度器·oracle", "costmodel_persist": "调度器·persist", "costmodel_lindy": "调度器·lindy"}


def e09(ctx):
    d = pd.read_csv(f"{ROOT}/results/summaries/drift_sim.csv")
    dc = pd.read_csv(f"{ROOT}/results/summaries/drift_sim_chatA.csv")
    pols = list(POL_ZH)

    def g(df, exp, B, dur=None, cb=None, pol=None, col="goodput"):
        q = df[(df.exp == exp) & (df.B == B) & (df.policy == pol)]
        if dur is not None:
            q = q[q.dur_s == dur]
        if cb is not None:
            q = q[q.C_B == cb]
        return float(q[col].iloc[0]) if len(q) else None

    zh_cm, zh_osd = g(d, "II_ceiling", "zh", pol="costmodel_persist") / 1e6, g(d, "II_ceiling", "zh", pol="osd_idle") / 1e6
    s10_cm = g(d, "III_future_load", "sql", cb="128->surge1.0", pol="costmodel_persist") / 1e6
    s10_osd = g(d, "III_future_load", "sql", cb="128->surge1.0", pol="osd_idle") / 1e6
    ctx["e09"] = dict(zh_cm=zh_cm, zh_osd=zh_osd, s10_cm=s10_cm, s10_osd=s10_osd)
    lead = (f"\"同样的接受率下降、不同的未来机会\"：调度器在三类里从不输。拉开差距的是结构天花板（中文：反应式策略和投机常开把 goodput "
            f"砍到 {zh_osd:.2f}M，调度器关投机、不训，保住 {zh_cm:.2f}M）和浪涌（平静期把头免费训满，1.0× 浪涌下 {s10_cm:.1f}M vs "
            f"OSD/accdrop {s10_osd:.1f}M）。持续时长这个轴的分界只有几秒：适应太便宜，反应式策略已经接近最优。")

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], [20, 12, 12, 12, 12, 12, 12, 12])
        s.title_(f"{e['id']} · {e['name']}", lead)
        s.section("实验配置")
        s.kv([("要回答", "同样一次接受率下降，未来的机会不同（持续多久、天花板够不够得着盈亏线、之后会不会来浪涌），调度器会不会做出不同、"
                       "而且更好的决策？"),
              ("性质", "本机模拟（同 E08 的模拟器），10-05。规则与模型在看到结果后修过三处建模、三处规则（见结论文档变更记录），属模型调试，不是预注册实验。"),
              ("场景", "源域 A 已适应（主表 A = 代码 N3200；另一版 A = 对话，结论相同），切到 B。"
                     "I 持续时长：B 饱和（需求 = 不投机容量），持续 3–1800 秒，B = 代码（原装头，frontier 陡）/ SQL（对话适应后迁移，frontier 平）。"
                     "II 结构天花板：B 饱和持续 20 分钟，B = 代码 / SQL / 中文。III 未来负载：B = SQL 先 10 分钟中等负载（C≈128，SLO 内），"
                     "再 10 分钟浪涌，幅度 = 不投机容量的 0.95–1.1 倍。"),
              ("策略", "不投机不训；投机常开；空闲就训（OSD：并发 <16 全速训）；掉了就训（accdrop：低于天花板 95% 全速训）；"
                     "调度器 × 三种预测器（oracle 真未来 / persist 当前延续 / lindy 再持续\"已持续时长\"）。")])
        s.section("结果（goodput 单位：百万 token；违约秒 = 不满足 SLO 的秒数）")
        pcols = [("k", "场景", None)] + [(p, POL_ZH[p], F2) for p in pols]
        for B, bz in [("code", "代码"), ("sql", "SQL")]:
            rows = [(f"{B}{dur}", {"k": f"B={bz}，持续 {dur} s", **{p: g(d, "I_persistence", B, dur=dur, pol=p) / 1e6 for p in pols}})
                    for dur in [3, 10, 30, 120, 600, 1800]]
            s.table(pcols, rows, caption=f"I 持续时长：B={bz}　goodput")
            rows = [(f"{B}{dur}v", {"k": f"B={bz}，持续 {dur} s", **{p: V(g(d, "I_persistence", B, dur=dur, pol=p, col="slo_viol_s"), fmt=INT)
                                                                for p in pols}}) for dur in [3, 10, 30, 120, 600, 1800]]
            s.table([(c, h, INT) for c, h, _ in pcols], rows, caption=f"I 持续时长：B={bz}　违约秒")
        rows = [(B, {"k": f"B={bz}", **{p: g(d, "II_ceiling", B, pol=p) / 1e6 for p in pols}})
                for B, bz in [("code", "代码（天花板远高于 a*）"), ("sql", "SQL（天花板≈a*，边缘）"), ("zh", "中文（天花板 1.45 < a*）")]]
        s.table(pcols, rows, caption="II 结构天花板（B 饱和 20 分钟）　goodput")
        rows = [(B, {"k": f"B={bz}", **{p: V(g(d, "II_ceiling", B, pol=p, col="slo_viol_s"), fmt=INT) for p in pols}})
                for B, bz in [("code", "代码"), ("sql", "SQL"), ("zh", "中文")]]
        s.table([(c, h, INT) for c, h, _ in pcols], rows, caption="II 结构天花板　违约秒")
        surges = [("none", "无浪涌"), ("0.95", "0.95×"), ("1.0", "1.0×"), ("1.05", "1.05×"), ("1.1", "1.1×")]
        rows = [(sg, {"k": lab, **{p: g(d, "III_future_load", "sql", cb=f"128->surge{sg}", pol=p) / 1e6 for p in pols}}) for sg, lab in surges]
        s.table(pcols, rows, caption="III 未来负载（B=SQL，先中等负载 10 分钟，再浪涌 10 分钟）　goodput")
        rows = [(sg, {"k": lab, **{p: V(g(d, "III_future_load", "sql", cb=f"128->surge{sg}", pol=p, col="slo_viol_s"), fmt=INT)
                                   for p in pols}}) for sg, lab in surges]
        s.table([(c, h, INT) for c, h, _ in pcols], rows, caption="III 未来负载　违约秒")
        rows = [(sg, {"k": lab, **{p: g(d, "III_future_load", "sql", cb=f"128->surge{sg}", pol=p, col="acc_B") for p in pols}})
                for sg, lab in surges]
        s.table([(c, h, F2) for c, h, _ in pcols], rows, caption="III 未来负载　B 段结束时的接受长度")
        rows = []
        for exp, B, cb, lab in [("II_ceiling", "sql", None, "II 天花板，B=SQL"), ("II_ceiling", "zh", None, "II 天花板，B=中文"),
                                ("III_future_load", "sql", "128->surgenone", "III 无浪涌"), ("III_future_load", "sql", "128->surge1.0", "III 浪涌 1.0×")]:
            rows.append((lab, {"k": lab, **{p: g(dc, exp, B, cb=cb, pol=p) / 1e6 if g(dc, exp, B, cb=cb, pol=p) is not None else None
                                             for p in pols}}))
        s.table(pcols, rows, caption="对照版：源域 A = 对话（drift_sim_chatA.csv）　goodput")
        s.section("怎么读")
        s.bullets(["代码域（frontier 陡）：任何对接受率下降做反应的策略都在 1 秒内翻盘（原装头 0.93× → 学 100 条后 1.07×）；持续时间要短到 3 秒以内才不值得训；"
                   "三种预测器没有差别。这是\"适应太便宜、时机不重要\"的典型。",
                   "SQL 域（frontier 平）：天花板处投机只比不投机好 3%，TPOT 卡在 SLO 线附近。调度器多训 10 秒把头推过线，之后整段达标；"
                   "accdrop 和 OSD 差一点点够不到。这是刀口上的差别（接受长度差 0.04），只能标\"边缘\"，不能当稳健结论。",
                   "中文是\"同样 drop、不同机会\"最硬的版本：调度器判定天花板够不到盈亏线 → 关投机、不训，goodput 是反应式策略的 2 倍。",
                   "浪涌：差别不在\"预测到浪涌\"，而在\"平静期免费时是否训满\"——调度器训到天花板 2.82，OSD/accdrop 停在 2.52–2.56。"])
        s.source("results/summaries/drift_sim.csv、drift_sim_chatA.csv（code/specsched/sched/drift_exp.py 生成）；图 results/figures/sim_drift.png；"
                 "结论文档 2026-10-05_Rung1_调度器.md §3")
    return meta(ctx, "E09", short="漂移三类(模拟)", name="模拟器：漂移三类（持续时长 / 结构天花板 / 未来负载）", date="10-05（本机）",
                q="同样的接受率下降、不同的未来机会，调度器的决策会不会不同、更好？",
                cfg="本机模拟：A 已适应 → 切到 B；I 持续 3–1800 s；II 结构天花板（代码 / SQL / 中文）；III 平静期后浪涌 0.95–1.1×；7 种策略",
                res=f"中文天花板：反应式策略 {zh_osd:.2f}M vs 调度器 {zh_cm:.2f}M；1.0× 浪涌：调度器 {s10_cm:.1f}M vs OSD {s10_osd:.1f}M；"
                    "持续时长的分界只有几秒",
                verdict="调度器从不输；拉开差距的是天花板判别与平静期训满（SQL 饱和翻盘属边缘）", word="成立",
                status="有效（模拟）", build=build)


def e10(ctx):
    pols = [("fixed0", "fixed0：不训（基线）"), ("costmodel_v1_ungated", "costmodel v1（未门控，移植错误）"),
            ("costmodel", "costmodel（门控后）"), ("osd_idle", "osd_idle：空闲就训"), ("fixed1", "fixed1：一直全速训")]
    data = {p: jload(f"{RUNS}/realsys/m1_chat/{p}/summary.json") for p, _ in pols}
    base = {r["phase"]: r for r in data["fixed0"]}
    cm, v1 = {r["phase"]: r for r in data["costmodel"]}, {r["phase"]: r for r in data["costmodel_v1_ungated"]}
    ctx["e10"] = dict(v1=v1[2]["output_throughput"] / base[2]["output_throughput"] - 1,
                      cm=cm[2]["output_throughput"] / base[2]["output_throughput"] - 1, cm_tpot=(cm[0]["median_tpot_ms"], cm[1]["median_tpot_ms"]),
                      cm_trn=(cm[0]["trainer_tok_s"], cm[1]["trainer_tok_s"]))
    e = ctx["e10"]
    lead = (f"控制回路在真机上工作：训练速率准确跟随控制器给的占空比（决策延迟 1 秒）。\"SLO 内训练免费\"成立：C=4/64 时 TPOT "
            f"{e['cm_tpot'][0]:.1f}/{e['cm_tpot'][1]:.1f} ms（余量线 15 ms 以内），同时拿到约 {e['cm_trn'][0] / 1000:.0f}k token/s 的训练量。"
            f"v1 控制器在 C=256 全速训是移植错误（吞吐 {e['v1']:+.0%}）；门控后重跑，C=256 占空比降到 0，吞吐只差 {e['cm']:+.1%}。")

    def build(wb, ctx, e_):
        s = Sheet(wb, e_["sheet"], [30, 7, 8, 9, 11, 11, 10, 10, 10, 11])
        s.title_(f"{e_['id']} · {e_['name']}", lead)
        s.section("实验配置")
        s.kv([("要回答", "控制器能不能在真机上按规则调训练占空比？模拟器里\"SLO 内训练免费\"的规则在真机上成立吗？"),
              ("作业与日期", "10-05（悉尼）。15263716（fixed0、costmodel v1、osd_idle、fixed1）；15264390（门控后的 costmodel）。"),
              ("系统", "单卡 H100；vllm serve 投机开（原装头，k=3）；同卡训练进程 train_head.py --mode load，在缓存的对话 hidden states 上训，"
                     "--duty-file 每步重读占空比；controller.py 每秒读 /metrics（运行中请求数 = 并发，spec_decode 计数器差分 = 接受长度，"
                     "生成 token/s），按规则写占空比文件。这时还没有热替换，训练只付代价、兑现不了收益。"),
              ("负载剖面", "闭环并发 C = 4 → 64 → 256 → 4，每段 10 × C 条对话请求（约 40–240 秒）；五种策略用完全相同的剖面。"),
              ("策略", "fixed0 不训；fixed1 一直全速训；osd_idle 并发 <16 全速训；costmodel 按 E08 的规则（天花板判别 + SLO 内最大免费占空比 + "
                     "违约时按价值）；v1 把规则 3 简化成\"天花板超过盈亏线就训\"，门控后改为\"无热替换时违约不训\"。")])
        s.section("结果")
        rows = []
        for p, lab in pols:
            for r in data[p]:
                ph = r["phase"]
                key = f"{p}{ph}"
                row = {"k": lab if ph == 0 else "", "ph": f"{ph}", "c": r["C"], "dur": r["dur"], "thr": r["output_throughput"],
                       "t50": r["median_tpot_ms"], "t99": r["p99_tpot_ms"], "duty": r.get("trainer_duty_mean", 0),
                       "trn": (r.get("trainer_tok_s") or 0) / 1000}
                row["x"] = V(0.0, fmt=PCT) if p == "fixed0" else F(f"={{thr}}{{r}}/{{@fixed0{ph}|thr}}-1",
                                                                   r["output_throughput"] / base[ph]["output_throughput"] - 1, PCT)
                rows.append((key, row))
        s.table([("k", "策略", None), ("ph", "阶段", None), ("c", "并发", INT), ("dur", "时长 s", F1), ("thr", "吞吐 tok/s", INT),
                 ("x", "相对 fixed0\n同阶段", PCT), ("t50", "TPOT p50\nms", F1), ("t99", "TPOT p99\nms", F1), ("duty", "占空比\n均值", F2),
                 ("trn", "训练速率\nk tok/s", F1)], rows, caption="表 1　四段负载 × 五种策略",
                note="costmodel（门控后）阶段 2 的占空比均值 0.29 来自 bench 起止时并发尚低的几秒；控制器日志里并发 ≥200 的 89 个 tick 占空比全是 0。")
        s.section("怎么读")
        s.bullets(["控制回路工作：训练速率准确跟随占空比（0.70 → 约 40k、0.37 → 28k、0.19 → 15k token/s），决策延迟 1 秒。",
                   "免费占空比在真机成立：C=64、占空比约 0.7 时实测吞吐 −31%，模型 κ·d 预测 −39%。非饱和时吞吐损失不计入 goodput 是目标函数的定义，"
                   "不是免费午餐——它换来的是延迟（5 → 10 ms）。",
                   "v1 移植错误：对话天花板 2.37 在 256 路时 TPOT 仍 17.3 ms > SLO，训了也回不到线内；而且没有热替换，违约时训练永远兑现不了。"
                   "门控后真机行为 = \"SLO 内把训练当免费资源用满，饱和时让路\"。",
                   "osd_idle 的阈值在阶段切换时抖动（并发 64/256 阶段仍有 0.37/0.19 的占空比，来自 bench 起止时的低并发瞬间）。"])
        s.source("data/runs/realsys/m1_chat/{fixed0,costmodel_v1_ungated,costmodel,osd_idle,fixed1}/{summary.json,controller.jsonl,trainer.jsonl}；"
                 "code/specsched/{controller.py,realsys_run.py}；结论文档 2026-10-05_Rung1_调度器.md §4")
    return meta(ctx, "E10", short="真机占空比控制", name="真机占空比控制（控制器 v1 与门控后）",
                q="控制器在真机上能不能按规则调训练占空比？\"SLO 内训练免费\"成立吗？",
                cfg="8B 单卡，投机开 + 同卡训练进程 + controller.py 每秒写占空比；负载 4→64→256→4；fixed0 / costmodel / osd_idle / fixed1",
                res=f"训练速率跟随占空比（1 s 延迟）；C=4/64 免费拿到约 40k token/s 训练；v1 在 C=256 全速训 {e['v1']:+.0%}（移植错误），"
                    f"门控后 {e['cm']:+.1%}",
                verdict="通过（v1 的移植错误已修，门控后重跑确认）", word="通过", status="有效（含作废轮次）", build=build)


def hs_rows(group, modes, base_mode, extra=None):
    rows = []
    b = hotswap(group, base_mode)
    for mode, label, job in modes:
        h = hotswap(group, mode)
        if h is None:
            continue
        B = h["B"]
        row = {"k": V(label, bold=(mode == base_mode)), "job": job, "thr": B["output_throughput"],
               "t50": B["median_tpot_ms"], "t99": B["p99_tpot_ms"], "a0": h["acc_first"], "a1": h["acc_last"], "duty": h["duty"],
               "sw": swaps_text(h["swaps"]), "a1t": h["A1"].get("median_tpot_ms"), "a2t": h["A2"].get("median_tpot_ms"),
               "a2": h["A2"].get("output_throughput")}
        row["x"] = V(0.0, fmt=PCT) if mode == base_mode else F(f"={{thr}}{{r}}/{{@{base_mode}|thr}}-1",
                                                               B["output_throughput"] / b["B"]["output_throughput"] - 1, PCT)
        row["x2"] = V(0.0, fmt=PCT) if mode == base_mode else F(f"={{a2}}{{r}}/{{@{base_mode}|a2}}-1",
                                                                h["A2"]["output_throughput"] / b["A2"]["output_throughput"] - 1, PCT)
        if extra:
            row.update(extra.get(mode, {}))
        rows.append((mode, row))
    return rows


HS_COLS = [("k", "模式", None), ("job", "作业", None), ("thr", "B 段吞吐\ntok/s", INT), ("x", "相对\n不训练", PCT),
           ("t50", "B TPOT\np50 ms", F2), ("t99", "B TPOT\np99 ms", F2), ("a0", "接受长度\nB 开始", F2), ("a1", "接受长度\nB 末尾", F2),
           ("duty", "B 段训练\n占空比均值", F3), ("sw", "换头（距 B 开始的秒数；热载耗时）", None), ("a1t", "A1 TPOT\np50 ms", F2),
           ("a2t", "A2 TPOT\np50 ms", F2), ("a2", "A2 吞吐\ntok/s", INT), ("x2", "A2 相对\n不训练", PCT)]
HS_W = [30, 10, 10, 9, 9, 9, 9, 9, 10, 34, 9, 9, 9, 9]


def e11(ctx):
    st, ts, tn = hotswap("m1_code", "static"), hotswap("m1_code", "train_swap"), hotswap("m1_code", "train_noswap")
    g = ts["B"]["output_throughput"] / st["B"]["output_throughput"] - 1
    ctx["e11"] = dict(g=g, st=st, ts=ts, tn=tn)
    lead = (f"第一次端到端闭环：服务不停、同卡训练、热替换 4 次（每次约 0.7 秒，请求不中断）。B 段吞吐 {st['B']['output_throughput']:,.0f} → "
            f"{ts['B']['output_throughput']:,.0f} tok/s（{g:+.1%}），接受长度 {st['acc_last']:.2f} → {ts['acc_last']:.2f}；frontier 预测稳态 +26%、"
            f"时间线实测稳态 +24%。训了不换（train_noswap）的吞吐 = 不训，增益全部来自换上去的头；回到对话段 A2 的 TPOT 也从 "
            f"{st['A2']['median_tpot_ms']:.2f} 降到 {ts['A2']['median_tpot_ms']:.2f} ms（正迁移在线兑现）。")

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], HS_W)
        s.title_(f"{e['id']} · {e['name']}", lead)
        s.section("实验配置")
        s.kv([("要回答", "服务不停、同卡训练、把训好的头热替换进正在服务的引擎——端到端能不能兑现 frontier 预测的增益？"),
              ("作业与日期", "10-06（悉尼）。train_swap / train_noswap 15284127；static 15283467；代码 800 条 hidden states 缓存 15283295。"
                          "第一版（训练进程吃完缓存不退出）作废，见「作废与教训」。"),
              ("热替换", "给 vLLM 加了 reload_draft_weights（worker RPC，PR vllm-project/vllm#60139；0.27.1 回移植 apply_reload_draft_backport.py），"
                      "在 VLLM_SERVER_DEV_MODE=1 下用 POST /collective_rpc 触发，不中断请求。"),
              ("流量剖面", "A1：对话，C=16，120 条 → B：代码，C=256，9000 条（约 2.5–3 分钟，饱和）→ A2：对话，C=16，120 条。"
                        "提示池 = 各域 eval 的 300 条循环使用（训练用 train 切分、服务用 eval 切分，不相交）。"),
              ("训练", "B 段开始时启动同卡训练进程，占空比固定 0.5，在预先缓存的 800 条代码 train 请求的 hidden states 上学（与服务的请求同分布，"
                     "贪心输出等价）；每学完 100/200/400/800 条导出一个头，swap 线程立即热载；学完即退出。"),
              ("三种模式", "static：原装头，不训；train_swap：训 + 换头；train_noswap：训但不换（对照，分离训练本身的副作用）。")])
        s.section("结果")
        s.table(HS_COLS, hs_rows("m1_code", [("static", "static：原装头，不训", "15283467"),
                                             ("train_swap", "train_swap：训 + 热替换", "15284127"),
                                             ("train_noswap", "train_noswap：训但不换（对照）", "15284127")], "static"),
                caption="表 1　三种模式（B 段 = 代码、C=256、9000 条）",
                note="B 段 bench 吞吐包含开头约 25 秒的客户端准备时间（渲染 9000 条提示）和训练期（占空比 0.5 的前 25 秒）；"
                     "时间线上的稳态吞吐：static 43.9k、train_swap 54.7k（+24%）、train_noswap 44.5k（采样器绝对值偏高约 3 倍，只看相对，见「作废与教训」）。")
        s.section("怎么读")
        s.bullets(["train_noswap 与 static 一致，证明 +22–24% 全部来自换上去的头，不是训练过程的副作用；训练进程退出后代价归零。",
                   "frontier 预测稳态增益 1.17 ÷ 0.93 = +26%，实测 +24%——闭式 frontier 可以用来给调度器估价。",
                   "训练的代价只在前约 25 秒（占空比 0.5，吞吐约减半）。\"训完就停\"是调度器必须有的规则：第一版训练进程不退出时，稳态反而低于 static。",
                   "诚实边界：训练数据是预先缓存的 hidden states，不是从服务实时导出的；占空比固定 0.5，没接控制器——这两点在 E12–E13 解决。"])
        s.source("data/runs/hotswap/m1_code/{static,train_swap,train_noswap}/{phase*.json,events.jsonl,timeline.jsonl,trainer.jsonl}；"
                 "图 results/figures/hotswap_m1_code.png；结论文档 2026-10-05_Rung1_调度器.md §4.1")
    return meta(ctx, "E11", short="热替换闭环", name="热替换闭环（缓存版 hidden states，A→B→A）",
                q="服务不停、同卡训练、热替换草稿头，端到端能不能兑现 frontier 预测的增益？",
                cfg="8B 单卡；A1 对话 C=16 → B 代码 C=256（9000 条）→ A2 对话；B 段占空比 0.5 学 800 条缓存的 hidden states，每 100/200/400/800 条导出并热载",
                res=f"B 段吞吐 {g:+.1%}（{st['B']['output_throughput']:,.0f} → {ts['B']['output_throughput']:,.0f}），接受长度 → "
                    f"{ts['acc_last']:.2f}，热载 4 次各约 0.7 s；训了不换 = 不训；frontier 预测 +26%、实测 +24%",
                verdict="通过：热替换可行，增益全部来自换头", word="通过", status="有效（含作废轮次）", build=build)


def e12(ctx):
    a, b = hotswap("m1_code", "static"), hotswap("m1_code", "static_live_v1_pool300")
    d = {ph: b[ph]["output_throughput"] / a[ph]["output_throughput"] - 1 for ph in ("A1", "B", "A2")}
    tB = b["B"]["median_tpot_ms"] / a["B"]["median_tpot_ms"] - 1
    ctx["e12"] = dict(d=d, tB=tB, tap=b["tap"])
    lead = (f"vLLM 实时导出 EAGLE-3 训练特征（live tap）的开销约 1.5%：饱和段吞吐 {d['B']:+.1%}、TPOT p50 {tB:+.1%}；低负载对话段吞吐 "
            f"{d['A1']:+.1%} / {d['A2']:+.1%}。饱和段 3 分钟写约 68 GB、122 万条记录，只丢 1 步。可以常开。")

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], [22, 12, 12, 11, 12, 12, 11, 12, 12, 11])
        s.title_(f"{e['id']} · {e['name']}", lead)
        s.section("实验配置")
        s.kv([("要回答", "让 vLLM 在服务时顺手把训练草稿头要用的特征（目标模型 (2,18,33) 层的 hidden states + token）导出来，要付多少吞吐？"),
              ("作业与日期", "导出：10-08（悉尼）15342983 的 static_live；不导出：10-06 的 static（15283467）。"),
              ("实现", "code/specsched/livetap.py + apply_live_tap_backport.py（打进 0.27.1 的 gpu_model_runner）：EAGLE 分支里把已拼好的 "
                     "[T, 3H] bf16 aux 特征非阻塞拷到 pinned 内存池（3 槽），后台线程按请求切块写进 /dev/shm 环形文件；被拒绝的草稿位置截掉；"
                     "只有 TP rank 0 写；池满就丢整步并给该批请求发 GAP；任何异常自禁，不拖垮服务。开关 = 环境变量 VLLM_SPEC_LIVE_TAP"
                     "（容量 4 GB，抽样率 1）。前提：--no-enable-prefix-caching（前缀命中的位置没有 hidden state）。"),
              ("对比方式", "同一 A1/B/A2 剖面（同 E11）、同一个 300 条循环提示池、都不训练，只差导出开或关。"
                        "注意：15342983 里同时跑的 train_swap_live 因记忆泄漏作废，但 static_live 不训练、不受影响，可用于测开销。")])
        s.section("结果")
        rows = []
        for ph, lab in [("A1", "A1 对话 C=16"), ("B", "B 代码 C=256"), ("A2", "A2 对话 C=16")]:
            rows.append((ph, {"k": lab, "t0": a[ph]["output_throughput"], "t1": b[ph]["output_throughput"],
                              "dt": F("={t1}{r}/{t0}{r}-1", d[ph], PCT),
                              "p0": a[ph]["median_tpot_ms"], "p1": b[ph]["median_tpot_ms"],
                              "dp": F("={p1}{r}/{p0}{r}-1", b[ph]["median_tpot_ms"] / a[ph]["median_tpot_ms"] - 1, PCT),
                              "q0": a[ph]["p99_tpot_ms"], "q1": b[ph]["p99_tpot_ms"],
                              "dq": F("={q1}{r}/{q0}{r}-1", b[ph]["p99_tpot_ms"] / a[ph]["p99_tpot_ms"] - 1, PCT)}))
        s.table([("k", "阶段", None), ("t0", "不导出\ntok/s", INT), ("t1", "导出\ntok/s", INT), ("dt", "吞吐变化", PCT),
                 ("p0", "不导出\nTPOT p50", F2), ("p1", "导出\nTPOT p50", F2), ("dp", "p50 变化", PCT), ("q0", "不导出\nTPOT p99", F2),
                 ("q1", "导出\nTPOT p99", F2), ("dq", "p99 变化", PCT)], rows, caption="表 1　导出开销")
        rows = []
        for grp, mode, lab in [("m1_code", "static_live_v1_pool300", "代码，300 条循环池，只导出（本表）"),
                               ("m1_code", "static_live", "代码，唯一池，只导出（E13）"),
                               ("m1_zh", "static_live", "中文，唯一池，只导出（E14）"),
                               ("m2_code", "static_live", "MoE 代码，唯一池，只导出（E15）")]:
            t = hotswap(grp, mode)["tap"]
            rows.append((f"{grp}{mode}", {"k": lab, "st": t["steps"], "rec": t["records"], "mb": t["MB"] / 1024, "drop": t["dropped_steps"]}))
        s.table([("k", "运行", None), ("st", "导出步数", INT), ("rec", "记录数", INT), ("mb", "写入量 GB", F1), ("drop", "丢弃步数", INT)], rows,
                caption="表 2　导出量（整场累计：A1 + B + A2）",
                note="MoE 的隐藏维度更小（3H = 6144），同样的请求写入量约为 8B 的一半。写线程是瓶颈（池 3 槽、队列常满），后续可加槽或批量写。")
        s.section("怎么读")
        s.bullets(["导出代价约 1.5%，在跨实例噪声之上但很小，可以常开；它换来的是\"训练数据就是正在服务的请求\"。",
                   f"这次对比的两个运行相隔两天、不是同一作业，但同一剖面、同一提示池；跨实例噪声的变异系数 ≤{ctx['e05']['cv_max']:.1%}（E05），"
                   "差异主要来自导出本身。"])
        s.source("data/runs/hotswap/m1_code/{static,static_live_v1_pool300}/phase*.json、server.log（[live-tap] 统计行）；"
                 "code/specsched/{livetap.py,apply_live_tap_backport.py}；结论文档 2026-10-08_demo冲刺_实时导出与控制器.md §1")
    return meta(ctx, "E12", short="实时导出开销", name="实时导出（live tap）开销",
                q="从 vLLM 实时导出训练特征（live tap）要多少开销？",
                cfg="8B 单卡，同一剖面、同一提示池，导出 vs 不导出，都不训练",
                res=f"饱和段吞吐 {d['B']:+.1%}、TPOT {tB:+.1%}；3 分钟写约 68 GB、122 万条记录，只丢 1 步",
                verdict="通过：开销约 1.5%，可以常开", word="通过", status="有效", build=build)


CTL_VERSIONS = [
    ("v2", "第一版真管闭环：规则 0 停（天花板 2% 内 / 结构天花板）、1 frontier 开关（只记录）、2 SLO 内最大免费占空比、3 饱和时按价值；"
           "导出按域内消费数倍增（100/200/400/800），新头立即热载", "—"),
    ("v3/v4", "① regime 时钟从\"观察到流量\"起算（bench 先 tokenize 约 65 秒，域标签比流量早到）；② 饱和判别改为 waiting>0 或 running≥0.95×max_seqs"
              "（C<250 会抖）；③ 换域即 flush 训练缓冲 + 60 秒保鲜 + 导出最小间隔 5 秒；④ 用实测接受长度反查学习曲线位置（不用消费数）",
     "v2 换头 13 次、A2 里吃旧域数据连发换头"),
    ("v5", "规则 3 改为\"到目标的时间\"口径：占空比与数据到达率匹配，净价值 = 终态增益 × (视界 − 训练时长) − 训练期损失；视界先验 90 秒；"
           "换头后实测涨幅 <0.02 判收敛停", "v4 在 MoE 上只选 0.1 慢训 170 秒，换头太晚"),
    ("v5.1（最终）", "饱和段占空比上限 0.5（延迟护栏：TPOT 最多放大约 1.4 倍）", "v5 缓冲积到 56 万 token 时选占空比 1.0 爆训，p99 变差"),
    ("v6（开关版）", "--spec-switch 1：规则 1 真执行运行时投机开关（连续 3 个 tick 同向 + 10 秒最小间隔；换头时若关着先开 8 秒测新头）；"
                    "规则 3 的目标吞吐取\"开 / 关\"两者较优", "用户指出规则 1 只记录不执行"),
]


def e13(ctx):
    modes = [("static_live", "不训练（只导出）", "15343658"),
             ("static_sched", "静态表：并发 >64 关投机（不训）", "15345274"),
             ("train_swap_live", "固定策略：占空比 0.5，N800 后停", "15343658"),
             ("controller_live_v2", "控制器 v2", "15343658"),
             ("controller_live_v4", "控制器 v4", "15344614"),
             ("controller_live_v5", "控制器 v5", "15345623"),
             ("controller_live", "控制器 v5.1（最终）", "15345869"),
             ("controller_sw_live", "开关版控制器（worker 单侧开关；本轮没关投机）", "15347873")]
    rows = hs_rows("m1_code", modes, "static_live")
    h = {m: hotswap("m1_code", m) for m, _l, _j in modes}
    base = h["static_live"]["B"]["output_throughput"]
    gx = {m: h[m]["B"]["output_throughput"] / base - 1 for m in h}
    ctx["e13"] = dict(gx=gx, h=h)
    lead = (f"实时闭环（训练数据直接来自服务，B 段 9000 条提示全不重复）：固定策略 {gx['train_swap_live']:+.1%}；控制器 v5.1 "
            f"{gx['controller_live']:+.1%}，B 段 p99 {h['controller_live']['B']['p99_tpot_ms']:.1f} ms（固定策略 "
            f"{h['train_swap_live']['B']['p99_tpot_ms']:.1f}），训练占空比均值只有 {h['controller_live']['duty']:.2f}。"
            f"但饱和段光把投机关掉就 {gx['static_sched']:+.1%}、p99 回到 {h['static_sched']['B']['p99_tpot_ms']:.1f} ms——训好的头保持开只再多"
            f"几个点；训练真正值钱的地方在 SLO 内的对话段（A2 TPOT {h['static_live']['A2']['median_tpot_ms']:.2f} → "
            f"{h['controller_live']['A2']['median_tpot_ms']:.2f} ms）。")

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], HS_W)
        s.title_(f"{e['id']} · {e['name']}", lead)
        s.section("实验配置")
        s.kv([("要回答", "训练数据实时来自正在服务的请求、由控制器决定训多少和何时换头时，代码域能多出多少吞吐？控制器比固定策略好吗？"),
              ("作业与日期", "10-08（悉尼），见表中\"作业\"列。之前用 300 条循环池的一轮（15342983）因记忆泄漏作废。"),
              ("系统", "M1 单卡；vllm serve 投机开（原装头，k=3）+ live tap 实时导出（E12）；同卡训练进程 train_head.py --mode live：从环形缓冲"
                     "拼装完成的请求，CPU 侧后进先出缓冲（≤60 万 token），按占空比训练，按阈值或控制器指令导出；新头经 POST /collective_rpc 热载。"),
              ("流量剖面", "A1：对话 C=16，120 条 → B：代码 C=256，9000 条 → A2：对话 C=16，120 条。提示池用 make_bench_pool.py 从原始数据集"
                        "建不重复池（代码 16222 条、对话 16383 条，排除了 C2 流量里的 3800 条），B 段 9000 条全部唯一。"),
              ("控制器", "controller.py 每秒读 /metrics（并发、接受长度、TPOT 直方图、生成 token/s）、harness 写的域标签文件、训练器日志"
                      "（已消费请求数、缓冲 token 数），按规则 0–3 写占空比、发导出指令、热替换（规则见 E08；版本差异见表 2）。"
                      "\"不训练\"基线也开着导出，保证比较只差训练与换头。"),
              ("静态表", "vLLM 0.27 已有的按批大小的投机长度表 num_speculative_tokens_per_batch_size = [[1,64,3],[65,256,0]]：并发 ≤64 用 k=3，"
                      ">64 不投机；启动时定死，不能按头变。")])
        s.section("结果")
        s.table(HS_COLS, rows, caption="表 1　代码域（B 段 = 代码、C=256、9000 条唯一请求）",
                note="接受长度为 5 秒窗口的 /metrics 估计（B 开始 = 前 3 个窗口均值，末尾 = 最后 6 个窗口均值）；静态表在 B 段不投机，接受长度来自 B 段开头并发尚低的几秒。"
                     "换头一栏里负的秒数 = A1 对话段免费训出的头在 B 开始前就换上了。")
        s.table([("v", "版本", None), ("chg", "改了什么", None, 9), ("why", "为什么改", None, 4)],
                [(v, {"v": V(v, bold=True), "chg": c, "why": w}) for v, c, w in CTL_VERSIONS], caption="表 2　控制器版本差异")
        s.section("怎么读")
        s.bullets(["控制器 v5.1 用固定策略约 1/5 的训练量，拿到同量级（略高）的增益，p99 更好：A1 对话段免费训出的头在 B 一开始就顶上，"
                   "饱和后按数据到达率以 0.5 训两小段，换头实测 +0.12 / +0.08，之后净价值 ≤0 就停。",
                   "要诚实：饱和段出厂头开投机是负收益（gain 0.88），光关投机就 +13%；训好的头保持开只再多 3–5 个点，p50 更好但 p99 差（训练期尾部）。"
                   "所以叙事是\"两个杠杆：饱和段按 frontier 开关投机 + SLO 内免费训并换头\"，不是\"饱和段训练大赚\"。",
                   "训练最值钱的地方在 SLO 内的对话段：A2 TPOT 降约 27%、吞吐 +30% 以上（A1 免费训 + 代码训练的正迁移）；A1 为此付了 TPOT 5.7 → 6.5 ms（规则 2 允许到 15 ms）。",
                   "v2 少拿约 2.5 个点，来自表 2 里的四个毛病；v4/v5/v5.1 之间 1–2 个点的差别在噪声量级。",
                   "开关版控制器这一轮从没关投机：A1 免费训出的头让 B 一开始就 a≈2.47、gain 1.07 > 1，控制器判定保持开，结果与 v5.1 一致。"])
        s.source("data/runs/hotswap/m1_code/<模式>/{phase*.json,events.jsonl,timeline.jsonl,trainer.jsonl,controller.jsonl}；"
                 "图 results/figures/hotswap_m1_code.png；结论文档 2026-10-08_demo冲刺_实时导出与控制器.md §2.1、§2.4")
    return meta(ctx, "E13", short="实时闭环-代码", name="实时闭环 + 控制器：Qwen3-8B 代码域",
                q="训练数据实时来自服务、控制器真管闭环时，代码域能多出多少吞吐？控制器比固定策略好吗？",
                cfg="8B 单卡，唯一提示池；A1 对话 C=16 → B 代码 C=256（9000 条不重复）→ A2；固定 0.5 / 控制器 v2–v5.1 / 静态表关投机 / 开关版",
                res=f"固定 {gx['train_swap_live']:+.1%}；控制器 v5.1 {gx['controller_live']:+.1%}（占空比 {h['controller_live']['duty']:.2f}、"
                    f"p99 {h['controller_live']['B']['p99_tpot_ms']:.1f} ms）；饱和段关投机 {gx['static_sched']:+.1%}；A2 对话 TPOT "
                    f"{h['static_live']['A2']['median_tpot_ms']:.2f}→{h['controller_live']['A2']['median_tpot_ms']:.2f} ms",
                verdict="通过：控制器用约 1/5 的训练量略超固定策略；饱和段最大的杠杆是开关投机", word="通过",
                status="有效（含作废轮次）", build=build)


def e14(ctx):
    modes = [("static_live", "不训练（只导出）", "15343659"),
             ("static_sched", "静态表：并发 >64 关投机（不训）", "15344918"),
             ("train_swap_live", "固定策略：占空比 0.5，N800 后停", "15343659"),
             ("controller_live_v4", "控制器 v4（spec_switchable=0）", "15343659"),
             ("controller_live_v5", "控制器 v5", "15345624"),
             ("controller_live", "控制器 v5.1", "15345940"),
             ("controller_sw_live", "开关版控制器（调度器级运行时开关）", "15348479")]
    rows = hs_rows("m1_zh", modes, "static_live")
    h = {m: hotswap("m1_zh", m) for m, _l, _j in modes}
    base = h["static_live"]["B"]["output_throughput"]
    gx = {m: h[m]["B"]["output_throughput"] / base - 1 for m in h}
    ctx["e14"] = dict(gx=gx, h=h)
    lead = (f"中文推翻了\"结构天花板 → 拒训\"的剧本：vLLM 运行时关不掉投机，投机既然一直开着，把接受长度从 1.23 拉到 1.42 就是实打实的收益——"
            f"固定策略 {gx['train_swap_live']:+.1%}、控制器 v5.1 {gx['controller_live']:+.1%}。但真正的大杠杆是关投机：静态表饱和段关投机 "
            f"{gx['static_sched']:+.1%}、p99 回到 {h['static_sched']['B']['p99_tpot_ms']:.1f} ms；带运行时开关的控制器 {gx['controller_sw_live']:+.1%}，"
            f"同时判定天花板不训（占空比 {h['controller_sw_live']['duty']:.3f}）。三条判别第一次在一条时间线上全部兑现。")

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], HS_W)
        s.title_(f"{e['id']} · {e['name']}", lead)
        s.section("实验配置")
        s.kv([("要回答", "中文的草稿词表只覆盖 42.6% 的输出 token（结构天花板 1.45），饱和时该训头还是该关投机？"),
              ("作业与日期", "10-08（悉尼），见表中\"作业\"列。"),
              ("设置", "同 E13，只把 B 段换成中文：提示池 = 中文不重复池（44776 条），B 段 C=256、9000 条全部唯一。"
                     "原计划的剧本：中文天花板 1.45 在 C=256 的 frontier 上只有 0.65×，\"训了也赢不了，控制器拒训\"。"),
              ("控制器假设", "--spec-switchable 0：引擎不能运行时关投机（vLLM 0.27 的现实）→ 走价值规则照训；"
                         "开关版（v6，--spec-switch 1）：调度器级运行时开关可用（E16）→ 先关投机，再按结构天花板判定不训。")])
        s.section("结果")
        s.table(HS_COLS, rows, caption="表 1　中文（B 段 = 中文、C=256、9000 条唯一请求）",
                note="开关版控制器 B 段大部分时间投机关着，没有接受长度读数（空格）；它的导出在投机关闭时改从目标模型自己的输出继续，丢弃步数较多（480 步）。")
        s.section("怎么读")
        s.bullets(["\"拒训\"只在引擎能运行时关投机时才对：那时正确的动作是关投机（frontier：无投机比出厂头投机快 1.8 倍），训练才是白费。"
                   "所以结构天花板判别必须以引擎能力为条件（--spec-switchable）。",
                   "对中文这类词表受限的域，\"关投机\"是比\"训头\"大一个数量级的杠杆（+85–88% vs +10–13%）。",
                   "给腾讯的一个具体需求：推理引擎要有运行时投机开关（或按负载的投机 token 数调度表），否则这类域在饱和时白白慢约 1.8 倍。",
                   "v5 在中文上用固定策略约 1/11 的训练份额超过固定策略；v5.1 与 v5 的差别（±2%）在噪声内。"])
        s.source("data/runs/hotswap/m1_zh/<模式>/…；图 results/figures/hotswap_m1_zh.png；结论文档 2026-10-08_demo冲刺_实时导出与控制器.md §2.1b、§2.5")
    return meta(ctx, "E14", short="实时闭环-中文", name="实时闭环 + 控制器：Qwen3-8B 中文（词表天花板）",
                q="中文（草稿词表只覆盖 42.6%）在饱和时该训头还是该关投机？",
                cfg="同 E13，B 段换成中文；固定 0.5 / 控制器 v4–v5.1 / 静态表关投机 / 带运行时开关的控制器",
                res=f"照训：固定 {gx['train_swap_live']:+.1%}、控制器 v5.1 {gx['controller_live']:+.1%}；关投机（静态表）{gx['static_sched']:+.1%}、"
                    f"p99 回到 {h['static_sched']['B']['p99_tpot_ms']:.1f} ms；开关版控制器 {gx['controller_sw_live']:+.1%} 且判定不训",
                verdict="推翻\"天花板 → 拒训\"剧本：引擎关不掉投机时照训也赚；能关时应先关投机", word="推翻原剧本",
                status="有效", build=build)


def e15(ctx):
    df, none = analyze_c2.load_domain(f"{RUNS}/c2/m2/code")
    df = df.sort_values("N").reset_index(drop=True)
    Cs = [1, 4, 16, 64, 128, 256]
    heads = ["S1_N0", "S1_N100", "S1_N400", "S1_N3200"]
    fr = {h: frontier("m2_code", h) for h in ["S0"] + heads}
    sp = {h: {C: fr[h][C]["output_throughput"] / fr["S0"][C]["output_throughput"] for C in Cs} for h in heads}
    fits = jload(f"{ROOT}/code/specsched/sched/model_m2.json")["frontier"]["code"]["fit"]
    modes = [("static_live", "不训练（只导出）", "15343660"), ("train_swap_live", "固定策略：占空比 0.5，N800 后停", "15343660"),
             ("controller_live_v4", "控制器 v4", "15345431"), ("controller_live", "控制器 v5.1", "15345870")]
    h = {m: hotswap("m2_code", m) for m, _l, _j in modes}
    gx = {m: h[m]["B"]["output_throughput"] / h["static_live"]["B"]["output_throughput"] - 1 for m in h}
    a0, a3 = float(df[df.N == 0].acc_len.iloc[0]), float(df[df.N == 3200].acc_len.iloc[0])
    ctx["e15"] = dict(gx=gx, a0=a0, a3=a3, sp=sp, h=h)
    sat = [sp[hh][256] for hh in heads]
    lead = (f"MoE（Qwen3-30B-A3B）上同样成立，但形状不同：出厂头起点高（代码 {a0:.2f}），在线 3200 条再 +{a3 - a0:.2f}（{a3:.2f}）；"
            f"投机在所有负载都是大正收益（饱和 C=256 仍 {min(sat):.2f}–{max(sat):.2f}×），永远不该关。实时闭环：固定策略 "
            f"{gx['train_swap_live']:+.1%}，控制器 v5.1 {gx['controller_live']:+.1%}（训练只用约 1/6）；头小，热载约 0.32 秒。")

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], HS_W)
        s.title_(f"{e['id']} · {e['name']}", lead)
        s.section("实验配置")
        s.kv([("要回答", "目标模型换成 MoE，这套\"在线训头 + 调度\"还成立吗？投机的负载曲线和 dense 有什么不同？"),
              ("作业与日期", "10-08（悉尼）。流量 + 缓存 15342982；C2 代码 15342984；frontier 15343170 + 15345395；闭环 15343660（不训练 / 固定）、"
                          "15345431（控制器 v4）、15345870（控制器 v5.1）。"),
              ("模型", "M2 Qwen3-30B-A3B + AngelSlim/Qwen3-a3B_eagle3，aux 层 (2,24,45)，k=3，贪心；训练进程预留 5 GB（闭环 gpu_memory_utilization 0.8375）。"),
              ("C2 / frontier", "同 E04 / E06 的做法：M2 生成代码 / SQL / 中文 / 对话流量（各 3800 条），代码域单遍在线更新 N=100–3200，"
                             "评测代码 + 对话（迁移）；frontier 头 = 不投机 / N0 / N100 / N400 / N3200，C=1–256。"),
              ("闭环", "同 E13（唯一池，A1 对话 → B 代码 C=256 9000 条 → A2）；控制器用 model_m2.json（κ≈0.50–0.55、训练速率 106k token/s/占空比）。")])
        s.section("结果")
        Ns = [0, 100, 200, 400, 800, 1600, 3200]

        def at(N, col):
            return float(df[df.N == N][col].iloc[0])
        rows = [("acc", {"k": "代码域接受长度", **{f"n{N}": at(N, "acc_len") for N in Ns}}),
                ("chat", {"k": "对话域接受长度（迁移）", **{f"n{N}": at(N, "acc_len_chat") for N in Ns}}),
                ("g", {"k": "饱和 GPU 毫秒/请求", **{f"n{N}": V(at(N, "sat_gpu_s_per_req") * 1000, fmt=F2) for N in Ns}}),
                ("id", {"k": "逐 token 一致率", **{f"n{N}": V(at(N, "identical"), fmt=PCTP) for N in Ns}})]
        s.table([("k", "指标", None)] + [(f"n{N}", f"N{N}", F2) for N in Ns], rows, caption="表 1　MoE 代码域在线更新（C2）",
                note=f"单遍 3200 条只含头的 GPU 时间 {at(3200, 'C_train_gpu_s'):.1f} 秒；不投机的饱和 GPU 毫秒/请求 = "
                     f"{none['saturation']['gpu_s_per_req'] * 1000:.2f}。")
        hn = {"S0": "不投机", "S1_N0": "N0（原装头）", "S1_N100": "N100", "S1_N400": "N400", "S1_N3200": "N3200"}
        pr = s.table([("h", "头", None)] + [(f"c{C}", f"C={C}", INT) for C in Cs],
                     [(hh, {"h": V(hn[hh], bold=(hh == "S0")), **{f"c{C}": fr[hh][C]["output_throughput"] for C in Cs}}) for hh in ["S0"] + heads],
                     caption="表 2　frontier：吞吐 tok/s（代码域）")
        s.table([("h", "头", None), ("a", "接受长度\n（均值）", F2)] + [(f"c{C}", f"C={C}", X2) for C in Cs],
                [(hh, {"h": hn[hh], "a": float(np.mean([fr[hh][C]["acc_len"] for C in Cs])),
                       **{f"c{C}": F(f"={pr[(hh, f'c{C}')]}/{pr[('S0', f'c{C}')]}", sp[hh][C], X2) for C in Cs}}) for hh in heads],
                caption="表 3　frontier：相对不投机的倍数（公式）")
        s.table([("k", "并发", None), ("al", "α(C)", F3), ("be", "β(C)", F3), ("err", "最大残差", F3),
                 ("ast", "盈亏接受长度\na*", F2)],
                [(C, {"k": f"C={C}", "al": fits[str(C)]["alpha"], "be": fits[str(C)]["beta"], "err": fits[str(C)]["max_abs_err"],
                      "ast": F("=(1-{be}{r})/{al}{r}", (1 - fits[str(C)]["beta"]) / fits[str(C)]["alpha"], F2)}) for C in Cs],
                caption="表 4　线性拟合 gain = α(C)·a + β(C)（sched/model_m2.json）")
        s.table(HS_COLS, hs_rows("m2_code", modes, "static_live"), caption="表 5　MoE 实时闭环（B 段 = 代码、C=256、9000 条唯一请求）")
        s.section("怎么读")
        s.bullets(["MoE 与 dense 相反：投机在所有负载都是大正收益（MoESD 的预测），控制器的规则 1 永远保持开；在线适应在此基础上再加 0.1–0.26×。",
                   "闭环增益比 dense 小（+5–8% vs +17–18%），因为出厂头起点更高、可学的空间更小；frontier 上饱和段 +13% 与闭环 +8% 同向（闭环含训练期代价）。",
                   "控制器比固定策略少约 3 个点：两者学到同样的 2.94，但控制器的导出按域内消费数 100/200/400 倍增，第三次换头在 +130 秒（固定策略 +65 秒就换完），"
                   "而 B 段只有 165 秒，剩给\"赚回来\"的时间短。真实流量切换以分钟计时差距会消失；要压时间可把首个导出阈值从 100 降到 50。",
                   "对话段 MoE 没有标定曲线 → 控制器按\"未知域不训\"处理（正确且诚实）。"])
        s.source("data/runs/c2/m2/code/*；data/runs/frontier/m2_code/*；data/runs/hotswap/m2_code/*；code/specsched/sched/model_m2.json；"
                 "图 results/figures/{c2_m2,hotswap_m2_code}.png；结论文档 2026-10-08_demo冲刺_实时导出与控制器.md §2.1c–2.3")
    return meta(ctx, "E15", short="MoE", name="MoE（Qwen3-30B-A3B）：学习曲线、frontier、实时闭环",
                q="MoE 目标模型上这套方法还成立吗？",
                cfg="Qwen3-30B-A3B：代码域 C2 学习曲线 + frontier（C=1–256）+ 实时闭环（固定 / 控制器 v4 / v5.1）",
                res=f"代码 {a0:.2f}→{a3:.2f}；投机所有负载都大正收益（饱和仍 {min(sat):.2f}–{max(sat):.2f}×）；闭环 {gx['train_swap_live']:+.1%}（固定）"
                    f" / {gx['controller_live']:+.1%}（控制器 v5.1，训练约 1/6）；热载 0.32 s",
                verdict="成立：MoE 永远不该关投机，训头仍有 5–8%", word="成立", status="有效", build=build)


def e16(ctx):
    runs = [("m1_code_v2", "代码（15348478）"), ("m1_code", "代码重跑（15349223，hook 显式缓存 next tokens 后）"),
            ("m1_zh", "中文（15348478）")]
    data = {k: jload(f"{RUNS}/switch/{k}/summary.json") for k, _ in runs}
    g = {k: data[k][1]["output_throughput"] / data[k][0]["output_throughput"] - 1 for k in data}
    ctx["e16"] = g
    c, z = data["m1_code_v2"], data["m1_zh"]
    lead = (f"给 vLLM 加了调度器级的运行时投机开关：代码 C=256 开 → 关 {g['m1_code_v2']:+.1%}、TPOT p99 {c[0]['p99_tpot_ms']:.1f} → "
            f"{c[1]['p99_tpot_ms']:.1f} ms；中文 {g['m1_zh']:+.0%}、p99 {z[0]['p99_tpot_ms']:.1f} → {z[1]['p99_tpot_ms']:.1f} ms；切换 ≤10 ms，"
            "再开完全恢复。只在 worker 一侧关的第一版是错的（吞吐 −46%）。")

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], [44, 8, 8, 11, 10, 10, 10, 10, 12, 10])
        s.title_(f"{e['id']} · {e['name']}", lead)
        s.section("实验配置")
        s.kv([("要回答", "vLLM 0.27 运行时没有投机开关（控制器规则 1 只能记录）。能不能加一个？开 / 关在饱和时值多少？"),
              ("作业与日期", "10-08（悉尼）。调度器级：15348478（代码 + 中文）、15349223（代码重跑）。worker 单侧版：15347807（引擎断言崩溃，取消）、"
                          "15347872（关后 −46%，取消）。"),
              ("实现", "apply_spec_switch_backport.py：runner 加 set_speculation(enabled)；调度器 spec_enabled=False 时不再调度草稿槽"
                     "（num_spec_tokens_to_schedule=0、不 pad），worker 不出草稿；EngineCore.collective_rpc 拦截 set_speculation 把两半同步切；"
                     "drafter 权重留在显存，打开即下一步生效；经已有的 POST /collective_rpc 调用。"),
              ("测试", "M1 单卡，vllm serve 投机开（原装头），同一个服务里 开 → 关 → 开 三段，每段 2000 条请求，C=256，唯一提示池（代码 / 中文）。")])
        s.section("结果")
        rows = []
        for k, lab in runs:
            for r in data[k]:
                ph = r["phase"]
                rows.append((f"{k}{ph}", {"k": lab if ph == 0 else "", "ph": ["开", "关", "再开"][ph], "thr": r["output_throughput"],
                                          "x": V(0.0, fmt=PCT) if ph == 0 else F(f"={{thr}}{{r}}/{{@{k}0|thr}}-1",
                                                                                 r["output_throughput"] / data[k][0]["output_throughput"] - 1, PCT),
                                          "t50": r["median_tpot_ms"], "t99": r["p99_tpot_ms"], "acc": r.get("acc_len"),
                                          "dr": r.get("drafts"), "sw": (r["switch_s"] * 1000) if r.get("switch_s") is not None else None}))
        s.table([("k", "运行", None), ("ph", "投机", None), ("thr", "吞吐 tok/s", INT), ("x", "相对\"开\"", PCT), ("t50", "TPOT p50\nms", F2),
                 ("t99", "TPOT p99\nms", F2), ("acc", "接受长度", F2), ("dr", "草稿次数", INT), ("sw", "切换耗时\nms", F1)], rows,
                caption="表 1　同一服务里 开 → 关 → 开（C=256，每段 2000 条）")
        s.table([("k", "版本", None), ("what", "做法", None, 3), ("res", "结果", None, 5), ("v", "判定", None)], [
            ("w", {"k": "worker 单侧版（15347872）", "what": "只让 worker 不出草稿", "res": "关掉后吞吐 14168 → 7633（−46%），接受长度 1.00、草稿计数反而翻倍",
                   "v": V("作废", status="作废")}),
            ("s", {"k": "调度器级（15348478 / 15349223）", "what": "调度器与 worker 两半同步切", "res": "见表 1", "v": V("通过", status="通过")})],
            caption="表 2　为什么必须两半一起切",
            note="worker 单侧版的数字转录自结论文档 §2.5（作业被取消，结果未镜像到本地）。padded 模式下调度器仍给每个请求预留 k 个草稿槽（占位 −1），"
                 "worker 不出草稿时槽里是垃圾、全被拒绝，等于白付 k+1 倍验证。")
        s.section("怎么读")
        s.bullets(["饱和时关投机：代码 +10%、中文 +85%，而且 p99 回到 SLO 以内。这就是控制器规则 1 的价值，也是给引擎方（腾讯）的具体需求。",
                   "再开后吞吐和接受长度完全恢复，切换在 10 毫秒内——可以按秒级负载变化来回切。",
                   "同一改动已进 vLLM 主干分支并通过 e2e（关后不出草稿、输出逐 token 不变、开后恢复），见 E18。"])
        s.source("data/runs/switch/{m1_code_v2,m1_code,m1_zh}/{summary.json,phase*.json}；code/specsched/{apply_spec_switch_backport.py,spec_switch_check.py}；"
                 "结论文档 2026-10-08_demo冲刺_实时导出与控制器.md §2.5")
    return meta(ctx, "E16", short="投机开关", name="运行时投机开关（开 → 关 → 开）",
                q="vLLM 能不能在运行时开关投机？开 / 关在饱和时值多少？",
                cfg="8B 单卡，同一服务 开 → 关 → 开，C=256，每段 2000 条；调度器 + worker 两半一起切",
                res=f"代码关投机 {g['m1_code_v2']:+.1%}（重跑 {g['m1_code']:+.1%}）、p99 回 SLO 内；中文 {g['m1_zh']:+.0%}；切换 ≤10 ms，再开完全恢复；"
                    "worker 单侧版 −46%（作废）",
                verdict="通过", word="通过", status="有效（含作废轮次）", build=build)


def e17(ctx):
    tp = {n: jload(f"{RUNS}/tp/m1_tp{n}_code/summary.json") for n in (2, 4)}
    probe = jload(f"{RUNS}/tp/probe_tp2.json")
    g = {n: tp[n][1]["output_throughput"] / tp[n][0]["output_throughput"] - 1 for n in tp}
    ctx["e17"] = dict(g=g, tp=tp)
    lead = (f"多卡张量并行下热替换功能正确：TP=2 出厂头 → 适应头 {g[2]:+.1%}（热载 {tp[2][1]['swap_s']:.2f} 秒），TP=4 {g[4]:+.1%}"
            f"（{tp[4][1]['swap_s']:.2f} 秒），换回出厂头后吞吐和接受长度恢复。探针确认重载本身精确（无操作重载 32/32 逐 token 一致）；"
            "TP 下\"换头后逐 token 不变\"取决于内核是否 batch-invariant，不是重载能保证的。")

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], [22, 16, 11, 10, 10, 10, 10, 10, 30])
        s.title_(f"{e['id']} · {e['name']}", lead)
        s.section("实验配置")
        s.kv([("要回答", "多卡张量并行（TP）时，reload_draft_weights 还能正确地把新头换进每个 rank 吗？会不会改变输出？"),
              ("作业与日期", "15343033（10-09 悉尼，gpu 分区 4×H100：TP=2 e2e + TP=2 服务）；15363115（10-10 悉尼，TP=4 服务）；15391476（10-10 悉尼，TP=2 探针）。"
                          "OSC debug 分区禁止多卡（QOSMaxGRESPerNode），改投 gpu 分区，每次排队约 10 小时。"),
              ("服务测试", "vllm serve -tp N，代码 eval 提示，C=64；三段：出厂头 → 热载 E04 的代码适应头 N3200 → 换回出厂头，每段测吞吐与接受长度。"),
              ("精确性探针", "tp_exactness_probe.py，TP=2，32 条代码提示、贪心 128 token：同头重跑 / 无操作重载（同一检查点）/ 清零 lm_head 的头重载 / "
                          "恢复出厂头，各自与基线逐 token 比较。")])
        s.section("结果")
        rows = []
        for n in (2, 4):
            for r in tp[n]:
                ph = r["phase"]
                rows.append((f"tp{n}{ph}", {"k": f"TP={n}" if ph == 0 else "", "h": {"stock": "出厂头", "adapted": "代码适应头 N3200",
                                                                                     "stock_again": "换回出厂头"}[r["head"]],
                                            "thr": r["output_throughput"],
                                            "x": V(0.0, fmt=PCT) if ph == 0 else F(f"={{thr}}{{r}}/{{@tp{n}0|thr}}-1",
                                                                                   r["output_throughput"] / tp[n][0]["output_throughput"] - 1, PCT),
                                            "t50": r["median_tpot_ms"], "t99": r["p99_tpot_ms"], "acc": r["acc_len"],
                                            "sw": r["swap_s"]}))
        s.table([("k", "并行度", None), ("h", "当前的头", None), ("thr", "吞吐 tok/s", INT), ("x", "相对出厂头", PCT),
                 ("t50", "TPOT p50\nms", F2), ("t99", "TPOT p99\nms", F2), ("acc", "接受长度", F2), ("sw", "换头耗时 s", F2)], rows,
                caption="表 1　TP 服务热替换（代码，C=64）")
        prow = []
        for k, lab, why in [("rerun_same_head", "同头重跑", "引擎本身可复现"), ("after_noop_reload", "无操作重载（同一检查点）", "重载本身是精确的"),
                            ("after_zeroed_reload", "清零 lm_head 的头重载", "草稿全被拒、批形状变了，TP 下验证数值微变，贪心平局翻转"),
                            ("after_restore", "恢复出厂头", "完全恢复")]:
            p = probe[k]
            prow.append((k, {"k": lab, "i": p["identical"], "n": p["total"],
                             "f": F("={i}{r}/{n}{r}", p["identical"] / p["total"], PCTP),
                             "d": "、".join(str(x) for x in p["first_divergence"]) or "—", "w": why}))
        s.table([("k", "检查", None), ("i", "逐 token 一致", INT), ("n", "总条数", INT), ("f", "一致比例", PCTP), ("d", "首次分叉位置", None),
                 ("w", "说明", None, 4)], prow, caption="表 2　TP=2 逐 token 精确性探针（32 条代码提示，贪心 128 token）")
        s.section("怎么读")
        s.bullets(["功能上，TP=2/4 每个 rank 都正确换上了新头（rank 全部返回），增益与单卡一致（+25–26%）。",
                   "TP=2 的 standalone e2e 里\"逐 token 一致\"断言失败（TP=1 通过）：TP 下验证数值随批形状、归约顺序略变，贪心可能翻 token。"
                   "探针把两件事分开了——重载精确，翻 token 是 vLLM TP 内核不保证 batch-invariance。PR 描述已补这条证据。",
                   "TP=4 的 e2e 没跑：TP 环境变量没传进作业脚本（只跑了 TP=2），TP=4 只有服务测试。"])
        s.source("data/runs/tp/{m1_tp2_code,m1_tp4_code}/summary.json、data/runs/tp/probe_tp2.json；code/specsched/{tp_serve_reload.py,tp_exactness_probe.py}；"
                 "结论文档 2026-10-08_demo冲刺_实时导出与控制器.md §3")
    return meta(ctx, "E17", short="多卡TP", name="多卡张量并行热替换与精确性探针",
                q="多卡张量并行下热替换还正确吗？输出会不会变？",
                cfg="TP=2 / TP=4（gpu 分区），代码 eval C=64：出厂头 → 适应头 N3200 → 换回；TP=2 逐 token 探针 32 条",
                res=f"TP=2 {g[2]:+.1%}（{tp[2][1]['swap_s']:.2f} s）、TP=4 {g[4]:+.1%}（{tp[4][1]['swap_s']:.2f} s），换回恢复；"
                    f"无操作重载 {probe['after_noop_reload']['identical']}/{probe['after_noop_reload']['total']} 逐 token 一致",
                verdict="通过：重载精确；TP 下逐 token 一致取决于内核 batch-invariance", word="通过", status="有效", build=build)


def e18(ctx):
    tests = [
        ("reload_draft_weights e2e", "0.27.1 回移植", "热载新头后服务不断、输出与直接加载该头一致", "通过", "15283466", "10-06"),
        ("reload_draft_weights e2e", "vLLM 主干（PR #60139）", "同上；主干与 0.27.1 都通过", "通过", "15283773、15283987", "10-06"),
        ("reload_draft_weights e2e，TP=2", "0.27.1 回移植", "功能正确；\"逐 token 一致\"断言失败（TP 内核非 batch-invariant，见 E17 探针）",
         "部分成立", "15343033", "10-09"),
        ("live tap 单元测试", "主干分支 feat/spec-live-tap", "环形缓冲回绕 / 套圈、from_env、GPU 上含拒绝位的 record", "通过", "15349325（ddab4c4）", "10-08"),
        ("live tap e2e", "主干分支 feat/spec-live-tap", "padded / unpadded 两种模式，从环里重建全部 token 序列须与 prompt + 输出一致", "通过",
         "15349325（ddab4c4）", "10-08"),
        ("set_speculation e2e", "主干分支 feat/spec-live-tap", "关后不出草稿、输出逐 token 不变、开后恢复", "通过", "15349325（ddab4c4）", "10-08"),
        ("中间调试轮", "主干分支", "单测里 queue.join() 没有 task_done 永久挂起（15345163）；force-push 后 OSC 树 fetch 静默拒绝非快进，两轮\"失败\"其实跑的是旧代码",
         "参考", "15343744 等 16 个", "10-08"),
    ]

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], [28, 24, 46, 12, 18, 10])
        s.title_(f"{e['id']} · {e['name']}", "所有改动在两套代码路径上都通过了测试：OSC 上实验用的 vLLM 0.27.1 回移植，以及 vLLM 主干分支"
                                               "（热替换 PR #60139；实时导出与运行时开关在 feat/spec-live-tap，最终验证 commit ddab4c4：单测 3 passed、"
                                               "导出 e2e 2 passed、开关 e2e 1 passed）。")
        s.section("实验配置")
        s.kv([("要回答", "实验里用的三处 vLLM 改动（热替换、实时导出、运行时开关）是否正确，能否提交上游？"),
              ("代码路径", "① OSC 实验环境：vLLM 0.27.1（bursar 的 bsr312 环境）+ apply_*_backport.py 锚点幂等打补丁；"
                        "② vLLM 主干：PR #60139（reload_draft_weights，issue #60138）；分支 ANTI-Tony:feat/spec-live-tap（live tap + set_speculation，"
                        "未开 PR，issue #60622）。"),
              ("测试模型", "Qwen3-8B + AngelSlim/Qwen3-8B_eagle3（单卡 H100；TP 测试在 gpu 分区）。")])
        s.section("结果")
        s.table([("t", "测试", None), ("p", "代码路径", None), ("w", "检查内容", None), ("v", "结果", None), ("j", "作业", None), ("d", "日期", None)],
                [(str(i), {"t": t, "p": p, "w": w, "v": V(v, status=v), "j": j, "d": d}) for i, (t, p, w, v, j, d) in enumerate(tests)],
                caption="表 1　功能测试")
        s.section("怎么读")
        s.bullets(["调试中确认的引擎行为：请求 ID 带引擎后缀；FINISH 在下一次调度时才写；最后一步接受的草稿可以越过 EOS / max_tokens"
                   "（输出被截断但位置已计算，导出多 ≤k+1 个 token，对训练无害）；tap 挂在 V1 runner（VLLM_USE_V2_MODEL_RUNNER=0），V2 移植待做。",
                   "上游状态（10-16）：PR #60139 被打 needs-rebase、还没有人评审；feat/spec-live-tap 按 AGENTS.md 要先开 issue，建议拆成两个 PR（tap / 开关）。"])
        s.source("code/vllm_pr/{reload_draft.patch,spec_live_tap.patch,vllm_pr.md,vllm_tap_pr.md}；code/specsched/test_*.py；"
                 "结论文档 2026-10-08_demo冲刺_实时导出与控制器.md §3b")
    return meta(ctx, "E18", short="功能测试", name="vLLM 改动的功能测试（回移植 + 主干分支）",
                q="热替换、实时导出、运行时开关三处改动，在 0.27.1 回移植和 vLLM 主干上是否都通过测试？",
                cfg="reload_draft_weights（PR #60139）、live tap、set_speculation 的单测与 e2e；单卡 + TP=2",
                res="0.27.1 与主干的 reload e2e 通过；主干 ddab4c4：单测 3 passed、导出 e2e 2 passed、开关 e2e 1 passed；TP=2 逐 token 断言属内核性质",
                verdict="通过", word="通过", status="有效", build=build)



ABL_DIR = os.environ.get("ABL_DIR", f"{ROOT}/results/abl_all" if os.path.exists(f"{ROOT}/results/abl_all/abl_runs.csv") else f"{ROOT}/results/abl")   # 81-run main table (with TIDE) when present
STRAT_ZH = {"vanilla": "现网静态投机（参考）", "nospec": "纯解码（无草稿器）", "static": "静态投机（同基础设施）", "switch": "仅运行时开关",
            "simple": "简单规则联合控制", "tide": "TIDE 式触发训练", "fixed": "固定训练（全程 0.5）", "fixed800": "固定训练（已知切换点，到 N800 停）", "v6": "完整 v6 控制器",
            "tide_r800e03": "TIDE 式（每轮 800、阈值 0.3）"}
STRAT_DESC = {
    "vanilla": "今天的部署：出厂头、投机常开、不开导出、显存全给 KV（util 0.90）。只有它和别的策略引擎配置不同，用作\"相对现网\"的参考。",
    "nospec": "服务启动时不加载草稿器，纯自回归解码；KV 预算与其它策略相同（util 0.90 − 训练预留）。",
    "static": "出厂头、投机常开、实时导出（tap）开着但没人训，训练显存预留着：它和 v6 只差\"策略\"，是消融的主对照。",
    "switch": "控制器只执行规则 1（按 frontier gain(a,C)>1 开/关投机），从不训练。",
    "simple": "两条阈值规则：饱和（排队>0 或 running ≥0.95×256）就关投机，否则开；TPOT 在 SLO 内且不饱和时按 0.5 占空比训（OSD 的\"不在高峰才训\"），"
              "饱和不训；换头按 100/200/400… 倍增。没有 frontier、没有价值计算。",
    "tide": "冻结后补的基线（同一 checkpoint，controller_tide.py + hotswap_run_tide.py，冻结文件未改）：投机开关与 v6 相同（frontier）；"
            "训练按 TIDE（SC'26）的 Algorithm 1——开机即收集，短/长两条接受率滑动平均（0.3 / 0.02 每秒）在短 < 长 − 0.1 时重新开收集；"
            "收集时 0.5 占空比训，每学 200 条导出换头，换上后测 8 s：接受率比换头前高就保留继续，否则退回上一个头并停止收集。"
            "TIDE 原文是离线留出集评估、训练在另一类 GPU 上，这里改为换头后在线评估、同卡训练；评估窗口内业务标签变了判为无法判定、保留新头。",
    "fixed": "从第一个请求到最后一个都按 0.5 占空比训，100/200/400… 消费量倍增换头；不知道负载也不知道域变了。",
    "fixed800": "10-08 的\"固定策略\"：B 段开始时启动训练器（相当于知道切换点），0.5 占空比，导出 N100/200/400/800 后退出。",
    "v6": "完整控制器：规则 1 frontier 开关（真执行）+ 规则 2 SLO 内免费占空比 + 规则 3 饱和时\"时间到目标\"价值规则 + 到顶/收敛/结构天花板停训。",
}
SCEN_ZH = {"m1_code": "Qwen3-8B · 代码", "m1_zh": "Qwen3-8B · 中文", "m2_code": "Qwen3-30B-A3B · 代码"}


def abl_tables():
    runs = pd.read_csv(f"{ABL_DIR}/abl_runs.csv")
    sm = pd.read_csv(f"{ABL_DIR}/abl_summary.csv")
    pairs = pd.read_csv(f"{ABL_DIR}/abl_pairs.csv")
    order = [k for k in STRAT_ZH if k in set(runs.strategy)]
    return runs, sm, pairs, order


def e19(ctx):
    runs, sm, pairs, order = abl_tables()
    scens = [sc for sc in SCEN_ZH if sc in set(runs.scenario)]
    manifest = jload(f"{ROOT}/data/runs/abl/MANIFEST.json") or {}
    n_runs = len(runs)
    n_rep = int(runs.groupby(["scenario", "strategy"]).size().max())

    def pr(sc, st, ref="static", metric="B_thr"):
        q = pairs[(pairs.scenario == sc) & (pairs.strategy == st) & (pairs.ref == ref) & (pairs.metric == metric)]
        return q.iloc[0] if len(q) else None

    def mean(sc, st, col):
        q = sm[(sm.scenario == sc) & (sm.strategy == st)]
        return float(q[f"{col}_mean"].iloc[0]) if len(q) else None

    def std(sc, st, col):
        q = sm[(sm.scenario == sc) & (sm.strategy == st)]
        return float(q[f"{col}_std"].iloc[0]) if len(q) else None

    def pct(x):
        return f"{x:+.1f}%" if finite(x) else "—"
    v6 = {sc: pr(sc, "v6") for sc in scens}

    def pm(sc, st, ref, metric="B_thr"):
        """paired (same seed) mean / std / same-sign of (st - ref) / ref in %, from the raw runs."""
        a = runs[(runs.scenario == sc) & (runs.strategy == st)].set_index("seed")[metric]
        b = runs[(runs.scenario == sc) & (runs.strategy == ref)].set_index("seed")[metric]
        com = sorted(set(a.index) & set(b.index))
        d = np.array([(a[i] - b[i]) / b[i] * 100 for i in com])
        return dict(m=d.mean(), s=d.std(ddof=1) if len(d) > 1 else float("nan"), same=bool((d > 0).all() or (d < 0).all()), n=len(d), d=d)

    def cv(sc, st):
        v = runs[(runs.scenario == sc) & (runs.strategy == st)].B_thr
        return float(v.std(ddof=1) / v.mean()) if len(v) > 1 else float("nan")
    S = {sc: {st: pm(sc, st, "static") for st in order} for sc in scens}
    ALL = {sc: {st: pm(sc, st, "vanilla", "all_thr") for st in order} for sc in scens}
    P99 = {sc: pm(sc, "v6", "static", "B_p99") for sc in scens}
    A2 = {sc: {st: pm(sc, st, "static", "A2_p50") for st in ("v6", "fixed800")} for sc in scens}
    C, Z, M = "m1_code", "m1_zh", "m2_code"

    def f(x, dec=1):
        return f"{x['m']:+.{dec}f}%±{x['s']:.{dec}f}" if finite(x["s"]) else f"{x['m']:+.{dec}f}%"
    notes = []
    if all(sc in S for sc in (C, Z, M)):
        vs = {sc: pm(sc, "v6", "switch") for sc in scens}
        vf = {sc: pm(sc, "v6", "fixed800") for sc in scens}
        vsim = {sc: pm(sc, "v6", "simple") for sc in scens}
        sw_ns = {sc: pm(sc, "switch", "nospec") for sc in scens}
        notes = [
            f"① 最终系统（v6）三个场景都正常：B 段吞吐相对同基础设施的静态投机 代码 {f(S[C]['v6'])}、中文 {f(S[Z]['v6'])}、MoE {f(S[M]['v6'])}"
            f"（三次都同号）；全程（A1+B+A2）相对今天的部署（vanilla）代码 {f(ALL[C]['v6'])}、中文 {f(ALL[Z]['v6'])}、MoE {f(ALL[M]['v6'])}；"
            f"B 段 p99 TPOT 代码 {f(P99[C])}、中文 {f(P99[Z])}、MoE {f(P99[M])}。",
            f"② 杠杆因场景而异，v6 每次都选对了：8B 饱和段最硬的基线是干脆不投机（纯解码相对静态投机 代码 {f(S[C]['nospec'])}、中文 {f(S[Z]['nospec'])}），"
            f"MoE 恰好相反（纯解码 {f(S[M]['nospec'])}）。v6 在 8B 自动关投机（中文）或换上适应头后保持开（代码），在 MoE 保持开并训练。",
            f"② 联合控制 vs 简单规则：阈值规则（饱和就关投机）在 8B 上与开关接近（代码 {f(S[C]['simple'])}、中文 {f(S[Z]['simple'])}，但波动大），"
            f"在 MoE 上是灾难（{f(S[M]['simple'])}，因为它不知道 MoE 饱和时投机仍快 2 倍）；按 frontier 判别的开关在 MoE 不亏不赚（{f(S[M]['switch'])}）。"
            f"这一对比是\"为什么要 frontier 判别而不是阈值\"的直接证据：v6 − 简单规则 = 代码 {f(vsim[C])}、中文 {f(vsim[Z])}、MoE {f(vsim[M])}。",
            f"② 训练部分的贡献（v6 − 仅开关）：代码 {f(vs[C])}（适应头+保持开 比 关投机 多拿的部分）、中文 {f(vs[Z])}（结构天花板，v6 判定不训，等于开关）、"
            f"MoE {f(vs[M])}（MoE 的收益全部来自训练换头）。",
            f"② 固定训练：已知切换点、训到 N800 就停的固定策略在可适应域比 v6 还高一点（代码 v6−fixed800 {f(vf[C])}、MoE {f(vf[M])}），"
            f"但在中文输 {-vf[Z]['m']:.0f} 个点（它不会关投机，{f(S[Z]['fixed800'])} vs v6 {f(S[Z]['v6'])}）；不知道切换点、全程 0.5 占空比的固定训练"
            f"≈ 静态投机（代码 {f(S[C]['fixed'])}、中文 {f(S[Z]['fixed'])}、MoE {f(S[M]['fixed'])}）：一直训的代价吃掉了更好的头带来的收益。",
            f"② 基础设施的代价：实时导出 + 训练显存预留 = vanilla − static = 代码 {f(S[C]['vanilla'])}、中文 {f(S[Z]['vanilla'])}、MoE {f(S[M]['vanilla'])}。"
            f"另外，运行时把投机关掉的状态比根本不装草稿器慢（仅开关 − 纯解码：代码 {f(sw_ns[C])}、中文 {f(sw_ns[Z])}），一部分是切换前几秒还开着，一部分是关着的草稿器仍有开销——引擎层可以改进。",
            f"③ 重复性：B 段吞吐的变异系数 vanilla/static/纯解码 ≤{max(cv(sc, st) for sc in scens for st in ('vanilla', 'static', 'nospec')):.1%}，"
            f"v6 代码 {cv(C, 'v6'):.1%}、中文 {cv(Z, 'v6'):.1%}、MoE {cv(M, 'v6'):.1%}，仅开关 / 简单规则 {min(cv(sc, st) for sc in (C, Z) for st in ('switch', 'simple')):.1%}–"
            f"{max(cv(sc, st) for sc in (C, Z) for st in ('switch', 'simple')):.1%}（关投机状态和切换时机的波动）。v6 对静态投机三次同号、配对 t 检验 p<0.01；"
            f"v6 与 fixed800 的 1–2 个点差距和 v6 自身的波动同量级——\"控制器优于固定策略\"不成立，\"同量级、且不需要知道切换点\"成立。",
            f"延迟取舍：v6 的 B 段 p99 全部下降；但轻载 A2 段 v6 ≈ 静态投机（代码 {f(A2[C]['v6'])}、中文 {f(A2[Z]['v6'])}、MoE {f(A2[M]['v6'])}），"
            f"而 fixed800 是 {f(A2[C]['fixed800'])} / {f(A2[Z]['fixed800'])} / {f(A2[M]['fixed800'])}：v6 的规则 2 把 SLO 内的余量拿去训练了（\"SLO 内训练免费\"），"
            "适应头本可带来的延迟红利被训练占掉；论文要写明这是设计选择，并给出不训时的延迟。",
        ]
        if "tide" in order:
            vt = {sc: pm(sc, "v6", "tide") for sc in scens}
            notes.insert(5, f"② TIDE 式触发训练（冻结后补的基线，见 E22）：v6 − TIDE 式 = 代码 {f(vt[C])}、中文 {f(vt[Z])}、MoE {f(vt[M])}；"
                            "可适应的场景两者接近，中文上 TIDE 式反复重开训练一个过不了盈亏线的头。")
        if os.path.exists(E22_ANOM):
            an_ = pd.read_csv(E22_ANOM)
            nt_ = int((an_.strategy == "tide").sum())
            notes.append(f"④ 主表里有 {len(an_)} 次运行是换节点重跑（干扰扫描标记的冻结矩阵基线 {len(an_) - nt_} 次 + TIDE 式 {nt_} 次，"
                         "原始运行在 data/runs/abl_anomaly/，见 E22）；环境按原版本重建，TIDE 式与重跑都在重建环境里。")
    ctx["e19"] = dict(v6=v6, scens=scens, n_runs=n_runs, n_rep=n_rep, runs=runs, sm=sm, pairs=pairs,
                      mean=mean, std=std, pr=pr, pm=pm, cv=cv, S=S, ALL=ALL, P99=P99, A2=A2, notes=notes)
    ctx["e19_notes"] = notes
    if all(sc in S for sc in (C, Z, M)):
        ctx["e19_verdict"] = (f"三场景成立（B 段 {S[C]['v6']['m']:+.0f}% / {S[Z]['v6']['m']:+.0f}% / {S[M]['v6']['m']:+.0f}%，三次同号）；"
                              f"算法的独立贡献 = frontier 判别（阈值规则在 MoE {S[M]['simple']['m']:+.0f}%）+ 训练换头（MoE 全部、代码约一半）；"
                              f"8B 饱和段最大杠杆是关投机，v6 追平纯解码；已知切换点的固定训练在可适应域高 1–2 点")
        ctx["e19_word"] = "成立"
    parts = []
    for sc in scens:
        p = v6[sc]
        if p is not None:
            parts.append(f"{SCEN_ZH[sc]} {pct(p.mean_pct)}" + (f"（±{p.std_pct:.1f}，{int(p.n)} 次{'同号' if p.same_sign else '异号'}）" if finite(p.std_pct) else f"（{int(p.n)} 次）"))
    lead = (f"同一冻结 checkpoint 上 {len(scens)} 个场景 × {len(order)} 个策略 × {n_rep} 次重复 = {n_runs} 次闭环。完整 v6 相对\"同基础设施的静态投机\"，"
            f"B 段吞吐：{'；'.join(parts)}。各策略逐条比较见下表；结论句见「怎么读」。")

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], [24, 11, 11, 11, 12, 11, 11, 11, 11, 11, 11, 24])
        s.title_(f"{e['id']} · {e['name']}", lead)
        s.section("实验配置")
        hashes = manifest.get("code", {})
        vp = manifest.get("vllm_patched", {})
        heads = manifest.get("heads", {})
        s.kv([("要回答", "论文前三项补充：① 最终完整系统（v6）在三个场景上是否都正常；② 联合控制算法相对简单规则、单项机制的独立贡献；"
                       "③ 关键策略的差别是否大于随机波动（≥3 次独立重复，报均值、标准差、按次配对的差值）。"),
              ("冻结", f"全部 {n_runs} 次运行用同一套代码、同一个引擎补丁、同一个出厂头、同一个环境、同一组提示池（清单 data/runs/abl/MANIFEST.json，"
                     f"生成 {manifest.get('generated', '')[:16]}）。关键 hash（sha256 前 16 位）：controller.py {hashes.get('code/controller.py', '?')}、"
                     f"hotswap_run.py {hashes.get('code/hotswap_run.py', '?')}、train_head.py {hashes.get('code/train_head.py', '?')}、"
                     f"vLLM gpu_model_runner.py {vp.get('v1/worker/gpu_model_runner.py', '?')}、8B 头 {heads.get('qwen3-8b-eagle3/pytorch_model.bin', '?')}、"
                     f"MoE 头 {heads.get('qwen3-a3b-eagle3/model.safetensors', '?')}；环境 {manifest.get('env', '')}。"
                     "10-08 的 v4/v5/v5.1 数字（E13–E15）只当开发记录，不进论文主表。"
                     "冻结后补的 TIDE 式基线、被干扰扫描标记后换节点的重跑，用的是按原 203 个包版本重建的环境 ss312（同代码、同补丁、同头，见 E22）。"),
              ("场景", "A 段对话 C=16（200 条）→ B 段新域饱和 C=256（16000 条唯一提示，约 4.5–10 分钟）→ A2 段对话 C=16（200 条，与 A1 不同的提示）。"
                     "三个场景：8B 代码、8B 中文、30B-A3B（MoE）代码。SLO TPOT ≤ 16.7 ms。"),
              ("引擎配置", "除 vanilla 外全部相同：util = 0.90 − 训练预留（8B 18 GB、MoE 5 GB），实时导出开着，max_num_seqs 256，无前缀缓存，贪心；"
                        "无投机的策略也用同样的 KV 预算（全程零抢占，预算不是瓶颈）。"),
              ("重复", f"每个（场景, 策略）独立跑 {n_rep} 次：第 r 次用第 r 套提示顺序（三段提示池各自用种子 r 重洗）和训练器种子 r；"
                     "同一个 r 下所有策略拿到完全相同的请求序列，所以差值按 r 配对。"),
              ("口径", "B 段吞吐 = bench 的输出 token/s（主指标）；B 段 p99 TPOT；A2 段 p50 TPOT（对话段受益）；全程吞吐 = 三段输出 token ÷ 三段时长。"
                     "配对差 = (策略 − 对照) ÷ 对照，按 r 配对后取均值 ± 标准差；\"同号\"= 三次差值符号一致；p 为配对 t 检验（n=3，自由度 2，只作参考）。")])
        s.table([("k", "策略", None), ("d", "做法", None, 11)], [(k, {"k": V(STRAT_ZH[k], bold=True), "d": STRAT_DESC[k]}) for k in order],
                caption=f"表 0　{len(order)} 个策略")
        s.section("结果")
        for sc in scens:
            rows = []
            for st in order:
                p_s, p_v = pr(sc, st), pr(sc, st, ref="vanilla")
                rows.append((st, {"k": V(STRAT_ZH[st], bold=(st == "v6")), "n": V(int(sm[(sm.scenario == sc) & (sm.strategy == st)].n.iloc[0]), fmt=INT),
                                  "thr": mean(sc, st, "B_thr"), "sd": std(sc, st, "B_thr"),
                                  "cv": F("={sd}{r}/{thr}{r}", std(sc, st, "B_thr") / mean(sc, st, "B_thr"), PCTP) if finite(std(sc, st, "B_thr")) else None,
                                  "dx": F("={thr}{r}/{@static|thr}-1", mean(sc, st, "B_thr") / mean(sc, "static", "B_thr") - 1, PCT) if st != "static" else V("对照", status="对照"),
                                  "pd": (f"{p_s.mean_pct:+.1f} ± {p_s.std_pct:.1f}" + ("" if p_s.same_sign else "（异号）") if p_s is not None and finite(p_s.std_pct)
                                         else ("对照" if st == "static" else "—")),
                                  "dv": (f"{p_v.mean_pct:+.1f} ± {p_v.std_pct:.1f}" if p_v is not None and finite(p_v.std_pct) else ("对照" if st == "vanilla" else "—")),
                                  "p99": mean(sc, st, "B_p99"), "a2": mean(sc, st, "A2_p50"), "acc": mean(sc, st, "acc_last"),
                                  "duty": mean(sc, st, "duty_B"), "sp": V(mean(sc, st, "spec_on_B"), fmt=PCTP), "sw": V(mean(sc, st, "swaps"), fmt=F1),
                                  "tk": V(mean(sc, st, "train_tok") / 1e6, fmt=F2)}))
            s.table([("k", "策略", None), ("n", "次数", INT), ("thr", "B 段吞吐\n均值 tok/s", INT), ("sd", "标准差", INT), ("cv", "变异\n系数", PCTP),
                     ("dx", "均值相对\n静态投机", PCT), ("pd", "配对差 %\n均值 ± 标准差", None), ("dv", "相对现网\n配对差 %", None),
                     ("p99", "B 段 p99\nTPOT ms", F1), ("a2", "A2 段 p50\nTPOT ms", F2), ("acc", "B 末接受\n长度", F2)],
                    rows, caption=f"表 {scens.index(sc) + 1}　{SCEN_ZH[sc]}：B 段饱和（C=256，16000 条）",
                    note="接受长度为空 = B 段末尾投机是关着的（没有草稿）。")
            rows = []
            for st in order:
                rows.append((st, {"k": STRAT_ZH[st], "duty": V(mean(sc, st, "duty_B"), fmt=F2), "sp": V(mean(sc, st, "spec_on_B"), fmt=PCTP),
                                  "sw": V(mean(sc, st, "swaps"), fmt=F1), "tk": V(mean(sc, st, "train_tok") / 1e6, fmt=F2),
                                  "bd": V(mean(sc, st, "B_dur_s"), fmt=INT), "a1": V(mean(sc, st, "A1_p50"), fmt=F2),
                                  "all": V(mean(sc, st, "all_thr"), fmt=INT),
                                  "ax": F("={all}{r}/{@static|all}-1", mean(sc, st, "all_thr") / mean(sc, "static", "all_thr") - 1, PCT) if st != "static" else V("对照", status="对照")}))
            s.table([("k", "策略", None), ("duty", "B 段训练\n占空比", F2), ("sp", "B 段投机\n开着的比例", PCTP), ("sw", "换头次数\n（全程）", F1),
                     ("tk", "训练 token\nM（全程）", F2), ("bd", "B 段时长 s", INT), ("a1", "A1 段 p50\nTPOT ms", F2), ("all", "全程吞吐\ntok/s", INT),
                     ("ax", "全程相对\n静态投机", PCT)], rows, caption=f"表 {scens.index(sc) + 1}b　{SCEN_ZH[sc]}：控制行为与全程口径（三次均值）",
                    note="全程吞吐 = A1+B+A2 的输出 token ÷ 三段总时长（每个策略服务同样的请求，时间短就是好）。训练 token 含 A 段。")
            rows = []
            for st in order:
                q = runs[(runs.scenario == sc) & (runs.strategy == st)].sort_values("seed")
                cells = {"k": STRAT_ZH[st]}
                for _, r in q.iterrows():
                    cells[f"r{int(r.seed)}"] = r.B_thr
                    cells[f"q{int(r.seed)}"] = r.B_p99
                    cells[f"d{int(r.seed)}"] = V(float(r.duty_B), fmt=F2)
                rows.append((st, cells))
            seeds = sorted(set(int(x) for x in runs.seed))[:3]
            s.table([("k", "策略", None)] + [(f"r{r}", f"第 {r} 次\nB 吞吐", INT) for r in seeds] + [(f"q{r}", f"第 {r} 次\np99 ms", F1) for r in seeds]
                    + [(f"d{r}", f"第 {r} 次\n占空比", F2) for r in seeds],
                    rows, caption=f"表 {scens.index(sc) + 1}c　{SCEN_ZH[sc]}：逐次原始值（配对看这里）")
        s.section("关键配对比较")
        rows = []
        for sc in scens:
            for st, ref, why in [("v6", "static", "完整系统的总收益（同基础设施）"), ("v6", "vanilla", "相对今天的部署（含导出与预留的代价）"),
                                 ("v6", "simple", "联合控制算法 vs 简单规则"), ("v6", "switch", "训练部分的贡献（开关之上）"),
                                 ("v6", "fixed800", "控制器 vs 已知切换点的固定训练"), ("v6", "tide", "控制器 vs TIDE 式触发训练"),
                                 ("switch", "static", "开关单独的贡献"),
                                 ("simple", "static", "简单规则单独的收益"), ("nospec", "static", "纯解码 vs 静态投机（饱和时投机值不值）")]:
                if ref == "static" or ref == "vanilla":
                    p = pr(sc, st, ref=ref)
                else:
                    # paired difference st - ref computed from the raw runs (pairs.csv only has static/vanilla refs)
                    a = runs[(runs.scenario == sc) & (runs.strategy == st)].set_index("seed").B_thr
                    b = runs[(runs.scenario == sc) & (runs.strategy == ref)].set_index("seed").B_thr
                    com = sorted(set(a.index) & set(b.index))
                    if not com:
                        continue
                    d = np.array([(a[i] - b[i]) / b[i] * 100 for i in com])
                    p = pd.Series(dict(n=len(d), mean_pct=d.mean(), std_pct=d.std(ddof=1) if len(d) > 1 else float("nan"),
                                       same_sign=bool((d > 0).all() or (d < 0).all()), p=float("nan"),
                                       per_seed=json.dumps([round(x, 2) for x in d])))
                    if len(d) > 1 and d.std(ddof=1) > 0:
                        try:
                            from scipy import stats
                            p["p"] = float(2 * stats.t.sf(abs(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))), len(d) - 1))
                        except Exception:
                            pass
                if p is None:
                    continue
                verdict = ("样本不足" if int(p.n) < 3 else
                           "差别可信" if (p.same_sign and abs(p.mean_pct) > 2 * (p.std_pct if finite(p.std_pct) else 0)) else "噪声内")
                rows.append((f"{sc}{st}{ref}", {"sc": SCEN_ZH[sc], "cmp": f"{STRAT_ZH[st]} − {STRAT_ZH[ref]}", "why": why, "m": V(p.mean_pct / 100, fmt=PCT),
                                                "sd": V(p.std_pct / 100 if finite(p.std_pct) else None, fmt=PCTP), "ps": str(p.per_seed),
                                                "ss": "同号" if p.same_sign else "异号", "p": V(float(p.p) if finite(p.p) else None, fmt=F3),
                                                "v": V(verdict, status="成立" if verdict == "差别可信" else ("参考" if verdict == "样本不足" else "边缘"))}))
        s.table([("sc", "场景", None), ("cmp", "比较（B 段吞吐）", None, 2), ("why", "看什么", None, 2), ("m", "配对差\n均值", PCT), ("sd", "标准差", PCTP),
                 ("ps", "逐次差值 %", None, 2), ("ss", "符号", None), ("p", "配对 t\np 值", F3), ("v", "判定", None)], rows,
                caption="表 4　关键配对比较（\"差别可信\" = 三次同号且 |均值| > 2 × 标准差；n=3 的 p 值只作参考）")
        s.section("怎么读")
        s.bullets(ctx.get("e19_notes", ["（结论句在矩阵跑完后填写）"]))
        s.source(f"data/runs/abl/<场景>/<策略>/r<次>/（done.json、phase*.json、timeline/trainer/controller.jsonl）；{ABL_DIR}/abl_{{runs,summary,pairs}}.csv"
                 "（code/specsched/analyze_abl.py）；data/runs/abl/MANIFEST.json；矩阵定义 code/specsched/abl_matrix.py")
    res = "；".join(f"{SCEN_ZH[sc]} v6 vs 静态投机 {pct(v6[sc].mean_pct)}±{v6[sc].std_pct:.1f}" for sc in scens if v6[sc] is not None)
    return meta(ctx, "E19", short="消融×重复矩阵", name="消融 × 重复矩阵：统一 v6、同一冻结 checkpoint",
                q="最终系统在三个场景都正常吗？联合控制相对简单规则/单项机制多贡献多少？差别大于随机波动吗？",
                cfg=f"3 场景 × {len(order)} 策略 × {n_rep} 次重复 = {n_runs} 次闭环；B 段 C=256 16000 条；同一 MANIFEST", res=res,
                verdict=ctx.get("e19_verdict", "见明细"), word=ctx.get("e19_word", "成立"), status="有效", build=build)


def e20(ctx):
    runs = ctx["e19"]["runs"]

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], [14, 20, 6, 10, 10, 8, 10, 9, 9, 9, 9, 9, 9, 8, 8, 10, 7, 7, 8, 8])
        s.title_(f"{e['id']} · {e['name']}", f"{len(runs)} 次闭环运行的逐条原始指标（E19 的表由这里汇总）。每行一次运行：场景 / 策略 / 第几次 / 作业。")
        rows = []
        for i, r in runs.iterrows():
            rows.append((str(i), {"sc": SCEN_ZH.get(r.scenario, r.scenario), "st": STRAT_ZH.get(r.strategy, r.strategy), "seed": V(int(r.seed), fmt=INT),
                                  "job": str(r.job), "host": str(r.host), "srv": V(float(r.server_s), fmt=INT), "util": V(float(r.util), fmt=F3),
                                  "thr": V(float(r.B_thr), fmt=INT), "dur": V(float(r.B_dur_s), fmt=F1), "p50": V(float(r.B_p50), fmt=F2), "p99": V(float(r.B_p99), fmt=F1),
                                  "a1": V(float(r.A1_p50), fmt=F2), "a2": V(float(r.A2_p50), fmt=F2),
                                  "accf": V(float(r.acc_first) if finite(r.acc_first) else None, fmt=F2), "accl": V(float(r.acc_last) if finite(r.acc_last) else None, fmt=F2),
                                  "duty": V(float(r.duty_B), fmt=F2), "tk": V(float(r.train_tok) / 1e6, fmt=F2), "sp": V(float(r.spec_on_B), fmt=PCTP),
                                  "sw": V(int(r.swaps), fmt=INT), "all": V(float(r.all_thr), fmt=INT)}))
        s.table([("sc", "场景", None), ("st", "策略", None), ("seed", "次", INT), ("job", "作业", None), ("host", "节点", None), ("srv", "服务启动 s", INT),
                 ("util", "util", F3), ("thr", "B 吞吐\ntok/s", INT), ("dur", "B 时长 s", F1), ("p50", "B p50\nms", F2), ("p99", "B p99\nms", F1),
                 ("a1", "A1 p50\nms", F2), ("a2", "A2 p50\nms", F2), ("accf", "接受\nB 首", F2), ("accl", "接受\nB 末", F2), ("duty", "占空比\nB", F2),
                 ("tk", "训练 M\ntoken", F2), ("sp", "投机开\nB", PCTP), ("sw", "换头", INT), ("all", "全程吞吐\ntok/s", INT)], rows, caption="表 1　逐次运行")
        s.ws.freeze_panes = "D5"
        s.source(f"{ABL_DIR}/abl_runs.csv（analyze_abl.py 从 data/runs/abl/ 生成）")
    return meta(ctx, "E20", short="矩阵逐次原始值", name="消融 × 重复矩阵：逐次运行原始指标", q="每次运行的原始数字（供核对与配对）",
                cfg="同 E19", res=f"{len(runs)} 行", verdict="原始数据", word="参考", status="有效", build=build)


def e21(ctx):
    sd = f"{ROOT}/results/summaries"
    v = jload(f"{sd}/sim_heldout_verdict.json")
    tr = pd.read_csv(f"{sd}/sim_heldout_trace.csv")
    d1 = pd.read_csv(f"{sd}/sim_heldout_driftI.csv")
    d2 = pd.read_csv(f"{sd}/sim_heldout_driftII.csv")
    d3 = pd.read_csv(f"{sd}/sim_heldout_driftIII.csv")
    SIMPLE = ["static_off", "spec_only", "osd_idle", "fixed_interval", "accdrop"]
    POLZ = dict(POL_ZH, fixed_interval="定时训（每 5 分钟训 30 秒）")
    g = tr.pivot_table(index=["trace", "start_hour", "scale"], columns="policy", values="goodput") / 1e6
    t = tr.pivot_table(index=["trace", "start_hour", "scale"], columns="policy", values="train_tok") / 1e6
    V1, V2, V3, V4 = v["V1"], v["V2"], v["V3"], v["V4"]
    ctx["e21"] = v
    lead = (f"冻结模拟器（sim.py {v['frozen']['sim.py']}、model.json {v['frozen']['model.json']}）在 {V1['n']} 条新 trace 窗口和 "
            f"{V2['n_all'] + V3['n'] + V4['n']} 个随机漂移场景上重跑：V1 goodput 成立（{V1['n_pass_ratio']}/{V1['n']} 条 ≥0.97×最好简单策略，"
            f"{V1['n_best']} 条第一，最差 {V1['min_ratio']:.3f}）；V2 成立（≥10 s 突发 {V2['n_win']}/{V2['n_long']} 不输投机常开）；"
            f"V3 排序 {V3['n_cm_ge_spec']}/{V3['n']} 成立但 1.5× 幅度不成立（可行饱和 {V3['feasible_sat']['min']:.2f}–{V3['feasible_sat']['max']:.2f}×）；"
            f"V4 成立（{V4['n_ge']}/{V4['n']}，唯一例外是浪涌开始 1 秒的切换瞬态）。三处原文说法改弱：\"≥3 s 就值得训\"→ ≥60 s 才可靠回本；"
            "\"中文砍半\"→ 1.4–2×；\"只有 cost-model 扛 1.1× 浪涌\"→ accdrop 也能。")

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], [22, 12, 9, 12, 17, 12, 10, 12, 12, 12, 12])
        s.title_(f"{e['id']} · {e['name']}", lead)
        s.section("实验配置")
        s.kv([("要回答", "E08/E09 的模拟结论是看过结果后调过模型和规则得到的；把模拟器冻结，在没参与调参的新负载轨迹上还成立吗？"),
              ("冻结", f"sched/sim.py（sha256 前 16 位 {v['frozen']['sim.py']}）与 sched/model.json（{v['frozen']['model.json']}）；占空比档 {{0,0.1,0.25,0.5,1}}、"
                     "规则 3 训练窗 10 s、缓冲 4M token、lindy 下限 30 s、SLO = 2×T_base(1) + 排队 ≤1 s、trace 按 p95 需求 = 尺度 × 容量缩放。判据先写后跑，未按结果改。"),
              ("新轨迹", "BurstGPT 对话日志第 4/6/8/12/16/20 天（原只用第 2 天）× 尺度 0.6/0.8/1.0；API 日志第 15/21/22 天（原只用第 10 天）× 3 尺度，当代码域；"
                       "Azure 2023 对话/代码 1 小时 trace（从未用过）× 0.6/0.8/1.0/1.2。随机漂移：I 60 个（A∈{对话,代码,fc}→B∈{代码,SQL,fc} 饱和 1.0–1.2×，持续 5–600 s）、"
                       "II 40 个（→中文，C=128/192/256 或 1.1× 过载）、III 20 个 × 7 档浪涌（0.9–1.2× 容量，平静期 120–900 s）。"),
              ("策略", "不投机不训 / 投机常开 / OSD 空闲训 / 定时训 / accdrop / cost-model × {oracle, persist, lindy}；部署用的是 lindy，判据都按它算。"),
              ("脚本", "code/specsched/sched/validate_heldout.py（本机 8 进程约 6 分钟）→ results/summaries/sim_heldout_{trace,driftI,driftII,driftIII}.csv + sim_heldout_verdict.json")])
        s.section("判据与判定")
        rows = [("V1", {"k": V("V1 trace 回放", bold=True),
                        "c": "每条新轨迹 cost-model goodput ≥ 0.97 × 最好的简单策略，且训练 token ≤ OSD 的 10%",
                        "r": f"goodput {V1['n_pass_ratio']}/{V1['n']} 过（最差 {V1['min_ratio']:.3f}，中位 {V1['median_ratio']:.3f}，{V1['n_best']} 条第一；"
                             f"相对投机常开 {V1['gain_vs_spec_only'][0]:.2f}–{V1['gain_vs_spec_only'][1]:.2f}×）。训练 token {V1['n_pass_train']}/{V1['n']}："
                             f"cost-model 每条 {V1['cm_train_M'][0]:.1f}–{V1['cm_train_M'][1]:.1f}M；例外是 Azure 对话 4 个 1 小时窗口，没有空闲期，OSD 自己只训 "
                             f"{min(x['osd_M'] for x in V1['train_exceptions']):.2f}–{max(x['osd_M'] for x in V1['train_exceptions']):.2f}M",
                        "v": V("goodput 成立；\"少训 100×\"限有空闲期的 trace", status="部分成立")}),
                ("V2", {"k": V("V2 漂移 I 持续时长", bold=True), "c": "≥10 s 的饱和突发里 cost-model ≥ 投机常开 的比例 ≥ 80%",
                        "r": f"{V2['n_win']}/{V2['n_long']}（{V2['frac']:.0%}）；全部 {V2['n_all']} 个里 {V2['n_win_all']} 个 ≥，输的最多输 {V2['max_loss_pct']:.1f}%。"
                             f"但多数是平手：≤20 s 严格赢 {min(V2['by_dur'][k]['strict_win'] for k in ('5', '10', '20')):.0%}–"
                             f"{max(V2['by_dur'][k]['strict_win'] for k in ('5', '10', '20')):.0%}，60 s 中位 {V2['by_dur']['60']['median']:+.1%}，"
                             f"180 s {V2['by_dur']['180']['strict_win']:.0%} 严格赢、中位 {V2['by_dur']['180']['median']:.2f}×，600 s 中位 {V2['by_dur']['600']['median']:.1f}×",
                        "v": V("成立；可靠回本改为 ≥60 s", status="成立")}),
                ("V3", {"k": V("V3 漂移 II 结构天花板", bold=True), "c": "中文饱和：cost-model ≥ 1.5 × 投机常开，且不输反应式训练（OSD、accdrop）",
                        "r": f"排序 {V3['n_cm_ge_spec']}/{V3['n']} 成立（每个场景都严格高于投机常开、OSD、accdrop，最差 {V3['min_ratio_vs_spec_all']:.2f}×）。幅度：可行饱和 "
                             f"{V3['feasible_sat']['n']} 个 {V3['feasible_sat']['min']:.2f}–{V3['feasible_sat']['max']:.2f}×（中位 {V3['feasible_sat']['median']:.2f}）；"
                             f"1.1× 过载 {V3['overload']['n']} 个 {V3['overload']['min']:.2f}–{V3['overload']['max']:.2f}×（过载期谁都违约）；轻载 {V3['light']['n']} 个 "
                             f"≥{V3['light']['min']:.1f}×（投机常开在中文 C≥64 就违约）",
                        "v": V("排序成立；1.5× 不成立", status="部分成立")}),
                ("V4", {"k": V("V4 漂移 III 浪涌", bold=True), "c": "无违约能扛的最大浪涌：cost-model ≥ OSD 且 ≥ accdrop 的场景 ≥ 80%",
                        "r": f"{V4['n_ge']}/{V4['n']}。均值：cost-model {V4['max_surge_mean']['costmodel_lindy']:.2f}×、accdrop {V4['max_surge_mean']['accdrop']:.2f}、"
                             f"OSD {V4['max_surge_mean']['osd_idle']:.2f}、投机常开 {V4['max_surge_mean']['spec_only']:.2f}。对 accdrop 赢 {V4['vs_accdrop']['better']} / 平 "
                             f"{V4['vs_accdrop']['equal']} / 输 {V4['vs_accdrop']['worse']}；对 OSD 赢 {V4['vs_osd']['better']} / 平 {V4['vs_osd']['equal']} / 输 {V4['vs_osd']['worse']}。"
                             f"唯一的\"输\"是浪涌开始那 1 秒的切换瞬态（容忍 10 s 则 {V4['tol10s']['n_ge']}/{V4['n']}，均值 {V4['tol10s']['max_surge_mean']['costmodel_lindy']:.2f}×）",
                        "v": V("成立；但 accdrop 同样能扛", status="成立")})]
        s.table([("k", "判据", None), ("c", "通过线（预先写定）", None, 3), ("r", "结果", None, 5), ("v", "判定", None, 2)], rows, caption="表 1　四条判据")
        s.section("V1：35 条新 trace 窗口")
        rows = []
        for idx, rr in g.iterrows():
            trc, sh, sc = idx
            best = rr[SIMPLE].idxmax()
            key = f"{trc}{sh}{sc}"
            rows.append((key, {"t": f"{trc}（第 {sh / 24 + 1:.0f} 天）" if "burstgpt" in trc else trc, "s": V(float(sc), fmt=F1),
                               "b": V(float(rr[SIMPLE].max()), fmt=F1), "bn": POLZ[best], "cm": V(float(rr["costmodel_lindy"]), fmt=F1),
                               "x": F("={cm}{r}/{b}{r}", float(rr["costmodel_lindy"] / rr[SIMPLE].max()), F3),
                               "so": V(float(rr["spec_only"]), fmt=F1), "off": V(float(rr["static_off"]), fmt=F1),
                               "tc": V(float(t.loc[idx, "costmodel_lindy"]), fmt=F2), "to": V(float(t.loc[idx, "osd_idle"]), fmt=F2)}))
        s.table([("t", "轨迹", None), ("s", "尺度", F1), ("b", "最好简单策略\ngoodput M", F1), ("bn", "是谁", None), ("cm", "cost-model\ngoodput M", F1),
                 ("x", "比值", F3), ("so", "投机常开\nM", F1), ("off", "不投机\nM", F1), ("tc", "cost-model\n训练 M", F2), ("to", "OSD\n训练 M", F2)], rows,
                caption="表 2　每条窗口（goodput 单位百万 token；尺度 = p95 需求 / 单卡容量）")
        s.section("V2：按突发时长")
        rows = []
        for dur in ["5", "10", "20", "60", "180", "600"]:
            b = V2["by_dur"][dur]
            rows.append((dur, {"d": V(int(dur), fmt=INT), "n": V(b["n"], fmt=INT), "w": V(b["strict_win"], fmt=PCTP), "t": V(b["tie"], fmt=PCTP), "l": V(b["loss"], fmt=PCTP),
                               "wa": V(b["strict_win_adaptable"], fmt=PCTP), "m": V(b["median"], fmt=X2), "mx": V(b["max"], fmt=X2)}))
        s.table([("d", "突发时长 s", INT), ("n", "场景数", INT), ("w", "严格赢\n(>0.1%)", PCTP), ("t", "平手\n(±0.1%)", PCTP), ("l", "输", PCTP),
                 ("wa", "可适应域\n严格赢", PCTP), ("m", "中位\ncost-model/投机常开", X2), ("mx", "最大", X2)], rows,
                caption="表 3　cost-model 相对投机常开（fc 域学习曲线平，训练基本无收益，拉低了\"严格赢\"）")
        s.section("V3 / V4：按场景")
        g2 = d2.pivot_table(index="id", columns="policy", values="goodput") / 1e6
        m2 = d2.drop_duplicates("id").set_index("id")
        rows = []
        for i in g2.index:
            r_, mm = g2.loc[i], m2.loc[i]
            rows.append((str(i), {"i": V(int(i), fmt=INT), "a": f"{DOM_ZH[mm.A]}@C{int(mm.C_A)}{'（已适应）' if mm.NA else ''}",
                                  "b": f"中文 {'1.1× 过载' if finite(mm.lamB_mult) else 'C=' + str(int(mm.C_B))} {int(mm.dur)} s",
                                  "cm": V(float(r_["costmodel_lindy"]), fmt=F2), "so": V(float(r_["spec_only"]), fmt=F2), "osd": V(float(r_["osd_idle"]), fmt=F2),
                                  "ac": V(float(r_["accdrop"]), fmt=F2), "off": V(float(r_["static_off"]), fmt=F2),
                                  "x": F("={cm}{r}/{so}{r}", float(r_["costmodel_lindy"] / r_["spec_only"]), X2)}))
        s.table([("i", "#", INT), ("a", "A 段", None), ("b", "B 段", None), ("cm", "cost-model", F2), ("so", "投机常开", F2), ("osd", "OSD", F2),
                 ("ac", "accdrop", F2), ("off", "不投机", F2), ("x", "cost-model\n/投机常开", X2)], rows, caption="表 4　V3：40 个中文场景的 goodput（M token）")
        ok = d3[d3.surge_viol_s == 0].groupby(["id", "policy"]).surge.max().unstack().reindex(columns=["spec_only", "osd_idle", "accdrop", "costmodel_lindy"]).fillna(0)
        m3 = d3.drop_duplicates("id").set_index("id")
        rows = []
        for i in ok.index:
            mm = m3.loc[i]
            rows.append((str(i), {"i": V(int(i), fmt=INT), "a": f"{DOM_ZH[mm.A]}@C{int(mm.C_A)}{'（已适应）' if mm.NA else ''}，平静 {int(mm.pre)} s",
                                  "b": f"{DOM_ZH[mm.B]} C={int(mm.C_B)} {int(mm.dur)} s → 浪涌 600 s",
                                  **{p: V(float(ok.loc[i, p]), fmt=X2) for p in ok.columns}}))
        s.table([("i", "#", INT), ("a", "A 段", None), ("b", "B 段", None), ("spec_only", "投机常开", X2), ("osd_idle", "OSD", X2), ("accdrop", "accdrop", X2),
                 ("costmodel_lindy", "cost-model", X2)], rows, caption="表 5　V4：无违约能扛的最大浪涌（× 不投机容量；0 = 连 0.9× 都有违约）",
                note="#8 的 cost-model 为 0：浪涌开始的第 1 秒 TPOT 19.0 ms > 16.7（开关切换瞬态），之后全程达标；其它策略的瞬态违约发生在 B 段开始，不在浪涌窗口里。")
        s.section("怎么读")
        s.bullets(["E08 的主结论（cost-model 不输任何简单策略、训练量小两个数量级）在 35 条新轨迹上成立；差距最大的是过载窗口（Azure 对话 1.2×：比最好的简单策略高 28%）。",
                   "E09 三类里，结构天花板（中文）仍是差距最大的一项，但 2× 是饱和且可行时的典型值，过载时谁都违约、差距缩到 1.1–1.8×。",
                   "\"训练太便宜、什么时候训退化\"要加限定：短于 20 s 的突发训练不吃亏也不赚（平手），≥60 s 才可靠回本。",
                   "抗浪涌不是 cost-model 独有：accdrop 在平静期也把头训满。cost-model 的独有之处是四条同时成立（accdrop 在 V3 输、OSD 在 V1 多训 100 倍）。",
                   "冻结代码上 E08 对话回放的 cost-model 为 253.4M（结论文档旧版 246.3M 是加按域 frontier 之前的数字）。"])
        s.source("results/summaries/sim_heldout_{trace,driftI,driftII,driftIII}.csv、sim_heldout_verdict.json；"
                 "results/summaries/2026-10-23_模拟冻结验证.md；code/specsched/sched/{validate_heldout.py,sim.py,model.json}；data/traces/")
    return meta(ctx, "E21", short="模拟冻结验证", name="模拟器冻结后在新轨迹上的验证（E08/E09）", date="10-23（本机）",
                q="冻结模拟器和规则后，E08/E09 的结论在没参与调参的新负载轨迹上还成立吗？",
                cfg="冻结 sim.py/model.json；35 条新 trace 窗口 + 120 个随机漂移场景 × 8 策略；判据 V1–V4 预先写定",
                res=f"V1 goodput {V1['n_pass_ratio']}/{V1['n']}（{V1['n_best']} 条第一）；V2 {V2['n_win']}/{V2['n_long']}；V3 排序 {V3['n_cm_ge_spec']}/{V3['n']}，"
                    f"幅度 1.4–2×；V4 {V4['n_ge']}/{V4['n']}；三处原文说法改弱",
                verdict="主结论成立；三处口径改弱（回本时长 ≥60 s、中文 1.4–2×、抗浪涌非独有）", word="部分成立", status="有效（模拟）", build=build)


E22_SCAN = f"{ROOT}/results/abl_all/anomaly_scan.csv"
E22_DEC = f"{ROOT}/results/abl_all/anomaly_scan_decision.csv"
E22_ANOM = f"{ROOT}/results/abl_anomaly/abl_runs.csv"
E22_ENVC = f"{ROOT}/results/abl_envcheck/abl_runs.csv"
E22_TV = f"{ROOT}/results/abl_tidevar/abl_runs.csv"
E22_ORIG72 = f"{ROOT}/results/abl/abl_runs.csv"


def tide_behaviour(sc, root="abl", strat="tide"):
    """per-run TIDE-style controller behaviour (rounds, kept / inconclusive / rejected, drift triggers, training share of B)"""
    out = []
    for d in sorted(glob.glob(f"{ROOT}/data/runs/{root}/{sc}/{strat}/r*")):
        ev = [json.loads(l) for l in open(f"{d}/events.jsonl") if l.strip()]
        tB = next(e["t"] for e in ev if e.get("event") == "phase" and e.get("name") == "B")
        tBe = next(e["t"] for e in ev if e.get("event") == "phase_end")
        ctl = [json.loads(l) for l in open(f"{d}/controller.jsonl") if l.strip()]
        evs = [e for e in ctl if e.get("event") == "tide_eval"]
        ticks = [e for e in ctl if "duty" in e and "event" not in e and "at" in e]
        tb = [e for e in ticks if tB <= e["at"] <= tBe]
        rejB = [e["at"] for e in evs if not e["keep"] and tB <= e["at"] <= tBe]
        drB = [e["at"] for e in ctl if e.get("event") == "drift" and tB <= e["at"] <= tBe]
        out.append(dict(seed=int(d.rsplit("r", 1)[1]), rounds=len(evs), rejB=len(rejB),
                        rejdrift=sum(1 for t_ in rejB if any(0 <= x - t_ <= 2 for x in drB)),
                        kept=sum(1 for e in evs if e["keep"] and not e.get("inconclusive")),
                        inconcl=sum(1 for e in evs if e.get("inconclusive")), rejected=sum(1 for e in evs if not e["keep"]),
                        drift=sum(1 for e in ctl if e.get("event") == "drift"),
                        trainB=sum(1 for e in tb if e["duty"] > 0) / max(1, len(tb))))
    return out


def e22(ctx):
    runs = ctx["e19"]["runs"]
    pm = ctx["e19"]["pm"]
    scens = ctx["e19"]["scens"]
    scan = pd.read_csv(E22_SCAN, dtype={"rep": str})
    anom = pd.read_csv(E22_ANOM)
    envc = pd.read_csv(E22_ENVC) if os.path.exists(E22_ENVC) else None
    o72 = pd.read_csv(E22_ORIG72)
    main_scan = scan[scan.root == "abl"]
    orig_scan = scan[scan.root == "abl_anomaly"]
    rer = {(a.scenario, a.strategy, int(a.seed)) for a in anom.itertuples()}
    is_rer = main_scan.apply(lambda r: (r.scenario, r.strategy, int(r.rep.lstrip("r"))) in rer, axis=1)
    w60 = pd.to_numeric(main_scan.worst60_ratio, errors="coerce")
    beh = {sc: tide_behaviour(sc) for sc in scens}
    has_tide = "tide" in set(runs.strategy)
    tvr, tv_beh = None, {}
    if has_tide and os.path.exists(E22_TV):
        tvr = pd.read_csv(E22_TV)
        tvr = tvr[tvr.strategy == "tide_r800e03"]
        tv_beh = {sc: tide_behaviour(sc, "abl_tidevar", "tide_r800e03") for sc in scens}

    def thr_table(df):
        t = {(sc, p): float(df[(df.scenario == sc) & (df.strategy == p)].B_thr.mean()) for sc in scens for p in set(df.strategy)}
        best = {sc: max((p for (s_, p) in t if s_ == sc), key=lambda p: t[(sc, p)]) for sc in scens}
        worst = {p: min(t[(sc, p)] / t[(sc, best[sc])] for sc in scens) for p in set(df.strategy)}
        return t, best, worst
    ro = runs.copy()
    for a in anom.itertuples():
        m = (ro.scenario == a.scenario) & (ro.strategy == a.strategy) & (ro.seed == a.seed)
        ro.loc[m, "B_thr"] = a.B_thr
    t_new, b_new, w_new = thr_table(runs)
    t_old, b_old, w_old = thr_table(ro)
    alt_new = min((p for p in w_new if p != "v6"), key=lambda p: 1 - w_new[p])
    alt_old = min((p for p in w_old if p != "v6"), key=lambda p: 1 - w_old[p])
    dec = pd.read_csv(E22_DEC, dtype={"rep": str})
    dec_w = {(r.scenario, r.strategy, int(r.rep.lstrip("r"))): float(r.worst60_ratio) for r in dec.itertuples() if int(r.flagged)}
    assert set(dec_w) == rer, "reruns must be exactly the runs the decision scan flagged"
    nflag = len(rer)
    C, Z, M = "m1_code", "m1_zh", "m2_code"
    if has_tide and all(sc in scens for sc in (C, Z, M)):
        ft = {sc: pm(sc, "v6", "tide") for sc in scens}
        fmt = lambda x: f"{x['m']:+.1f}%（逐次 {' / '.join(f'{v:+.1f}' for v in x['d'])}）"
        env_note = ""
        same_all = all(b_new[sc] == b_old[sc] for sc in scens) and abs(w_old["v6"] - w_new["v6"]) < 5e-4 and alt_old == alt_new \
            and abs(w_old[alt_old] - w_new[alt_new]) < 5e-4
        tt_o = ro[(ro.scenario == M) & (ro.strategy == "tide")].set_index("seed").B_thr
        vv = runs[(runs.scenario == M) & (runs.strategy == "v6")].set_index("seed").B_thr
        moe_o = float(np.mean([(vv[i] - tt_o[i]) / tt_o[i] * 100 for i in sorted(vv.index)]))
        sens_note = ((f"每个场景的最好策略、完整控制器离最好的差距（{(1 - w_new['v6']) * 100:.1f}%）、最稳替代策略（{STRAT_ZH[alt_new]}）的差距"
                      f"（{(1 - w_new[alt_new]) * 100:.1f}%）都不变——被换掉的运行不决定任何一个最差场景；" if same_all else
                      f"最好策略或最差场景有变化（完整控制器 {(1 - w_old['v6']) * 100:.1f}% vs 主表 {(1 - w_new['v6']) * 100:.1f}%），见表 4；")
                     + f"变的只有 MoE 上完整控制器对 TIDE 式的优势：主表 {ft[M]['m']:+.1f}%，换回原始运行 {moe_o:+.1f}%。")
        env_lo = env_hi = float("nan")
        if envc is not None:
            dlist = []
            for r_ in envc.itertuples():
                ref_ = o72[(o72.scenario == r_.scenario) & (o72.strategy == r_.strategy) & (o72.seed == r_.seed)]
                if len(ref_) == 1:
                    dlist.append((float(r_.B_thr) / float(ref_.B_thr.iloc[0]) - 1) * 100)
            env_lo, env_hi = min(dlist), max(dlist)
        if envc is not None:
            ev6 = envc[(envc.scenario == C) & (envc.strategy == "v6")].set_index("seed").B_thr
            tt = runs[(runs.scenario == C) & (runs.strategy == "tide")].set_index("seed").B_thr
            if len(ev6) == 3:
                de = [(ev6[i] - tt[i]) / tt[i] * 100 for i in sorted(ev6.index)]
                env_note = (f"TIDE 式运行用的是重建环境、新节点；把完整控制器也在重建环境里重跑 3 次再比，8B 代码是 {np.mean(de):+.1f}%"
                            f"（逐次 {' / '.join(f'{v:+.1f}' for v in de)}，{'同号' if (min(de) > 0 or max(de) < 0) else '异号'}），"
                            f"和冻结矩阵比是 {ft[C]['m']:+.1f}%——论文两个都报。")
        ctx["e22_notes"] = [
            f"① TIDE 式基线在可适应的场景接近完整控制器：完整控制器 − TIDE 式 = 8B 代码 {fmt(ft[C])}、MoE {fmt(ft[M])}"
            f"（MoE 三次{'同号' if ft[M]['same'] else '异号，在噪声内'}）；在中文上失败：{fmt(ft[Z])}。" + env_note,
            f"① 中文失败的机制：TIDE 只在投机开着时测得到接受率，关投机后长平均更新得很慢、还带着对话时期的水平，换头后 8 s 探测看到的低接受率被当成漂移："
            f"B 段 {sum(x['rejB'] for x in beh[Z])} 次退回里有 {sum(x['rejdrift'] for x in beh[Z])} 次在 2 秒内被漂移触发重新开训"
            f"（{sum(1 for x in beh[Z] if x['rejB'] == 0)} 次重复一次都没退回、整段在训），B 段 {np.mean([x['trainB'] for x in beh[Z]]):.0%} 的时间在训，"
            f"而这些头没有一个能过盈亏线——它不问\"训出来的头会不会被用上\"。",
            "① 代码场景提前停训的机制：固定每 200 条一轮，训到 400 条以后每轮的接受率增量（校准曲线 ≤0.03）落在 8 s 在线比较的噪声范围内"
            "（没换头时同一统计量 5–95% 在 ±0.03），第一次\"没涨\"就退回并停；完整控制器的导出间隔倍增，后面的增量仍然测得出来。",
            f"② 干扰扫描：规则先写定再扫（干净样本 = 槽位满、训练器空闲、无切换无换头；参照 = 其它运行同接受长度的中位数；连续 60 s 中位比值 < 0.85 标记），"
            f"标记 {nflag} 次：TIDE MoE 第 2 次与冻结矩阵里仅开关 ×2、简单规则 ×2（最差 60 s 比值 {min(dec_w.values()):.2f}–{max(dec_w.values()):.2f}），"
            "完整控制器一次都没有。仅开关那两次的下降紧跟在关投机之后，但全部关投机事件里第一分钟与第二分钟的吞吐中位比值约 1.00，切换本身不解释它。",
            f"② 换节点重跑（表 3），主表用重跑；因为被标记的全是基线，替换只会缩小控制器的优势。换回原始运行：" + sens_note,
            f"③ 环境重建核对：同一次运行在重建环境里的 B 段吞吐与冻结矩阵相差 {env_lo:+.1f}% 到 {env_hi:+.1f}%（表 6），与节点间的正常差别同量级。"]
        if tvr is not None:
            tvd = {}
            for sc in scens:
                fu = runs[(runs.scenario == sc) & (runs.strategy == "v6")].set_index("seed").B_thr
                tv_ = tvr[tvr.scenario == sc].set_index("seed").B_thr
                tvd[sc] = [(fu[i] - tv_[i]) / tv_[i] * 100 for i in sorted(tv_.index)]
            fz = lambda d: f"{np.mean(d):+.1f}%（逐次 {' / '.join(f'{v:+.1f}' for v in d)}）"
            zh_pos = all(v > 0 for v in tvd[Z])
            ctx["e22_notes"].insert(3,
                f"① 参数敏感性（表 2b）：把每轮改成 800 条、漂移阈值改成 0.3 再跑 9 次，完整 − 它 = 8B 代码 {fz(tvd[C])}、中文 {fz(tvd[Z])}、MoE {fz(tvd[M])}；"
                f"中文 B 段在训的时间 {np.mean([y['trainB'] for y in tv_beh[Z]]):.0%}（默认参数 {np.mean([y['trainB'] for y in beh[Z]]):.0%}），"
                f"漂移触发每次 {np.mean([y['drift'] for y in tv_beh[Z]]):.1f} 次（默认 {np.mean([y['drift'] for y in beh[Z]]):.1f}）。"
                + (f"中文上三次仍都是完整控制器更快，但平均差距从默认参数的 {ft[Z]['m']:+.1f}% 缩小到 {np.mean(tvd[Z]):+.1f}%：阈值大了漂移不再重开训练，"
                   "可每轮 800 条的新头仍会让接受率涨一点，保留规则照样让它训掉约一半的饱和段。中文的失败不靠我们定的参数，但幅度随参数变。"
                   if zh_pos else
                   f"中文上不再三次都是完整控制器更快（平均 {np.mean(tvd[Z]):+.1f}%），论文要如实写成依赖参数。"))
        ctx["e22_res"] = (f"完整控制器 − TIDE 式：8B 代码 {ft[C]['m']:+.1f}%、中文 {ft[Z]['m']:+.1f}%、MoE {ft[M]['m']:+.1f}%；"
                          f"扫描标记 {nflag} 次（全是基线），换节点重跑")
        ctx["e22_verdict"] = "TIDE 式在可适应场景接近控制器、在中文失败（不问头会不会被用上）；被干扰的只有基线，用重跑只会缩小控制器的优势"
        if tvr is not None:
            ctx["e22_res"] += f"；换参数（每轮 800、阈值 0.3）后中文仍 {np.mean(tvd[Z]):+.1f}%（三次{'都正' if all(v > 0 for v in tvd[Z]) else '有正有负'}）"
        ctx["e22_word"] = "成立"
    lead = (f"冻结矩阵之后补的三件事，都在同一个冻结 checkpoint 上：① TIDE 式基线（3 场景 × 3 次），以及换一组参数（每轮 800 条、漂移阈值 0.3）再跑 9 次的敏感性检验；② 一次 TIDE 运行吞吐突然掉了约三成，"
            f"于是先写定规则、再扫全部 {len(main_scan) - int(is_rer.sum()) + len(orig_scan)} 次运行，被标记 {nflag} 次（全是基线，没有完整控制器），"
            f"全部换节点重跑，主表用重跑；③ 冻结环境被 OSC scratch 清理删坏后按原版本重建，并用重复运行核对。")

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], [20, 16, 8, 12, 11, 11, 11, 11, 11, 11, 11, 20])
        s.title_(f"{e['id']} · {e['name']}", lead)
        s.section("实验配置")
        s.kv([("为什么补", "审稿人大概率会问：和 TIDE（Park et al., SC'26，按接受率漂移触发训练）这类规则比怎么样？TIDE 用独立训练卡，这里只比较\"何时训练\"的逻辑。"),
              ("TIDE 式基线", STRAT_DESC["tide"] + " 原文没给的数值（两条滑动平均的权重、ε、每轮样本数）是我们定的。"),
              ("冻结文件", "controller.py、hotswap_run.py、train_head.py、abl_runner.py、abl_matrix.py 等一个没改（集群上核对 hash 与 MANIFEST.json 一致）；"
                         "新增 controller_tide.py / hotswap_run_tide.py / abl_runner_any.py / abl_matrix_tide.py 只服务这个基线。提示池、种子、引擎配置、显存预留、段长与 E19 相同。"),
              ("环境重建", "10-24 OSC scratch 的 60 天清理删掉了共享环境 bsr312 里 122 个包的源文件（有 .pyc 时不读 .py，访问时间不更新）。按原 203 个包的确切版本 "
                         "pip --no-deps 重建为 ss312，再拷入 6 个打补丁的 vLLM 文件（hash 一致）。TIDE 运行、重跑、核对运行都用 ss312。"),
              ("干扰扫描规则（扫描前写定）", "B 段每 5 s 一个样本；\"干净样本\" = ≥90% 请求槽在跑、前 10 s 训练器完全空闲、前 10 s 没有开关切换也没有换头；"
                                     "参照 = 同场景、同投机状态的其它运行里接受长度最接近的 50 个干净样本的中位吞吐；某段连续 60 s 干净样本的中位比值 < 0.85 就标记。"
                                     "被标记的运行原样移到 runs/abl_anomaly/，同配置换节点重跑，主表用重跑（代码 scan_anomaly.py）。")])
        if has_tide:
            s.section("结果：TIDE 式基线")
            rows = []
            for sc in scens:
                b = beh[sc]
                q = runs[(runs.scenario == sc) & (runs.strategy == "tide")]
                rows.append((sc, {"sc": SCEN_ZH[sc], "thr": float(q.B_thr.mean()), "sp": V(float(q.spec_on_B.mean()), fmt=PCTP),
                                  "tr": V(float(np.mean([x["trainB"] for x in b])), fmt=PCTP), "duty": V(float(q.duty_B.mean()), fmt=F3),
                                  "rd": V(float(np.mean([x["rounds"] for x in b])), fmt=F1),
                                  "kir": f"{np.mean([x['kept'] for x in b]):.1f} / {np.mean([x['inconcl'] for x in b]):.1f} / {np.mean([x['rejected'] for x in b]):.1f}",
                                  "dr": V(float(np.mean([x["drift"] for x in b])), fmt=F1), "tk": V(float(q.train_tok.mean()) / 1e6, fmt=F2),
                                  "acc": V(float(q.acc_last.mean()) if q.acc_last.notna().any() else None, fmt=F2)}))
            s.table([("sc", "场景", None), ("thr", "B 吞吐\ntok/s", INT), ("sp", "B 段投机\n开着", PCTP), ("tr", "B 段在训\n的时间", PCTP),
                     ("duty", "B 段平均\n占空比", F3), ("rd", "评估轮数", F1), ("kir", "保留 / 无法判定 / 退回", None, 2), ("dr", "漂移触发", F1),
                     ("tk", "训练 token\nM（全程）", F2), ("acc", "B 末接受\n长度", F2)], rows, caption="表 1　TIDE 式基线做了什么（三次均值）")
            rows = []
            for sc in scens:
                for a_, b_, why in [("v6", "tide", "完整控制器 − TIDE 式"), ("tide", "switch", "TIDE 式 − 仅开关（训练逻辑值多少）"),
                                    ("tide", "fixed800", "TIDE 式 − 已知切换点的固定训练"), ("tide", "static", "TIDE 式 − 静态投机")]:
                    d = pm(sc, a_, b_)
                    rows.append((f"{sc}{a_}{b_}", {"sc": SCEN_ZH[sc], "cmp": why, "m": V(d["m"] / 100, fmt=PCT), "sd": V(d["s"] / 100, fmt=PCTP),
                                                   "ps": " / ".join(f"{x:+.1f}" for x in d["d"]), "ss": "同号" if d["same"] else "异号"}))
            s.table([("sc", "场景", None), ("cmp", "比较（B 段吞吐，按次配对）", None, 3), ("m", "配对差\n均值", PCT), ("sd", "标准差", PCTP),
                     ("ps", "逐次差值 %", None, 2), ("ss", "符号", None)], rows, caption="表 2　TIDE 式基线与其它策略的配对比较（主表口径：用重跑）")
            if tvr is not None:
                rows = []
                for sc in scens:
                    fu = runs[(runs.scenario == sc) & (runs.strategy == "v6")].set_index("seed").B_thr
                    td = runs[(runs.scenario == sc) & (runs.strategy == "tide")].set_index("seed").B_thr
                    tv_ = tvr[tvr.scenario == sc].set_index("seed").B_thr
                    for lab, b, x in (("每轮 200、阈值 0.1（默认）", beh[sc], td), ("每轮 800、阈值 0.3", tv_beh[sc], tv_)):
                        dd = [(fu[i] - x[i]) / x[i] * 100 for i in sorted(x.index)]
                        rows.append((f"{sc}{lab}", {"sc": SCEN_ZH[sc], "set": lab, "thr": float(x.mean()), "fv": V(float(np.mean(dd)) / 100, fmt=PCT),
                                                    "ps": " / ".join(f"{v:+.1f}" for v in dd), "tr": V(float(np.mean([y["trainB"] for y in b])), fmt=PCTP),
                                                    "rd": V(float(np.mean([y["rounds"] for y in b])), fmt=F1),
                                                    "kir": f"{np.mean([y['kept'] for y in b]):.1f} / {np.mean([y['inconcl'] for y in b]):.1f} / {np.mean([y['rejected'] for y in b]):.1f}",
                                                    "dr": V(float(np.mean([y["drift"] for y in b])), fmt=F1)}))
                s.table([("sc", "场景", None), ("set", "参数", None, 2), ("thr", "B 吞吐\ntok/s", INT), ("fv", "完整 −\n它", PCT), ("ps", "逐次 %", None, 2),
                         ("tr", "B 段在训", PCTP), ("rd", "评估\n轮数", F1), ("kir", "保留/无法判定/退回", None), ("dr", "漂移\n触发", F1)], rows,
                        caption="表 2b　TIDE 式参数敏感性：每轮 800 条请求、漂移阈值 0.3，其余不变（同一 checkpoint、同样的提示和种子，跑在 runs/abl_tidevar/，不进主表）")
        s.section("结果：干扰扫描与重跑")
        rows = []
        for (sc_, st_, sd_) in sorted(rer):
            a = anom[(anom.scenario == sc_) & (anom.strategy == st_) & (anom.seed == sd_)].iloc[0]
            n = runs[(runs.scenario == sc_) & (runs.strategy == st_) & (runs.seed == sd_)].iloc[0]
            ns = main_scan[(main_scan.scenario == sc_) & (main_scan.strategy == st_) & (main_scan.rep == f"r{sd_}")].iloc[0]
            rows.append((f"{sc_}{st_}{sd_}", {
                "sc": SCEN_ZH[sc_], "st": STRAT_ZH[st_], "rp": V(sd_, fmt=INT),
                "h0": str(a.host).split(".")[0], "w0": V(dec_w[(sc_, st_, sd_)], fmt=F3), "t0": float(a.B_thr),
                "h1": str(n.host).split(".")[0], "w1": V(float(ns.worst60_ratio), fmt=F3), "t1": float(n.B_thr),
                "ch": F("={t1}{r}/{t0}{r}-1", float(n.B_thr) / float(a.B_thr) - 1, PCT)}))
        s.table([("sc", "场景", None), ("st", "策略", None), ("rp", "第几次", INT), ("h0", "原节点", None), ("w0", "原始\n最差 60 s", F3),
                 ("t0", "原始 B 吞吐\ntok/s", INT), ("h1", "重跑节点", None), ("w1", "重跑\n最差 60 s", F3), ("t1", "重跑 B 吞吐\ntok/s", INT),
                 ("ch", "变化", PCT)], rows, caption=f"表 3　被标记的 {nflag} 次运行与重跑（标记线 0.85；原始的分数来自决定重跑的那次扫描 anomaly_scan_decision.csv；原始运行在 data/runs/abl_anomaly/）",
                note=f"重跑后主表里所有运行的最差 60 s 比值 ≥ {float(w60.min()):.2f}；全程训练的\"固定训练（全程 0.5）\"训练器从不空闲 60 s，无法打分"
                     f"（{int(w60.isna().sum())} 次）。")
        rows = []
        for p in sorted(w_new, key=lambda p: -w_new[p]):
            rows.append((p, {"k": V(STRAT_ZH[p], bold=(p == "v6")), "wn": V(w_new[p], fmt=F3), "wo": V(w_old[p], fmt=F3),
                             "d": F("={wn}{r}-{wo}{r}", w_new[p] - w_old[p], F3)}))
        s.table([("k", "策略", None, 2), ("wn", "最差场景 / 该场景最好\n（主表：用重跑）", F3, 2), ("wo", "同上\n（换回原始运行）", F3, 2), ("d", "差", F3)],
                rows, caption="表 4　敏感性：把重跑换回原始运行，结论变不变",
                note=f"每个场景的最好策略：{'不变' if all(b_new[sc] == b_old[sc] for sc in scens) else '有变化'}"
                     f"（{'、'.join(f'{SCEN_ZH[sc]} {STRAT_ZH[b_new[sc]]}' for sc in scens)}）；最稳的替代策略 主表 {STRAT_ZH[alt_new]} "
                     f"差 {(1 - w_new[alt_new]) * 100:.1f}%，换回原始 {STRAT_ZH[alt_old]} 差 {(1 - w_old[alt_old]) * 100:.1f}%。")
        rows = []
        for r in scan.sort_values(["root", "scenario", "strategy", "rep"]).itertuples():
            rows.append((f"{r.root}{r.scenario}{r.strategy}{r.rep}", {
                "root": {"abl": "主表", "abl_anomaly": "原始（已换下）", "abl_envcheck": "环境核对", "abl_tidevar": "TIDE 敏感性"}.get(r.root, r.root),
                "sc": SCEN_ZH[r.scenario], "st": STRAT_ZH[r.strategy], "rp": str(r.rep), "n": V(int(r.n_clean), fmt=INT),
                "md": V(float(r.median_ratio) if str(r.median_ratio) not in ("", "nan") else None, fmt=F3),
                "w": V(float(r.worst60_ratio) if str(r.worst60_ratio) not in ("", "nan") else None, fmt=F3),
                "fl": V("标记", status="失效") if int(r.flagged) else ""}))
        s.table([("root", "来源", None), ("sc", "场景", None), ("st", "策略", None), ("rp", "次", None), ("n", "干净样本数", INT),
                 ("md", "中位比值", F3), ("w", "最差 60 s", F3), ("fl", "标记", None)], rows, caption="表 5　全部运行的扫描分数（results/abl_all/anomaly_scan.csv）")
        if envc is not None:
            s.section("结果：环境重建核对")
            rows = []
            for r in envc.itertuples():
                ref = o72[(o72.scenario == r.scenario) & (o72.strategy == r.strategy) & (o72.seed == r.seed)]
                if len(ref) != 1:
                    continue
                rows.append((f"{r.scenario}{r.strategy}{r.seed}", {"sc": SCEN_ZH[r.scenario], "st": STRAT_ZH[r.strategy], "rp": V(int(r.seed), fmt=INT),
                                                                  "h0": str(ref.host.iloc[0]).split(".")[0], "t0": float(ref.B_thr.iloc[0]),
                                                                  "h1": str(r.host).split(".")[0], "t1": float(r.B_thr),
                                                                  "ch": F("={t1}{r}/{t0}{r}-1", float(r.B_thr) / float(ref.B_thr.iloc[0]) - 1, PCT)}))
            s.table([("sc", "场景", None), ("st", "策略", None, 2), ("rp", "第几次", INT), ("h0", "冻结矩阵\n节点", None), ("t0", "冻结矩阵\nB 吞吐", INT),
                     ("h1", "重建环境\n节点", None), ("t1", "重建环境\nB 吞吐", INT), ("ch", "差", PCT)], rows,
                    caption="表 6　同一次运行在冻结环境与重建环境下的 B 段吞吐",
                    note="完整控制器（v6）的三次也在重建环境里重跑了一遍，用来在同一环境下和 TIDE 式比较；节点间本身就有约 1–2% 的差。")
        s.section("怎么读")
        s.bullets(ctx.get("e22_notes", ["（结论句在全部运行跑完后填写）"]))
        s.source("data/runs/abl/<场景>/tide/r<次>/；data/runs/abl_anomaly/（被标记的原始运行，含 ANOMALY_NOTE.txt）；data/runs/abl_envcheck/；"
                 "results/abl_all/（主表口径，81 次）、results/abl_anomaly/、results/abl_envcheck/；code/specsched/scan_anomaly.py、controller_tide.py；"
                 "results/summaries/2026-10-26_TIDE式基线.md")
    return meta(ctx, "E22", short="TIDE基线与重跑", name="冻结后补充：TIDE 式基线、干扰扫描与重跑、环境重建核对",
                q="和按接受率漂移触发训练的 TIDE 式规则比怎么样？节点干扰会不会影响结论？重建的环境和冻结环境一样吗？",
                cfg=f"同一冻结 checkpoint；TIDE 式 3 场景 × 3 次 + 参数敏感性 9 次；扫描规则先写定；{nflag} 次被标记的运行换节点重跑；环境按原 203 个包版本重建",
                res=ctx.get("e22_res", "见明细"), verdict=ctx.get("e22_verdict", "见明细"), word=ctx.get("e22_word", "成立"), status="有效", build=build)


SGL_CHK = f"{ROOT}/data/runs/sgl_check"
SGL_ABL = f"{ROOT}/results/sgl_abl"


def e23(ctx):
    func = {m: jload(f"{SGL_CHK}/{m}/func/report.json") for m in ("m1", "m2")}
    perf = {m: (jload(f"{SGL_CHK}/{m}/perf/report.json") or {}).get("runs", {}) for m in ("m1", "m2")}
    perf_v1 = (jload(f"{SGL_CHK}/m1/perf_v1_draftextend/report.json") or {}).get("runs", {})
    mods = {m: (jload(f"{HERE}/sched/model_sgl_{m}.json"), jload(f"{HERE}/sched/{'model' if m == 'm1' else 'model_m2'}.json")) for m in ("m1", "m2")}
    man = jload(f"{ROOT}/data/runs/sgl/MANIFEST.json")
    has_m = os.path.exists(f"{SGL_ABL}/abl_runs.csv")
    runs = pd.read_csv(f"{SGL_ABL}/abl_runs.csv") if has_m else None
    SP = ["static", "nospec", "switch", "tide", "v6"]
    complete = has_m and len(runs) == 45 and set(runs.strategy) == set(SP)

    def paired(sc, a, b):
        x = runs[(runs.scenario == sc) & (runs.strategy == a)].set_index("seed").B_thr
        y = runs[(runs.scenario == sc) & (runs.strategy == b)].set_index("seed").B_thr
        d = [(x[i] - y[i]) / y[i] * 100 for i in sorted(set(x.index) & set(y.index))]
        return d

    if complete:
        thr = {(sc, p): float(runs[(runs.scenario == sc) & (runs.strategy == p)].B_thr.mean()) for sc in SCEN_ZH for p in SP}
        best = {sc: max(SP, key=lambda p: thr[(sc, p)]) for sc in SCEN_ZH}
        worst = {p: min(thr[(sc, p)] / thr[(sc, best[sc])] for sc in SCEN_ZH) for p in SP}
        fd = {sc: {o: paired(sc, "v6", o) for o in ("static", "nospec", "switch", "tide")} for sc in SCEN_ZH}
        SHORT_SC = {"m1_code": "8B 代码", "m1_zh": "8B 中文", "m2_code": "MoE 代码"}
        loaded = {sc: max(("static", "switch", "tide"), key=lambda p: thr[(sc, p)]) for sc in SCEN_ZH}
        gap = {sc: np.mean(paired(sc, "v6", loaded[sc])) for sc in SCEN_ZH}
        res = ("完整控制器 vs 静态投机 " + "，".join(f"{SHORT_SC[sc]} {np.mean(fd[sc]['static']):+.1f}%" for sc in SCEN_ZH)
               + f"；与每个场景最好的带草稿策略差 {' / '.join(f'{gap[sc]:+.1f}%' for sc in SCEN_ZH)}；最差场景比值 完整 {worst['v6']:.3f}、"
               + f"仅开关 {worst['switch']:.3f}、TIDE 式 {worst['tide']:.3f}、纯解码 {worst['nospec']:.3f}")
        word = "成立" if worst["v6"] >= max(worst[p] for p in SP if p != "v6") else "部分成立"
        verdict = ("决策迁移成立（中文关投机不训、代码训练），与最好的带草稿策略持平；但 SGLang 关投机比纯解码服务器慢约 15%，"
                   f"中文上纯解码领先完整控制器 {np.mean(paired('m1_zh', 'nospec', 'v6')):.0f}%，最差场景比值完整控制器不再领先")
    else:
        res, verdict, word = "矩阵进行中（见明细）", "进行中", "参考"

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], [24, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 16])
        s.title_(f"{e['id']} · {e['name']}",
                 "把 vLLM 研究的三个引擎机制（实时导出、草稿热替换、运行时投机开关）移植到 SGLang 0.5.10.post1，原样复用冻结的 harness / "
                 "控制器 / 训练器 / 提示 / 种子；SGLang 单独校准、单独冻结 checkpoint，跑 5 策略 × 3 场景 × 3 次。")
        s.section("配置")
        s.kv([("引擎", "SGLang 0.5.10.post1，默认 EAGLE-3 路径（spec v1，投机时 overlap 调度器关闭），fa3，page 1；环境 env/sgl0510（torch 2.9.1）"),
              ("移植方式", "运行时补丁 code/specsched/sgl/specsched_sgl.py（不改安装包）；/metrics 用 vLLM 0.27.1 指标名，/collective_rpc 走 /dev/shm 邮箱"),
              ("投机关时", "目标模型走启动时额外捕获的普通 decode CUDA graph，记账同\"0 接受的验证\"；草稿完全不跑（同 vLLM），重开第一步补算一次"),
              ("显存", "vLLM util u → SGLang mem_fraction_static = u − 非静态显存/总显存（8B 8.0/1.75 GiB、MoE 6.5/1.25 GiB，有/无草稿；实测 +余量）"),
              ("校准", "同脚本（c1_run / frontier_run 原样导入）；S0 = 投机服务器关掉投机（运行时 z=0 的真实代价）；学习曲线沿用 vLLM 研究的"),
              ("冻结", (f"runs/sgl/MANIFEST.json（{man['generated'][:16]}）；复用的 {len(man['reused'])} 个冻结文件哈希与 vLLM MANIFEST 一致"
                       if man else "未冻结"))])
        s.section("功能检查（作业 15965864 / 15966083）")
        rows = []
        for m, nm in (("m1", "Qwen3-8B"), ("m2", "Qwen3-30B-A3B")):
            f_ = func[m]
            if not f_:
                continue
            g, sw, rl, tp = f_["greedy_on_vs_off"], f_["switch_under_load"], f_["reload"], f_["tap"]
            aux = tp["hf_aux"]["per_request"]
            rows.append((m, {"m": nm, "g": f"{g['identical']}/{g['of']}", "fl": V(sw["flips"], fmt=INT), "cp": f"{sw['completed']}/64",
                             "a0": V(rl["factory_start"]["acc_len"], fmt=F3), "at": V(rl["trained_code_N800"]["acc_len"], fmt=F3),
                             "az": V(rl["fc_zero"]["acc_len"], fmt=F3), "ab": V(rl["factory_again"]["acc_len"], fmt=F3),
                             "tq": f"{tp['finish']}/{tp['requests']}，断档 {tp['gap']}", "cos": V(min(v["cos_min"] for v in aux.values()), fmt=F4),
                             "un": "是" if f_["target_unchanged_after_reloads"] else "否"}))
        s.table([("m", "目标", None), ("g", "贪心 开/关\n一致", None), ("fl", "负载下\n翻转次数", INT), ("cp", "完成", None),
                 ("a0", "接受：出厂", F3), ("at", "代码训练头\nN800", F3), ("az", "fc 置零", F3), ("ab", "再换回出厂", F3),
                 ("tq", "导出：FINISH/请求", None, 2), ("cos", "aux vs HF\n最低余弦", F4), ("un", "换头后目标\n输出不变", None)],
                rows, caption="表 1　移植功能检查")
        s.section("关投机这一步有多贵（代码提示，tok/s）")
        rows = []
        for m, nm in (("m1", "Qwen3-8B"), ("m2", "Qwen3-30B-A3B")):
            p = perf[m]
            for c in (1, 16, 64, 256):
                if f"on_C{c}" not in p:
                    continue
                g = lambda k: p.get(f"{k}_C{c}", {}).get("out_tok_s")
                r = {"m": nm if c == 1 else "", "c": V(c, fmt=INT), "on": V(g("on"), fmt=INT), "acc": V(p[f"on_C{c}"]["acc_len"], fmt=F2),
                     "off": V(g("off"), fmt=INT), "pl": V(g("plain"), fmt=INT), "no": V(g("plain_nooverlap"), fmt=INT),
                     "v1": V(perf_v1.get(f"off_C{c}", {}).get("out_tok_s") if m == "m1" else None, fmt=INT)}
                if g("plain"):
                    r["d1"] = F("={off}{r}/{pl}{r}-1", g("off") / g("plain") - 1, fmt=PCT)
                if g("plain_nooverlap"):
                    r["d2"] = F("={no}{r}/{pl}{r}-1", g("plain_nooverlap") / g("plain") - 1, fmt=PCT)
                rows.append((f"{m}{c}", r))
        s.table([("m", "目标", None), ("c", "并发", INT), ("on", "投机开", INT), ("acc", "接受长度", F2), ("off", "投机关\n（定版）", INT),
                 ("v1", "投机关（初版：\n关时续算草稿）", INT), ("pl", "普通服务器", INT), ("no", "普通服务器\n关 overlap", INT),
                 ("d1", "关投机 vs\n普通服务器", PCT), ("d2", "关 overlap vs\n普通服务器", PCT)], rows, caption="表 2　同一服务器开 / 关投机与不带草稿的服务器")
        s.note("C=256 时 8B 关投机比普通服务器慢约 15%：约 10 个百分点是 SGLang 带草稿时不能用 overlap 调度（普通服务器关掉 overlap 也慢这么多），"
               "约 5 个百分点是移植路径的记账；MoE 上关投机与\"普通服务器关 overlap\"几乎相同。vLLM 里关投机 ≈ 普通服务器（runs/switch/m1_code_v2：C=256 关投机比开高 10.5%）。"
               "初版（关时让草稿每步续算）在 C=256 比普通服务器慢 20%、比开着投机还慢 16%，校准前已换掉。")
        s.section("校准对比（S0 = 投机服务器关掉投机；vLLM 的 S0 = 普通服务器）")
        rows = []
        for m, nm in (("m1", "Qwen3-8B"), ("m2", "Qwen3-30B-A3B")):
            sg_, vl = mods[m]
            if not sg_:
                continue
            for c in sg_["C"]:
                a, b = sg_["spec_step"][str(c)], vl["spec_step"][str(c)]
                ga, gb = a["acc_meas"] * a["T_base"] / a["T_spec"], b["acc_meas"] * b["T_base"] / b["T_spec"]
                rows.append((f"{m}{c}", {"m": nm if c == 1 else "", "c": V(c, fmt=INT), "tb": V(a["T_base"], fmt=F2), "ts": V(a["T_spec"], fmt=F2),
                                         "ga": V(ga, fmt=F3), "ka": V(sg_["kappa"][str(c)], fmt=F3), "vtb": V(b["T_base"], fmt=F2),
                                         "gb": V(gb, fmt=F3), "kb": V(vl["kappa"][str(c)], fmt=F3)}))
        s.table([("m", "目标", None), ("c", "并发", INT), ("tb", "SGLang\nT_base ms", F2), ("ts", "SGLang\nT_spec ms", F2), ("ga", "SGLang 增益\n（对话出厂头）", F3),
                 ("ka", "SGLang κ", F3), ("vtb", "vLLM\nT_base ms", F2), ("gb", "vLLM 增益", F3), ("kb", "vLLM κ", F3)], rows,
                caption="表 3　单步时间、出厂头增益与训练干扰系数")
        rows = []
        for m, dom, nm in (("m1", "code", "8B 代码"), ("m1", "zh", "8B 中文"), ("m2", "code", "MoE 代码")):
            sg_, vl = mods[m]
            if not sg_ or dom not in sg_.get("frontier", {}):
                continue
            for c in ("1", "16", "256"):
                ps = sorted(sg_["frontier"][dom]["fit"][c]["points"])
                pv = {h: g for _a, g, h in vl["frontier"][dom]["fit"][c]["points"]}
                for a_, g_, h in ps:
                    rows.append((f"{m}{dom}{c}{h}", {"d": nm if (c == "1" and h == ps[0][2]) else "", "c": V(int(c), fmt=INT), "h": h.replace("S1_", ""),
                                                     "a": V(a_, fmt=F2), "gs": V(g_, fmt=F3), "gv": V(pv.get(h), fmt=F3)}))
        s.table([("d", "场景", None), ("c", "并发", INT), ("h", "草稿头", None), ("a", "接受长度", F2), ("gs", "SGLang 增益\n（对关投机）", F3),
                 ("gv", "vLLM 增益\n（对普通服务器）", F3)], rows, caption="表 4　前沿：各头的接受长度与增益（接受长度两引擎逐位一致）")
        if complete:
            s.section("矩阵结果（B 段吞吐，3 次配对）")
            rows = []
            for sc in SCEN_ZH:
                for p in SP:
                    g = runs[(runs.scenario == sc) & (runs.strategy == p)]
                    r = {"sc": SCEN_ZH[sc] if p == "static" else "", "st": STRAT_ZH[p], "thr": V(float(g.B_thr.mean()), fmt=INT),
                         "sd": V(float(g.B_thr.std(ddof=1)), fmt=INT), "r": V(thr[(sc, p)] / thr[(sc, best[sc])], fmt=F3, bold=p == best[sc]),
                         "sp": V(float(g.spec_on_B.mean()), fmt=PCTP), "du": V(float(g.duty_B.mean()), fmt=F2), "sw": V(float(g.swaps_B.mean()), fmt=F1),
                         "acc": V(float(g.acc_last.mean()) if g.acc_last.notna().any() else None, fmt=F2)}
                    if p != "static":
                        d = paired(sc, p, "static")
                        r["d"] = V(np.mean(d) / 100, fmt=PCT)
                        r["ds"] = " / ".join(f"{x:+.1f}" for x in d)
                    rows.append((f"{sc}{p}", r))
            s.table([("sc", "场景", None), ("st", "策略", None), ("thr", "B 吞吐\ntok/s", INT), ("sd", "标准差", INT), ("r", "/ 该场景最好", F3),
                     ("d", "vs 静态投机\n（配对均值）", PCT), ("ds", "逐次 %", None, 2), ("sp", "B 段投机开", PCTP), ("du", "B 段占空比", F2),
                     ("sw", "B 段换头", F1), ("acc", "B 末接受", F2)], rows, caption="表 5　SGLang 矩阵（每格 3 次）")
            rows = []
            for sc in SCEN_ZH:
                r = {"sc": SCEN_ZH[sc]}
                for o in ("static", "nospec", "switch", "tide"):
                    r[o] = V(np.mean(fd[sc][o]) / 100, fmt=PCT)
                rows.append((sc, r))
            rows.append(("worst", {"sc": "最差场景 / 最好（越大越稳）", **{o: V(worst[o], fmt=F3) for o in ("static", "nospec", "switch", "tide")}}))
            s.table([("sc", "完整控制器 vs", None, 2), ("static", "静态投机", PCT), ("nospec", "纯解码", PCT), ("switch", "仅开关", PCT), ("tide", "TIDE 式", PCT)],
                    rows, caption=f"表 6　完整控制器对各策略（最后一行：各策略最差场景比值；完整控制器 {worst['v6']:.3f}）")
            s.note("怎么读：① 决策和 vLLM 一样——中文上结构性天花板判定关投机、几乎不训（B 段投机开 1%、占空比 0.01），代码和 MoE 上开着投机并在线训练"
                   "（B 段换头 8B 代码 4 次、MoE 4.7 次；B 末接受长度 2.69 / 2.95，静态投机 2.13 / 2.62）。② 在\"带草稿\"的策略里（静态 / 仅开关 / TIDE 式 / 完整）完整控制器每个场景都与最好的持平"
                   "（差 ≤0.7%）。③ 但 SGLang 投机路径没有 overlap 调度，关投机后仍比纯解码服务器慢约 15%（表 2），所以中文上纯解码领先约 40%，"
                   "所有带草稿的策略最差场景都在中文、比值 0.71–0.72；最差场景比值仅开关 0.721 略高于完整 0.716（中文第 3 次的节点慢 4–8%，在噪声内）。"
                   "④ 8B 代码上训好的头让完整控制器和 TIDE 式追平纯解码（−0.1%）；MoE 上纯解码比静态投机慢 32%，和 vLLM 一致。"
                   "⑤ 干扰扫描 0 次标记；中文的仅开关 / 完整第 3 次同一作业、同一节点（c0809），整段慢 4–8% 但没有 60 s 低于 0.85，按预先写定的规则保留。")
            rows = []
            for i, r in runs.sort_values(["scenario", "strategy", "seed"]).iterrows():
                rows.append((str(i), {"sc": SCEN_ZH[r.scenario], "st": STRAT_ZH[r.strategy], "seed": V(int(r.seed), fmt=INT), "job": str(r.job),
                                      "host": str(r.host), "thr": V(float(r.B_thr), fmt=INT), "p99": V(float(r.B_p99), fmt=F1),
                                      "sp": V(float(r.spec_on_B), fmt=PCTP), "du": V(float(r.duty_B), fmt=F2), "sw": V(int(r.swaps), fmt=INT),
                                      "all": V(float(r.all_thr), fmt=INT)}))
            s.table([("sc", "场景", None), ("st", "策略", None), ("seed", "次", INT), ("job", "作业", None), ("host", "节点", None), ("thr", "B 吞吐", INT),
                     ("p99", "B p99 ms", F1), ("sp", "投机开", PCTP), ("du", "占空比", F2), ("sw", "换头", INT), ("all", "全程吞吐", INT)], rows, caption="表 7　逐次运行")
        else:
            s.section("矩阵")
            s.para("矩阵在跑（45 次，作业名 ss-sglm）；跑完后重新生成本表。")
        s.source("data/runs/sgl_check（功能 / 速度）、data/runs/sgl/{c1,frontier}（校准）、code/specsched/sched/model_sgl_{m1,m2}.json、"
                 "results/sgl_abl（analyze_abl.py 从 data/runs/sgl/abl 生成）、results/summaries/2026-10-28_SGLang移植.md")
    return meta(ctx, "E23", short="SGLang移植", name="第二个引擎：SGLang 移植、校准与缩小矩阵",
                q="控制器的结论换一个引擎还成立吗？",
                cfg="SGLang 0.5.10.post1 运行时补丁；同 harness / 控制器 / 训练器 / 提示 / 种子；SGLang 单独校准与冻结；5 策略 × 3 场景 × 3 次",
                res=res, verdict=verdict, word=word, status="有效" if complete else "对照", build=build)


ABLATE_DIR = f"{ROOT}/results/abl_ablate"
MIX_DIR = f"{ROOT}/results/abl_mix"
ABL_CELLS = [("v6_novalue", "去掉回本规则（饱和时固定 0.5）"), ("v6_nostop", "去掉三条停止测试"), ("v6_noswitch", "去掉投机开关（投机常开）"),
             ("v6_nocurve", "去掉每域学习曲线与标签（无标签变体）")]
MIX_POL = [("static", "静态投机"), ("nospec", "纯解码"), ("switch", "仅开关"), ("tide", "TIDE 式"), ("v6", "完整控制器（混合标签 + 混合校准）"),
           ("v6_stale", "完整控制器，标签仍为对话（没发现切换）"), ("v6_nocurve", "完整控制器，无标签变体")]
MIX_SC = [("m1_mixcc", "代码 + 对话 50/50"), ("m1_mixzc", "中文 + 对话 50/50")]


def e24(ctx):
    ap = pd.read_csv(f"{ABLATE_DIR}/cell_pairs.csv"); ar = pd.read_csv(f"{ABLATE_DIR}/cell_runs.csv")
    has_mix = os.path.exists(f"{MIX_DIR}/cell_runs.csv") and len(pd.read_csv(f"{MIX_DIR}/cell_runs.csv")) == 42
    mp = pd.read_csv(f"{MIX_DIR}/cell_pairs.csv") if has_mix else None
    mr = pd.read_csv(f"{MIX_DIR}/cell_runs.csv") if has_mix else None
    mj = jload(f"{HERE}/sched/model_mix.json")

    def pair(df, sc, cell, ref):
        q = df[(df.scenario == sc) & (df.cell == cell) & (df.ref == ref)]
        return q.iloc[0] if len(q) == 1 else None

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], [30, 13, 13, 13, 13, 13, 13, 13, 13, 13, 16])
        s.title_(f"{e['id']} · {e['name']}",
                 "用户 11-01 问：收益到底来自盈利性判断、学习上限判断还是回本判断？真实流量混合多域、没有清晰切换点怎么办？两组实验都在 vLLM 冻结 checkpoint 上，"
                 "新文件不动冻结代码（controller_ablate.py 是冻结 controller.py 的拷贝 + --ablate 开关；hotswap_run_ablate.py 只替换 harness 启动的控制器进程）。")
        s.section("一、消融：每次去掉一条判断（3 场景 × 3 次，与冻结矩阵同提示同种子，按次与矩阵的 Full 配对）")
        rows = []
        for cell, name in ABL_CELLS:
            r = {"k": name}
            for sc in SCEN_ZH:
                q = pair(ap, sc, cell, "v6")
                r[f"d_{sc}"] = V(float(q.mean_pct) / 100, fmt=PCT); r[f"s_{sc}"] = V(float(q.std_pct) / 100, fmt=PCTP)
                r[f"p_{sc}"] = " / ".join(f"{x:+.1f}" for x in json.loads(q.per_seed))
            rows.append((cell, r))
        cols = [("k", "去掉的判断", None)]
        for sc in SCEN_ZH:
            short = SCEN_ZH[sc].split(" · ")[1] if " · " in SCEN_ZH[sc] else SCEN_ZH[sc]
            cols += [(f"d_{sc}", f"{short}\n vs Full 均值", PCT), (f"s_{sc}", "标准差", PCTP), (f"p_{sc}", "逐次 %", None)]
        s.table(cols, rows, caption="表 1　B 段吞吐相对完整控制器（同一请求序列）")
        rows = []
        for cell, name in ABL_CELLS:
            for sc in SCEN_ZH:
                x = ar[(ar.scenario == sc) & (ar.strategy == cell)]
                rows.append((f"{cell}{sc}", {"k": name if sc == "m1_code" else "", "sc": SCEN_ZH[sc], "thr": V(float(x.B_thr.mean()), fmt=INT),
                                             "du": V(float(x.duty_B.mean()), fmt=F2), "sp": V(float(x.spec_on_B.mean()), fmt=PCTP),
                                             "sw": V(float(x.swaps_B.mean()), fmt=F1), "acc": V(float(x.acc_last.mean()) if x.acc_last.notna().any() else None, fmt=F2),
                                             "vs_sw": V(float(pair(ap, sc, cell, "switch").mean_pct) / 100, fmt=PCT),
                                             "vs_np": V(float(pair(ap, sc, cell, "nospec").mean_pct) / 100, fmt=PCT)}))
        s.table([("k", "变体", None), ("sc", "场景", None), ("thr", "B 吞吐\ntok/s", INT), ("du", "B 段占空比", F2), ("sp", "B 段投机开", PCTP),
                 ("sw", "B 段换头", F1), ("acc", "B 末接受", F2), ("vs_sw", "vs 仅开关", PCT), ("vs_np", "vs 纯解码", PCT)], rows, caption="表 2　各变体在 B 段做了什么")
        s.note("怎么读：① 中文的收益全部来自开关：去掉开关 ≈ 静态投机（−43.8%，三次同号）；代码 / MoE 上开关从不触发。② 回本规则只在代码上值 2–3 个点："
               "饱和时固定 0.5 会一直训到天花板（占空比 0.16 vs Full 0.07），多付的干扰吞吐就是差距；MoE 上训练很快到顶，无差。③ 三条停止测试在饱和时冗余："
               "价值规则在到顶处本来就算出不值，中文上没有结构性天花板测试它也拒绝训；它们只影响空闲余量规则。④ 无标签变体（比例前沿 + 先验增益 0.3 被实测换头增益替换 + 收敛后 120 s 重试）"
               "在三个单域场景里与 Full 噪声内持平；中文第 1 次 −5.8% 是余量训出的头在 B 末导出、换头后 8 s 探测把投机打开在饱和中文上（随机事件）。⑤ 干扰扫描 36 次 0 标记。")
        s.section("二、混合流量：B 段是两个域各半的均匀混合（bench 按种子洗牌，做不了渐变比例）")
        if mj:
            rows = []
            for sc, nm in MIX_SC:
                dom = sc.split("_")[1]; d = mj["domains"][dom]; fr = mj["frontier"][dom]["fit"]["256"]; pts = {h: y for _a, y, h in fr["points"]}
                rows.append((sc, {"sc": nm, "a0": V(d["acc_curve"]["0"], fmt=F3), "amax": V(max(d["acc_curve"].values()), fmt=F3),
                                  "g0": V(pts["S1_N0"], fmt=F3), "g32": V(pts["S1_N3200"], fmt=F3), "be": V((1 - fr["beta"]) / fr["alpha"], fmt=F2),
                                  "tp": V(d.get("tokens_per_request"), fmt=INT)}))
            s.table([("sc", "混合", None), ("a0", "出厂头接受", F3), ("amax", "学习曲线天花板", F3), ("g0", "C=256 增益\n出厂头", F3),
                     ("g32", "C=256 增益\nN3200", F3), ("be", "C=256 盈亏接受长度", F2), ("tp", "token / 请求", INT)], rows,
                    caption="表 3　混合域的校准（sched/model_mix.json = 冻结 model.json 原样 + 混合域；c2 学习曲线 + 前沿各一次）")
            s.note("代码+对话：出厂头在饱和时过不了盈亏线、训到 N3200 才略过（1.07）——正好卡在盈亏边界；中文+对话：天花板 1.84 低于盈亏线 → 结构性天花板，应关投机不训。")
        if has_mix:
            rows = []
            for pol, pname in MIX_POL:
                r = {"k": pname}
                for sc, nm in MIX_SC:
                    x = mr[(mr.scenario == sc) & (mr.strategy == pol)]
                    r[f"t_{sc}"] = V(float(x.B_thr.mean()), fmt=INT)
                    if pol != "static":
                        q = pair(mp, sc, pol, "static"); r[f"d_{sc}"] = V(float(q.mean_pct) / 100, fmt=PCT); r[f"p_{sc}"] = " / ".join(f"{v:+.1f}" for v in json.loads(q.per_seed))
                    r[f"b_{sc}"] = f"占空 {x.duty_B.mean():.2f}，投机开 {x.spec_on_B.mean():.0%}，换头 {x.swaps_B.mean():.1f}"
                rows.append((pol, r))
            cols = [("k", "策略", None)]
            for sc, nm in MIX_SC:
                cols += [(f"t_{sc}", f"{nm}\nB 吞吐", INT), (f"d_{sc}", "vs 静态投机", PCT), (f"p_{sc}", "逐次 %", None), (f"b_{sc}", "B 段行为", None, 2)]
            s.table(cols, rows, caption="表 4　混合流量矩阵（每格 3 次）")
            rows = []
            for sc, nm in MIX_SC:
                for pol in ("v6", "v6_stale", "v6_nocurve", "switch", "tide"):
                    r = {"sc": nm if pol == "v6" else "", "k": dict(MIX_POL)[pol]}
                    for ref in ("v6", "nospec", "switch"):
                        q = pair(mp, sc, pol, ref)
                        r[ref] = V(float(q.mean_pct) / 100, fmt=PCT) if q is not None and pol != ref else None
                    rows.append((f"{sc}{pol}", r))
            s.table([("sc", "混合", None), ("k", "策略", None), ("v6", "vs 完整（混合标签）", PCT), ("nospec", "vs 纯解码", PCT), ("switch", "vs 仅开关", PCT)], rows,
                    caption="表 5　标签失效 / 无标签 的代价（配对均值）")
        else:
            s.para("混合矩阵未完成；跑完后重新生成本表。")
        s.note("作业记录：混合矩阵工人 16072978 第一次在 c0817 跑了 41 min 后节点故障（NODE_FAIL，杀掉了 code+对话 仅开关 第 3 次），Slurm 自动重排后 2 s 退出；"
               "该运行由 16078067 补跑。sacct -X 只记重排后的那次，所以作业清单里 16072978 只有 2 s，节点故障的 0.68 GPU·h 不在清单里。")
        s.source("results/abl_ablate、results/abl_mix（analyze_cells.py 从 data/runs/abl_ablate、data/runs/abl_mix 与 data/runs/abl 配对生成）、"
                 "code/specsched/sched/model_mix.json、results/summaries/2026-10-28_SGLang移植.md 第 11–12 节")
    res = ("去掉开关：中文 −43.8%；去掉回本规则：代码 −2.5%；去掉停止测试：0；去掉每域曲线：三个单域场景噪声内" + ("；混合流量见明细" if has_mix else ""))
    return meta(ctx, "E24", short="消融与混合流量", name="收益来自哪个判断（消融）+ 混合流量与标签失效",
                q="收益来自盈利性 / 学习上限 / 回本判断的哪一个？混合流量、标签失效或没有标签时控制器怎么样？",
                cfg="vLLM 冻结 checkpoint；消融 4 格 × 3 场景 × 3 次 = 36；混合流量 7 策略 × 2 混合 × 3 次 = 42（混合域单独校准进 model_mix.json）",
                res=res, verdict="开关在中文是全部收益、回本规则在代码值 2–3 点、停止测试冗余、每域曲线在单域场景不值钱" + ("；混合流量见明细" if has_mix else "（混合矩阵进行中）"),
                word="成立", status="有效", build=build)


V7_DIR, V71_DIR, BUD_DIR = f"{ROOT}/results/abl_v7", f"{ROOT}/results/abl_v71", f"{ROOT}/results/abl_budget"
SEG_DIR, WAIT_DIR = f"{ROOT}/results/abl_seg", f"{ROOT}/results/abl_wait"
SC5 = [("m1_code", "8B 代码"), ("m1_zh", "8B 中文"), ("m2_code", "MoE 代码"), ("m1_mixcc", "8B 代码+对话"), ("m1_mixzc", "8B 中文+对话")]
BASE3 = [("m2_code", "MoE 代码"), ("m1_code", "8B 代码"), ("m1_mixcc", "8B 代码+对话")]


def _pairs(d):
    """single + mix cell_pairs.csv of one results root (analyze_cells.py), concatenated."""
    fs = [f"{d}/single/cell_pairs.csv", f"{d}/mix/cell_pairs.csv"]
    return pd.concat([pd.read_csv(f) for f in fs if os.path.exists(f)], ignore_index=True)


def _seeds(x):
    return " / ".join(f"{v:+.1f}" for v in json.loads(x))


def e25(ctx):
    P = {"switch_budget": (_pairs(BUD_DIR), "switch_budget"), "v70": (_pairs(V7_DIR), "v7"), "v71": (_pairs(V71_DIR), "v7")}
    NAMES = {"switch_budget": "Switch+Budget（校准固定预算；知道标签）", "v70": "v7.0 无标签自校准控制器", "v71": "v7.1（配对测量 + 关投机探测 + 多轮前瞻）"}

    def q(key, sc, ref):
        df, cell = P[key]
        x = df[(df.scenario == sc) & (df.cell == cell) & (df.ref == ref)]
        return x.iloc[0] if len(x) else None

    def m(key, sc, ref):
        x = q(key, sc, ref)
        return float(x.mean_pct) if x is not None else float("nan")
    bud_v6 = {sc: m("switch_budget", sc, "v6") for sc, _ in SC5}
    v70_v6 = {sc: m("v70", sc, "v6") for sc, _ in SC5[:3]}
    v71_v6 = {sc: m("v71", sc, "v6") for sc, _ in SC5[:3]}
    res = ("Switch+Budget vs v6：" + "、".join(f"{nm} {bud_v6[sc]:+.1f}%" for sc, nm in SC5) +
           f"；v7.0 vs v6：{v70_v6['m1_code']:+.1f} / {v70_v6['m1_zh']:+.1f} / {v70_v6['m2_code']:+.1f}%，vs 仅开关 代码+对话 {m('v70', 'm1_mixcc', 'switch'):+.1f}%"
           f"；v7.1 vs v6：{v71_v6['m1_code']:+.1f} / {v71_v6['m1_zh']:+.1f} / {v71_v6['m2_code']:+.1f}%，代码+对话 vs 仅开关 {m('v71', 'm1_mixcc', 'switch'):+.1f}%")

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], [34, 11, 15, 11, 15, 11, 15, 11, 15, 11, 15])
        s.title_(f"{e['id']} · {e['name']}",
                 "用户 11-05 的判断：现在的文章\"相当于只做了一个开关\"。四份外部评价指出 v6 的在线回本规则（式 3）在饱和下与占空比无关、只剩 go/no-go；"
                 "于是先造一个只用校准、不在线算价值的基线（Switch+Budget），再试一个完全不用域标签 / 学习曲线、在线自己估增益曲线的控制器（v7）。都在冻结 checkpoint 上，新文件不动冻结代码。")
        s.section("配置")
        s.kv([("Switch+Budget", "controller_budget.py：开关规则原样；每个域的训练预算 = 校准曲线上到 80% 余量所需请求数（天花板打不过前沿则 0）；倍增导出；饱和占空比 0.5。"
                                "与 v6 的差别只有一处：不在线算\"再训值不值\"。"),
              ("v7.0", "controller_v7.py：不读域标签、不用学习曲线；对接受长度曲线 a(N)=a0+D(1−e^{−N/N0}) 维护 48×48 网格后验（D、N0 对数正态先验，观测 sd 0.04），"
                       "每 tick 做一步前瞻（再训到下一导出点的期望净收益 > 0 才训），天花板测试 p(天花板过盈亏线)；regime 切换由饱和翻转 / 接受漂移检测。"),
              ("v7.1", "v7.0 + 前沿修正用配对的开 / 关测量（而非校准 Θ_plain）+ |增益−1|<0.12 时关投机探测 + 1–4 轮前瞻。"),
              ("配对", "与冻结矩阵（E19/E22）和混合矩阵（E24）同提示、同种子、同节点类型，按次配对；5 场景 × 3 次 × 3 策略 = 45 次 + v7.1 重跑 2 次。"),
              ("预注册", "ideas/2026-11-07_v7_自校准控制器_预注册.md（A：v7 H-A1..A5；D：Switch+Budget）。")])
        for ref, cap in (("v6", "表 1　相对完整控制器 v6（B 段吞吐，按次配对，%）"), ("switch", "表 2　相对仅运行时开关（%）"),
                         ("switch_budget", "表 3　v7 相对 Switch+Budget（%；Informed 基线，知道标签）")):
            rows = []
            for key in ("switch_budget", "v70", "v71"):
                if key == "switch_budget" and ref == "switch_budget":
                    continue
                r = {"k": NAMES[key]}
                for sc, _nm in SC5:
                    x = q(key, sc, ref)
                    if x is not None:
                        r[f"d_{sc}"] = V(float(x.mean_pct) / 100, fmt=PCT)
                        r[f"p_{sc}"] = _seeds(x.per_seed)
                rows.append((key, r))
            cols = [("k", "策略", None)]
            for sc, nm in SC5:
                cols += [(f"d_{sc}", f"{nm}\n均值", PCT), (f"p_{sc}", "逐次 %", None)]
            s.table(cols, rows, caption=cap)
        s.note(f"怎么读：① Switch+Budget 在五个场景都 ≥ v6（{'、'.join(f'{bud_v6[sc]:+.1f}' for sc, _ in SC5)}%）——有校准时 v6 的在线价值规则没有可证明的价值，"
               "式 (3) 的\"训多少\"在饱和下是空的（busy·T = n/r 与占空比无关），只剩 go/no-go。② v7.0 不用标签、不用曲线也能复现 v6 的决策（中文关投机不训、代码 / MoE 训），"
               f"单域三场景与 v6 噪声内（{v70_v6['m1_code']:+.1f} / {v70_v6['m1_zh']:+.1f} / {v70_v6['m2_code']:+.1f}），但混合代码+对话比仅开关差 {m('v70', 'm1_mixcc', 'switch'):+.1f}%："
               "前沿修正引用的校准 Θ_plain 是对话测的，混合流量上 c≈1.0–1.2。③ v7.1 用配对测量修这一点，结果更差（代码 "
               f"{v71_v6['m1_code']:+.1f}，代码+对话 vs 仅开关 {m('v71', 'm1_mixcc', 'switch'):+.1f}）：开关瞬态触发了伪接受漂移 regime 重置，加上探测 / 开关本身的成本；记为负结果，不做 v7.2。"
               "④ 结论：\"自校准\"在单域不输、在混合流量输 2–7 个点；带校准的固定预算已经是上界附近，算法空间在别处（见 E26/E27）。")
        s.source("results/abl_budget、results/abl_v7、results/abl_v71（analyze_cells.py 与 data/runs/abl、abl_mix 配对）；"
                 "code/specsched/controller_budget.py、controller_v7.py；results/summaries/2026-10-28_SGLang移植.md §18")
    return meta(ctx, "E25", short="自校准v7与预算基线", name="无标签自校准控制器 v7 与校准固定预算基线 Switch+Budget",
                q="v6 的在线回本规则有没有价值？不用域标签和学习曲线能不能做出同样的决策？",
                cfg="vLLM 冻结 checkpoint；5 场景 × 3 次 × {Switch+Budget, v7.0, v7.1}；按次与冻结矩阵 / 混合矩阵配对",
                res=res, verdict="Switch+Budget ≥ v6 处处成立（价值规则无增量）；v7.0 单域与 v6 持平、混合流量 −2.4；v7.1 负结果", word="部分成立", status="有效", build=build)


def e26(ctx):
    sp = pd.read_csv(f"{SEG_DIR}/seg_pairs.csv"); sr = pd.read_csv(f"{SEG_DIR}/seg_runs.csv")
    LEN = [("n2000", "短 2000 条"), ("n6000", "中 6000 条"), ("n16000", "全 16000 条")]
    STR = [("fixed200", "固定预算 200"), ("fixed800", "固定预算 800"), ("v6", "完整 v6"), ("v70", "v7.0"), ("v7", "v7.1")]

    def q(sc, st):
        x = sp[(sp.scenario == sc) & (sp.strategy == st)]
        return x.iloc[0] if len(x) else None

    def secs(sc):
        x = sr[(sr.scenario == sc) & (sr.strategy == "switch")].B_dur_s
        return float(x.mean()) if len(x) else float("nan")
    short = {b: (q(f"{b}_n2000", "fixed800").vs_switch if q(f"{b}_n2000", "fixed800") is not None else q(f"{b}_n2000", "fixed200").vs_switch) for b, _ in BASE3}
    res = ("45 s 段：固定预算 800 相对仅开关 " + "、".join(f"{nm} {short[b]:+.1f}%" for b, nm in BASE3) +
           f"；v6 在 8B 代码每个段长与最好策略差 ≤{max(abs(q(f'm1_code_{l}', 'v6').vs_best_fixed) for l, _ in LEN if q(f'm1_code_{l}', 'v6') is not None):.1f} 个点")

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], [22, 13, 11, 15, 11, 15, 11, 15, 11, 15, 11, 15])
        s.title_(f"{e['id']} · {e['name']}",
                 "新流量能持续多久不知道。把 B 段切成三种长度（请求数 2000 / 6000 / 16000 ≈ 45 s / 130 s / 全长），看固定预算、v6、v7 相对仅开关各亏赢多少；"
                 "同提示同种子，按次配对。预注册 ideas/2026-11-07（B 部分）。")
        rows = []
        for b, nm in BASE3:
            for l, ln in LEN:
                sc = f"{b}_{l}"
                r = {"sc": nm if l == "n2000" else "", "len": f"{ln}（{secs(sc):.0f} s）"}
                for st, _n in STR:
                    x = q(sc, st)
                    if x is not None:
                        r[f"d_{st}"] = V(float(x.vs_switch) / 100, fmt=PCT); r[f"p_{st}"] = _seeds(x.per_seed_vs_switch)
                rows.append((sc, r))
        cols = [("sc", "场景", None), ("len", "段长", None)]
        for st, n in STR:
            cols += [(f"d_{st}", f"{n}\nvs 仅开关", PCT), (f"p_{st}", "逐次 %", None)]
        s.table(cols, rows, caption="表 1　B 段吞吐相对仅开关（%）；段长秒数 = 仅开关运行的 B 段时长均值")
        rows = []
        for b, nm in BASE3:
            for l, ln in LEN:
                sc = f"{b}_{l}"
                r = {"sc": nm if l == "n2000" else "", "len": ln}
                for st in ("v6", "v70", "v7"):
                    x = q(sc, st)
                    if x is not None:
                        r[st] = V(float(x.vs_best_fixed) / 100, fmt=PCT)
                rows.append((sc, r))
        s.table([("sc", "场景", None), ("len", "段长", None), ("v6", "v6 vs 最好固定预算", PCT), ("v70", "v7.0 vs 最好固定预算", PCT), ("v7", "v7.1 vs 最好固定预算", PCT)],
                rows, caption="表 2　相对同一次里 fixed200 / fixed800 中更好的那个（事后选）")
        s.note(f"怎么读：① 固定预算在 45 s 的段上：8B 代码亏 {abs(q('m1_code_n2000', 'fixed200').vs_switch):.1f}–{abs(q('m1_code_n2000', 'fixed800').vs_switch):.1f}%、"
               f"代码+对话亏 {abs(q('m1_mixcc_n2000', 'fixed200').vs_switch):.1f}–{abs(q('m1_mixcc_n2000', 'fixed800').vs_switch):.1f}%（一来就训、段结束前回不了本——正是 E27 租买模型里 train_now 的损失）；"
               f"MoE 短段 800 持平（{q('m2_code_n2000', 'fixed800').vs_switch:+.1f}）、200 还赢 {q('m2_code_n2000', 'fixed200').vs_switch:+.1f}%（MoE 的头回本快）。② v6 在 8B 代码每个段长都在最好策略 1.5 个点内"
               f"（vs 最好固定预算 {' / '.join(f'{q(f'm1_code_{l}', 'v6').vs_best_fixed:+.1f}' for l, _ in LEN)}），MoE 上比最好的固定预算差 1.2–3.0，代码+对话短段 −8.9（天花板在盈亏线上，训什么都亏）。③ 排队时 v7 代码已升到 v7.1，\"v7\"列是 v7.1；"
               "v7.0 另跑 18 次（v70 列）：短段 8B −4.0、中段 +3.6（最好），MoE 中段 +7.0（最好）、短段 −3.7±7.6。④ 干扰扫描 99 + 18 次，标记见 results/abl_seg/anomaly_scan.csv。")
        s.source("results/abl_seg（analyze_segment.py 从 data/runs/abl_seg 生成）；code/specsched/abl_matrix_segment.py；results/summaries/2026-10-28_SGLang移植.md §19")
    return meta(ctx, "E26", short="段长×预算", name="新流量持续多久：段长 × 训练预算",
                q="流量持续时间未知时，固定预算、v6、v7 各在什么段长上亏？",
                cfg="vLLM 冻结 checkpoint；3 场景 × 3 段长 × {fixed200, fixed800, switch, v6, v7.1} × 3 次 = 135（v7.0 另 18 次）",
                res=res, verdict="固定预算在短段亏 5–14%；v6 在 8B 代码每个段长都在最好 1.5 点内；v7.0/v7.1 短段不稳", word="成立", status="有效", build=build)


def e27(ctx):
    wp = pd.read_csv(f"{WAIT_DIR}/wait_pairs.csv")
    has_prof = os.path.exists(f"{WAIT_DIR}/wait_profile_w4.csv")
    prof = pd.read_csv(f"{WAIT_DIR}/wait_profile_w4.csv") if has_prof else None
    real = pd.read_csv(f"{WAIT_DIR}/wait_realized_w4.csv") if os.path.exists(f"{WAIT_DIR}/wait_realized_w4.csv") else None
    trn = pd.read_csv(f"{WAIT_DIR}/wait_trainer.csv") if os.path.exists(f"{WAIT_DIR}/wait_trainer.csv") else None
    STR = [("train_now", "一来就训"), ("wait", "等 s=b/Δ 再训"), ("v6", "完整 v6"), ("wait_d05", "等（Δ̂×0.5）"), ("wait_d2", "等（Δ̂×2）")]
    LENS = [("w1", "0.5 s"), ("w2", "s+τ/2"), ("w3", "s+τ+0.5 s"), ("w4", "s+τ+2 s")]
    NSTAR = {"m2_code": "3200（s 38 s）", "m1_code": "3200（s 50 s）", "m1_mixcc": "1600（s 71 s）"}

    def q(sc, st):
        x = wp[(wp.scenario == sc) & (wp.strategy == st)]
        return x.iloc[0] if len(x) else None

    def v(sc, st, col):
        x = q(sc, st)
        return float(x[col]) if x is not None and pd.notna(x[col]) else float("nan")
    e1 = {b: (v(f"{b}_w1", "wait", "vs_switch"), v(f"{b}_w1", "train_now", "vs_switch")) for b, _ in BASE3}
    e2 = {b: v(f"{b}_w4", "wait", "vs_train_now") for b, _ in BASE3}
    best = {}
    for b, _ in BASE3:
        for l, _n in LENS:
            cands = {st: v(f"{b}_{l}", st, "vs_switch") for st, _ in STR[:3]}
            cands["switch"] = 0.0
            best[(b, l)] = max(cands, key=lambda k: cands[k] if pd.notna(cands[k]) else -1e9)
    n_v6 = sum(1 for k, s_ in best.items() if s_ == "v6")
    e1_ok = all(abs(e1[b][0]) <= 1.5 and e1[b][1] <= -3 for b, _ in BASE3)
    e2_ok = {b: abs(e2[b]) <= 1 for b, _ in BASE3}
    res = ("E1 " + ("成立" if e1_ok else "不成立") + "：0.5s 档 wait vs 仅开关 " + " / ".join(f"{e1[b][0]:+.1f}" for b, _ in BASE3) +
           "，train_now " + " / ".join(f"{e1[b][1]:+.1f}" for b, _ in BASE3) +
           "；E2（最长档 wait vs train_now ≤1 点）" + "、".join(f"{nm} {e2[b]:+.1f}{'✓' if e2_ok[b] else '✗'}" for b, nm in BASE3) +
           f"；v6 在 12 个（场景 × 段长）里 {n_v6} 个最好")

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], [26, 12, 11, 14, 11, 14, 11, 14, 11, 14, 11, 14])
        s.title_(f"{e['id']} · {e['name']}",
                 "两份外部评价把问题改写成租买（ski-rental）：一段新流量时长 T 未知，训一个固定检查点 n* 的头要付直接损失 b、τ 秒后才生效、生效后每秒多服务 Δ；"
                 "等 regime 持续 s=b/Δ 再训在受限模型里最坏损失 ≤ 2·OPT。实验 E 用同一检查点、同一开关、按种子配对，在四种段长上比 一来就训 / 等 s 再训 / 不训 / v6。"
                 "预注册 ideas/2026-11-09_租买模型_预注册.md（判据 E1–E4 先写定）。")
        s.section("配置")
        s.kv([("检查点 n*", "、".join(f"{nm} {NSTAR[b]}" for b, nm in BASE3) + "；b、Δ 从校准冻结（start 事件里有 calib），只训 B 域，单次导出。"),
              ("段长", "0.5s / s+τ/2 / s+τ+0.5s / s+τ+2s（请求数由到达率换算；τ 预注册按 n*/到达率估：MoE 71、8B 72、mixcc 55 s）。"),
              ("策略", "switch（不训）、train_now（一来就训）、wait（regime 年龄 ≥ s 才训）、v6（现有控制器原样）；MoE 另加 wait_d05 / wait_d2（控制器拿到的 Δ̂ ×0.5 / ×2，段长不变）。"),
              ("规模", "3 场景 × 4 段长 × 4 策略 × 3 次 = 144 + 偏差变体 24 = 168 次；按预先写定的异常规则重跑 7 次（见表 6）。")])
        rows = []
        for b, nm in BASE3:
            for l, ln in LENS:
                sc = f"{b}_{l}"
                r = {"sc": nm if l == "w1" else "", "len": f"{ln}（{v(sc, 'switch', 'seg_s'):.0f} s）"}
                for st, _n in STR:
                    x = q(sc, st)
                    if x is not None:
                        r[f"d_{st}"] = V(float(x.vs_switch) / 100, fmt=PCT, bold=(best[(b, l)] == st))
                        r[f"p_{st}"] = _seeds(x.per_seed_vs_switch)
                rows.append((sc, r))
        cols = [("sc", "场景", None), ("len", "段长", None)]
        for st, n in STR:
            cols += [(f"d_{st}", f"{n}\nvs 不训", PCT), (f"p_{st}", "逐次 %", None)]
        s.table(cols, rows, caption="表 1　B 段吞吐相对不训（仅开关）；粗体 = 该行最好（含不训=0）")
        rows = []
        for b, nm in BASE3:
            for l, ln in LENS:
                sc = f"{b}_{l}"
                rows.append((sc, {"sc": nm if l == "w1" else "", "len": ln,
                                  "tn_live": V(v(sc, "train_now", "live_s"), fmt=F1), "w_start": V(v(sc, "wait", "train_start_s"), fmt=F1),
                                  "w_live": V(v(sc, "wait", "live_s"), fmt=F1), "w_frac": V(v(sc, "wait", "live_frac"), fmt=PCTP),
                                  "v6_live": V(v(sc, "v6", "live_s"), fmt=F1), "d05": V(v(sc, "wait_d05", "train_start_s"), fmt=F1),
                                  "d2": V(v(sc, "wait_d2", "train_start_s"), fmt=F1), "p99": V(v(sc, "wait", "p99"), fmt=F1)}))
        s.table([("sc", "场景", None), ("len", "段长", None), ("tn_live", "train_now\n头上线 s", F1), ("w_start", "wait\n开训 s", F1), ("w_live", "wait\n头上线 s", F1),
                 ("w_frac", "wait 段内\n上线比例", PCTP), ("v6_live", "v6 首个头\n上线 s", F1), ("d05", "Δ̂×0.5\n开训 s", F1), ("d2", "Δ̂×2\n开训 s", F1), ("p99", "wait\np99 ms", F1)],
                rows, caption="表 2　时序（距 B 流量饱和起算的秒数，3 次均值；空 = 段内没发生）")
        s.section("判据")
        rows = [("E1", {"k": "E1　0.5s 档：wait ≈ switch（|差|≤1.5）且 train_now 差 ≥3 点",
                        "r": "wait " + " / ".join(f"{e1[b][0]:+.1f}" for b, _ in BASE3) + "；train_now " + " / ".join(f"{e1[b][1]:+.1f}" for b, _ in BASE3),
                        "v": V("成立" if e1_ok else "不成立", status="成立" if e1_ok else "不成立")}),
                ("E2", {"k": "E2　s+τ+2s 档：wait 与 train_now 差 ≤1 点", "r": "、".join(f"{nm} {e2[b]:+.1f}" for b, nm in BASE3),
                        "v": V("部分成立" if any(e2_ok.values()) and not all(e2_ok.values()) else ("成立" if all(e2_ok.values()) else "不成立"),
                               status="成立" if all(e2_ok.values()) else "参考")}),
                ("E3", {"k": "E3　s+τ/2 档：流量中途结束时训练中止、已付成本计入",
                        "r": f"wait 在 w2 无头上线、吞吐 8B {v('m1_code_w2', 'wait', 'vs_switch'):+.1f} / mixcc {v('m1_mixcc_w2', 'wait', 'vs_switch'):+.1f}（train_now {v('m1_code_w2', 'train_now', 'vs_switch'):+.1f} / {v('m1_mixcc_w2', 'train_now', 'vs_switch'):+.1f}）；B 是最后一段，\"中止\"由构造成立",
                        "v": V("成立（成本计入可见）", status="成立")}),
                ("E4", {"k": "E4　Δ̂×0.5 推迟开训、Δ̂×2 提前开训，损失方向与模型一致",
                        "r": f"MoE w2/w3 开训：wait {v('m2_code_w2', 'wait', 'train_start_s'):.0f}/{v('m2_code_w3', 'wait', 'train_start_s'):.0f} s，×0.5 → {v('m2_code_w2', 'wait_d05', 'train_start_s'):.0f}/{v('m2_code_w3', 'wait_d05', 'train_start_s'):.0f} s，×2 → {v('m2_code_w2', 'wait_d2', 'train_start_s'):.0f}/{v('m2_code_w3', 'wait_d2', 'train_start_s'):.0f} s；短段 ×0.5 更好、中段更亏",
                        "v": V("成立", status="成立")})]
        s.table([("k", "判据（预注册原文）", None, 4), ("r", "结果", None, 6), ("v", "判定", None, 2)], rows, caption="表 3　预注册判据")
        if has_prof:
            wins = [w for w in ["0-10s", "10-40s", "40-80s", "80-120s", "120-160s", "160-240s"] if w in set(prof.window)]
            rows = []
            for b, nm in BASE3:
                for st, n in STR:
                    x = prof[(prof.base == b) & (prof.strategy == st)]
                    if not len(x):
                        continue
                    r = {"sc": nm if st == STR[0][0] else "", "st": n}
                    for w in wins:
                        y = x[x.window == w]
                        if len(y):
                            r[w] = V(float(y.vs_switch_pct.iloc[0]) / 100, fmt=PCT)
                    rows.append((f"{b}{st}", r))
            s.table([("sc", "场景", None), ("st", "策略", None)] + [(w, f"{w}\nvs 不训", PCT) for w in wins], rows,
                    caption="表 4　最长档（w4）的时间剖面：按\"距 B 流量饱和的秒数\"分窗，吞吐相对不训（%）")
        if real is not None:
            rows = []
            for i, x in real.iterrows():
                rows.append((str(i), {"sc": dict(BASE3).get(x.base, x.base), "st": dict(STR).get(x.strategy, x.strategy), "re": V(float(x.realized_gain_pct) / 100, fmt=PCT),
                                      "ca": V(float(x.calibrated_gain_pct) / 100, fmt=PCT) if pd.notna(x.calibrated_gain_pct) else None,
                                      "ps": _seeds(x.per_seed), "hl": V(int(x.heads_live), fmt=INT)}))
            s.table([("sc", "场景", None), ("st", "策略", None), ("re", "实测增益\n（段末 40 s）", PCT), ("ca", "校准 Δ/Θ", PCT), ("ps", "逐次 %", None, 2), ("hl", "头上线次数", INT)],
                    rows, caption="表 5　训出的头到底快了多少：w4 段末 40 s 相对不训，对比校准的 Δ/Θ_off=(g1−off)/off")
        if trn is not None:
            fl = trn[trn.slow_trainer | trn.crash]
            rows = []
            for i, x in fl.iterrows():
                rows.append((str(i), {"sc": x.scenario, "st": x.strategy, "seed": V(int(x.seed), fmt=INT), "host": x.host, "t": x.started, "rate": V(x.rate_req_s, fmt=F1),
                                      "con": V(int(x.consumed), fmt=INT), "gap": V(x.dropped_gap, fmt=INT), "why": "训练器崩溃（cuDNN MHA）" if x.crash else "训练器慢"}))
            s.table([("sc", "场景", None), ("st", "策略", None), ("seed", "次", INT), ("host", "节点", None), ("t", "开始", None), ("rate", "消费\n条/秒", F1),
                     ("con", "消费条数", INT), ("gap", "tap 断档\ntoken", INT), ("why", "原因", None, 2)], rows,
                    caption="表 6　训练器扫描命中（规则 b/c；中位消费速率 " + "、".join(
                        f"{dict(BASE3).get(b, b)} {r:.1f}" for b, r in trn[(trn.train_ticks >= 20) & (trn.strategy != 'v6') & (~trn.scenario.str.endswith('_w1'))]
                        .groupby('base').rate_req_s.median().items()) +
                            f" 条/秒）；MoE 训练器 OOM 重试警告 {int(trn[trn.base == 'm2_code'].oom_warn.sum())}/{int((trn.base == 'm2_code').sum())} 次；"
                            "被标记的原运行在 data/runs/abl_wait_flagged，重跑结果已进上表")
        s.section("与预注册的偏差（事后按日志发现，不改规则、不重跑）")
        s.bullets(["等待时钟：实现从标签切换起算 regime 年龄，而 harness 在切换与 B 流量饱和之间有 22–35 s 空闲；有效等待 = s − (22–35) s（MoE 10–21、8B 23–49、mixcc 40–51 s）。",
                   "τ：预注册只按到达率估，没算训练器在饱和下算力受限：8B 训练器 duty 0.5 时只吃 27–37 条/秒（到达 56），3200 条实际 110–140 s；MoE 数据受限 66–72 s（与估计一致）；"
                   "mixcc 60–120 s。所以 8B 的\"s+τ+2s\"档其实没回本。",
                   "Δ：8B 代码在线训出的头段末实测约为校准增益的 2/3（表 5，逐次 9–24% 波动大）；mixcc 约 80%；MoE 与校准一致。b：训练器运行期间吞吐损失 13–17%（表 4 的 10–80 s 窗），乘以实际 τ 后 b 约为校准值的 2 倍。"
                   "三者叠加，8B 代码真实的 s=b/Δ 约 150 s 而非 50 s。",
                   "E2 的 1 个点判据写定时没算模型自身预测的差（Δ·s_eff/ΘT ≈ 1–4 个点）；按原判据报告，不事后放宽。",
                   "最重要的发现在 v6 列：训练划算的两个场景（MoE、8B 代码）每个段长 v6 都最好，而所有单个大检查点策略都在 0 附近或负——v6 的倍增导出让第一个 300 条的头在流量开始后 5–12 s 上线，"
                   "段末 1300–1500 条的头与 3200 条的头增益相同（表 5）。受限模型里这对应 OPT=b+Δτ 的 Δτ 项：决定收益的是检查点粒度 τ，不是等多久 s。代码+对话上没有任何训练策略赢过不训。",
                   "机会损失按 token 在本 harness 上无意义（固定请求数、总 token 相同），用吞吐表达。"])
        s.source("results/abl_wait（analyze_wait.py、analyze_wait_profile.py 从 data/runs/abl_wait 生成；被标记运行在 data/runs/abl_wait_flagged）；"
                 "code/specsched/controller_budget.py（--wait-mode payback --single-export）、abl_matrix_wait.py；results/summaries/2026-10-28_SGLang移植.md §20")
    return meta(ctx, "E27", short="租买模型实验E", name="租买模型：等回本再训 vs 一来就训 vs 不训 vs v6（四种段长）",
                q="\"等 regime 持续 s=b/Δ 再训\"在实机上是否如受限模型所说？单个大检查点和 v6 的递进小检查点谁更好？",
                cfg="vLLM 冻结 checkpoint；3 场景 × 4 段长 × 4 策略 × 3 次 + MoE 偏差变体 24 = 168 次，异常规则重跑 7 次；同检查点、同开关、按种子配对",
                res=res, verdict="E1 成立、E2 部分、E3/E4 成立；但 v6 的递进小检查点在训练划算的场景每个段长都赢单个大检查点——关键量是检查点粒度 τ 而非等待 s",
                word="部分成立", status="有效", build=build)


ARM_DIR, SIM_ARM_DIR = f"{ROOT}/results/abl_arm", f"{ROOT}/results/sim_arm"
VERD = {"better": "更好", "worse": "更差", "not separable": "分不开"}
SC5E = [("m1_code", "8B 代码"), ("m2_code", "MoE 代码"), ("m1_mixcc", "8B 代码+对话"), ("m1_zh", "8B 中文"), ("m1_mixzc", "8B 中文+对话")]
ARMN = {"switch": "Switch", "sw": "sw", "round800": "round800", "budget": "budget", "duty05": "duty05", "novalue": "novalue", "full": "Full"}
E28_CONTR = [("sw", "switch"), ("full", "sw"), ("full", "novalue"), ("full", "round800"), ("full", "duty05"), ("full", "budget"),
             ("round800", "sw"), ("budget", "sw"), ("duty05", "sw"), ("novalue", "sw"), ("full", "switch")]
E28_PRE = set(E28_CONTR[:6])


def e28(ctx):
    pb = pd.read_csv(f"{ARM_DIR}/pairs_B_thr.csv")
    con = pd.read_csv(f"{ARM_DIR}/conclusions.csv")
    cs = pd.read_csv(f"{ARM_DIR}/cum_summary.csv")
    runs = pd.read_csv(f"{ARM_DIR}/runs_full.csv")
    old = pd.read_csv(f"{ARM_DIR}/before_rerun/runs_full.csv")
    ci = pd.read_csv(f"{ARM_DIR}/plan_rows/row19_e28a_ci.csv")
    aa = pd.read_csv(f"{ARM_DIR}/plan_rows/row19_aa.csv")
    md = pd.read_csv(f"{ROOT}/results/abl_arm_diag/memdiag_runs.csv")

    def p(sc, a, b):
        x = pb[(pb.scenario == sc) & (pb.a == a) & (pb.b == b)]
        return x.iloc[0] if len(x) else None
    hold = {r.item: bool(r.holds) for r in con.itertuples()}
    word = {k: "成立" if v else "不成立" for k, v in hold.items()}
    res = (f"同起点下 Full 对 sw（B 段不训）：8B 代码 {p('m1_code', 'full', 'sw').mean_pct:+.1f}%、MoE {p('m2_code', 'full', 'sw').mean_pct:+.1f}%（三种子同号）；"
           f"对校准预算 budget：{p('m1_code', 'full', 'budget').mean_pct:+.1f} / {p('m2_code', 'full', 'budget').mean_pct:+.1f} / "
           f"{p('m1_mixcc', 'full', 'budget').mean_pct:+.1f}%（分不开）；预注册结论 A {word['A']}、B {word['B']}、C {word['C']}、D {word['D']}")
    lead = ("同一个起点（A1 预适配完全相同、B 段开始时用同一个头）上比较七个臂。Full 比 B 段不训的 sw 在 8B 代码高 3.3%、MoE 高 7.8%，三种子同号；"
            "但和校准固定预算 budget、固定 800 条 round800 在三个可训场景都分不开，9 个种子里 8 个 budget 领先 Full。预注册结论：(A) 成立，(B) 不成立，(C) 成立，(D) 不成立。"
            "主张收窄为\"校准给出的训或不训，加一个预算\"；在线回本规则在停止判据之外的价值\"在本协议下测不出\"。")

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], [22, 12, 12, 12, 11, 11, 11, 12, 12, 12])
        s.title_(f"{e['id']} · {e['name']}", lead)
        s.section("配置")
        s.kv([("预注册", "ideas/2026-11-24_同起点对照与累计回本_预注册.md（判据、异常规则、A/A 写在运行之前；变更记录逐条写在评估之前）。"),
              ("臂", "switch（冻结仅开关）；sw（A1 与 Full 相同，B 段不训不导出）；round800（B 段占空比 0.5 训到 800 条）；budget（E25 校准预算 8B 代码 428、MoE 659、"
                     "代码+对话 194，中文 0）；duty05（B 段整段 0.5）；novalue（v6 去掉饱和回本规则，换成固定 0.5，停止判据保留）；full（冻结 v6）。"),
              ("场景与规模", "5 场景（8B 代码 / MoE 代码 / 8B 代码+对话 / 8B 中文 / 8B 中文+对话）× 3 种子；中文两场景没有 budget、novalue。93 次 + 1 次预注册重跑；"
                            "37 个作业（含冒烟 2、显存诊断 2、重跑 1）共 17.7 GPU·h，约 2.4 美元。"),
              ("冻结", "vLLM 0.27.1、Qwen3-8B / Qwen3-30B-A3B、EAGLE-3 k=3、B 段饱和 C=256、16000 条闭环；控制器 controller.py v5.1 冻结，其余臂用 controller_arm.py 新文件。"),
              ("与 E25 的偏差", "round800 和 budget 停训后，若距上次导出又学了 ≥100 条，再导出换一次头（预注册写明）。实际学到：round800 805–870 条，budget 8B 501–507、MoE 721–771、代码+对话 217–234。"),
              ("开环 E28b", "取消，从未提交：两次开环冒烟都打满显存，其中 200 req/s 那次训练器 CUDA OOM 死掉而 harness 没发现。6 次诊断（不进对比）判定是服务进程分配器缓存增长，不是真实需求；"
                           "训练器先占显存可以保住训练器（单次、种子 1）。要恢复开环，需要登记一个偏离冻结显存安排的新子实验（36 次，约 1.2 美元），由用户决定。")])
        rows = []
        for sc, nm in SC5E:
            first = True
            for a, b in E28_CONTR:
                x = p(sc, a, b)
                if x is None:
                    continue
                c = ci[(ci.scenario == sc) & (ci.a == a) & (ci.b == b)]
                rows.append((f"{sc}|{a}|{b}", {"sc": nm if first else "", "c": f"{ARMN[a]} 对 {ARMN[b]}" + ("" if (a, b) in E28_PRE else "（描述）"),
                                               "m": V(float(x.mean_pct) / 100, fmt=PCT, bold=(a, b) in E28_PRE), "ps": str(x.per_seed).replace(";", "；"),
                                               "sd": V(float(x.sd_pct), fmt=F2),
                                               "ci": f"{c.ci_lo.iloc[0]:+.2f} 到 {c.ci_hi.iloc[0]:+.2f}" if len(c) else "",
                                               "v": V(VERD[x.verdict], status=VERD[x.verdict])}))
                first = False
        s.table([("sc", "场景", None), ("c", "对比（a 对 b）", None, 2), ("m", "均值", PCT), ("ps", "逐种子 %", None, 2), ("sd", "标准差", F2),
                 ("ci", "95% t 区间", None, 2), ("v", "判定", None)], rows,
                caption="表 1　B 段输出吞吐，按种子配对的相对差（a−b，%）；判定：均值 ≥+2% 且三种子同号=更好，≤−2% 且同号=更差，否则分不开；区间只作描述",
                note="8B 代码上 Full 对 round800 −1.62%（区间 −1.94 到 −1.30）、MoE 上 Full 对 round800 −1.73%、对 budget −1.41%：按预注册规则分不开，但区间不含 0，方向都是 Full 落后。")
        defs = {"A": "Full 对 sw、round800、duty05 三个对照都没有任何场景更差，且对每个对照至少在 5 个场景中的 2 个更好",
                "B": "在 (A) 之上，Full 对 budget 在 3 个可训练场景中至少 2 个更好且没有更差（在线效用计算相对校准固定预算有独立价值）",
                "C": "budget 在 3 个可训练场景上都与 Full 分不开或更好（主张收窄为\"校准给出的训或不训加一个预算\"）",
                "D": "Full 对 novalue 至少一个场景更好且没有更差（回本规则在停止判据之外仍有可测的值；否则写\"在本协议下测不出\"）"}
        s.table([("k", "结论", None), ("d", "预写内容", None, 5), ("h", "判定", None), ("x", "依据", None, 3)],
                [(r.item, {"k": f"({r.item})", "d": defs[r.item], "h": V(word[r.item], status=word[r.item]), "x": str(r.detail)}) for r in con.itertuples()],
                caption="表 2　四条预写结论（重跑前后判定相同）")
        rows = []
        for sc, nm in SC5E:
            first = True
            for a in ["switch", "round800", "budget", "duty05", "novalue", "full"]:
                x = cs[(cs.scenario == sc) & (cs.arm == a)]
                if not len(x):
                    continue
                x = x.iloc[0]
                be = "窗口内不回本" if not np.isfinite(x.breakeven_s) else f"{x.breakeven_s:.0f}"
                rows.append((f"{sc}|{a}", {"sc": nm if first else "", "W": V(float(x.W), fmt=INT), "a": ARMN[a],
                                           "d": V(float(x.delta_W_pct) / 100, fmt=PCT), "ps": str(x.delta_W_per_seed_pct).replace(";", "；"),
                                           "v": V(VERD[x.verdict_W], status=VERD[x.verdict_W]), "deep": V(float(x.deepest_s_of_ref), fmt=F2),
                                           "be": be, "bps": str(x.breakeven_per_seed).replace(";", "；").replace("inf", ">W")}))
                first = False
        s.table([("sc", "场景", None), ("W", "窗口 W s", INT), ("a", "臂", None), ("d", "W 内累计 token\n对 sw", PCT), ("ps", "逐种子 %", None, 2),
                 ("v", "判定", None), ("deep", "最深亏损\n（sw 的秒数）", F2), ("be", "回本 s\n（种子均值曲线）", None), ("bps", "逐种子回本 s", None)], rows,
                caption="表 3　固定窗口累计回本（第 7 项，闭环固定积压；不是固定到达时间戳）：W 内累计完成 token 相对 sw，最深亏损与回本时刻")
        s.section("异常处理（规则写在运行之前，93 次运行结束后只评估一次）")
        s.bullets(["(a) 干扰扫描：93 次里 78 次可评，0 次命中；duty05 的 15 次没有干净样本，只由 (b)–(e) 覆盖。最低：m1_zh sw 种子 2 为 0.866、m1_zh switch 种子 3 为 0.870，都在 c0802，高于 0.85。"
                   "混合场景以 data/runs/abl_mix 为参照重扫（第一次扫描不可评）。",
                   "(b) 训练器消费速率：命中 1 次，m1_zh round800 种子 1（c0807）11.74 条/秒 < 0.6 × 场景中位数 22.95。原因是中位数混合了只在 B 段开头训的 round800 和整段训的 duty05；按规则照做，不豁免。",
                   "(c) 进程死亡或失败请求：0；(d) 精确 A/A（MoE switch 对 sw）−0.58 / −0.62 / −0.11%：0；(e) 同起点（8B 恰好 1 个头、MoE 0 个）：0。",
                   "重跑 job 17132895 落在 c0802（两个近阈值 zh 运行所在节点），扫描 0.996、规则 (c)(e) 干净，不需要第二次重跑；原目录改名 r1_flagged 保留。"])
        o = old[(old.scenario == "m1_zh") & (old.strategy == "round800") & (old.seed == 1)].iloc[0]
        n = runs[(runs.scenario == "m1_zh") & (runs.strategy == "round800") & (runs.seed == 1)].iloc[0]
        s.table([("k", "字段", None, 2), ("o", "原运行（被标记）", None, 2), ("n", "重跑", None, 2), ("x", "说明", None, 4)],
                [("h", {"k": "节点", "o": str(o.host).split(".")[0], "n": str(n.host).split(".")[0], "x": "重跑排除 c0818、c0820 与原节点"}),
                 ("b", {"k": "B 段吞吐 tok/s", "o": V(float(o.B_thr), fmt=INT), "n": V(float(n.B_thr), fmt=INT), "x": "种子 2、3：15,309 / 15,314"}),
                 ("r", {"k": "饱和训练消费 条/秒", "o": V(float(o.train_req_s_sat), fmt=F2), "n": V(float(n.train_req_s_sat), fmt=F2), "x": "种子 2、3：15.50 / 14.89"}),
                 ("c", {"k": "zh full 对 round800", "o": "+7.58%（+9.79；+6.68；+6.28）", "n": "+6.66%（+7.02；+6.68；+6.28）", "x": "判定都是更好；四条结论不变"})],
                caption="表 4　唯一一次重跑：原值与重跑值")
        rows = [(f"{r.kind}|{r.scenario}|{r.a}", {"k": {"exact A/A": "精确 A/A", "near A/A": "近似 A/A", "cross-batch A/A": "跨批 A/A"}[r.kind],
                                                  "sc": dict(SC5E)[r.scenario], "c": f"{r.a} 对 {r.b}", "m": V(float(r.mean_pct) / 100, fmt=PCT),
                                                  "sd": V(float(r.sd_pct), fmt=F2), "ci": f"{r.ci_lo:+.2f} 到 {r.ci_hi:+.2f}",
                                                  "ps": str(r.per_seed).replace(";", "；")}) for r in aa.itertuples()]
        s.table([("k", "类型", None), ("sc", "场景", None), ("c", "比较", None, 2), ("m", "均值", PCT), ("sd", "标准差", F2), ("ci", "95% 区间", None, 2),
                 ("ps", "逐种子 %", None, 2)], rows, caption="表 5　A/A 噪声底（B 段吞吐）：同批精确 A/A 在 0.6% 以内，跨批单种子差最大 6.7%（中文）")
        rows = []
        for r in md.itertuples():
            rows.append((r.run, {"run": r.run, "arr": r.arrival.replace("closed loop", "闭环").replace("open loop", "开环"),
                                 "res": V(float(r.reserve_gib), fmt=F1) if r.reserve_gib == r.reserve_gib and r.reserve_gib > 0 else "无",
                                 "w": V(float(r.max_waiting), fmt=INT), "thr": V(float(r.B_tok_s), fmt=INT),
                                 "tt": f"{r.ttft_p50_s:.2f} / {r.ttft_p99_s:.2f}",
                                 "al": V(float(r.srv_alloc_peak_gib), fmt=F2) if finite(r.srv_alloc_peak_gib) else "无探针",
                                 "rv": V(float(r.srv_resv_peak_gib), fmt=F2) if finite(r.srv_resv_peak_gib) else "无探针",
                                 "rt": V(int(r.srv_retries), fmt=INT) if finite(r.srv_retries) else "无探针",
                                 "ok": "活到结束" if bool(r.trainer_alive_to_end) else "死亡"}))
        s.table([("run", "诊断运行", None), ("arr", "到达", None), ("res", "训练器预占\nGiB", F1), ("w", "最大排队", INT), ("thr", "B tok/s", INT),
                 ("tt", "TTFT p50 / p99 s", None), ("al", "服务端已分配\n峰值 GiB", F2), ("rv", "服务端保留\n峰值 GiB", F2), ("rt", "分配器\n重试", INT), ("ok", "训练器", None)],
                rows, caption="表 6　显存诊断（6 次，不进对比）：开环下已分配峰值与闭环相同，保留比已分配多 13–15 GiB（8B）/ 5–9 GiB（MoE），是分配器缓存增长")
        s.note("怎么读：① sw 对 switch 五个场景都分不开，A1 预适配本身在 B 段不带来收益；Full − sw（8B 代码 +3.3%、MoE +7.8%）就是 B 段训练的收益。"
               "② B 段整段 0.5 的 duty05 处处比 sw 差（−3.2% 到 −20.2%），训练的成本是真实的。③ 在三个可训场景，round800 的 9 个种子都领先 Full（−1.29 到 −1.73%），budget 领先 8 个；"
               "Full 对 round800 的\"更好\"只来自中文和中文+对话，那里 round800 在训不该训的东西。④ 回本时刻：8B 代码 Full 58 s、budget 84 s、round800 93 s；MoE 三者约 40 s；"
               "代码+对话没有一个臂和 sw 分得开（duty05 除外）。⑤ harness 不检测训练器死亡，分析里用 stop 事件 + 无 Traceback 判活：93 次全部活到结束。")
        s.source("results/abl_arm（analyze_arm.py：runs_full.csv、pairs_*.csv、conclusions.csv、cum_summary.csv、anomaly_*.csv；重跑前的全部输出在 before_rerun/）；"
                 "results/abl_arm/plan_rows（plan_rows_tables.py：区间与 A/A）；results/abl_arm_diag（analyze_memdiag.py）；data/runs/abl_arm、abl_arm_diag")
    return meta(ctx, "E28", short="同起点对照与累计回本", name="同起点对照（七个臂）与固定窗口累计回本",
                q="同一个起点上，效用调度比简单训练策略（固定一轮、固定占空比、校准预算）有没有额外价值？训练的损失多久回本？",
                cfg="vLLM 冻结 checkpoint；5 场景 × 3 种子 × 7 臂（中文两场景 5 臂）= 93 次 + 1 次预注册重跑；闭环固定积压；开环 E28b 取消",
                res=res, verdict="A 成立、B 不成立、C 成立、D 不成立：对 sw 有训练收益，对校准预算无可测增量", word="部分成立", status="有效", build=build)


def e29(ctx):
    rd = pd.read_csv(f"{SIM_ARM_DIR}/sim_arm_reading.csv")
    vj = jload(f"{SIM_ARM_DIR}/sim_arm_verdict.json")
    gate = jload(f"{SIM_ARM_DIR}/sim_arm_gate.json")
    fit = pd.read_csv(f"{ARM_DIR}/frontier_check/speedup_fit.csv")
    kap = pd.read_csv(f"{ARM_DIR}/row13/kappa_check.csv")
    r15 = pd.read_csv(f"{ARM_DIR}/value_replay/row15_scenarios.csv")
    rc = jload(f"{ARM_DIR}/value_replay/replay_check.json")
    P = pd.read_csv(f"{ARM_DIR}/plan_rows/pairs_all_metrics.csv")
    M = pd.read_csv(f"{ARM_DIR}/plan_rows/means_by_strategy.csv")
    GRP = {"trace": "轨迹（35 窗口）", "driftI": "I 组随机突发（60）", "driftII": "II 组中文（40）", "driftIII": "III 组浪涌（140）"}
    LAB = {"代价模型更好": "更好", "分不开": "分不开", "混合": "部分成立", "代价模型落后": "不成立"}
    res = ("模拟器闸门通过（4 张表最大相对误差 0）；代价模型对 switch_only 在轨迹、I、III 组更好，对两个固定预算策略只在轨迹上更好、随机突发上混合或分不开；"
           f"回放复现 {rc['duty_reproduced']}/{rc['ticks']} 个占空比决策；校准前沿在 8B 代码高估饱和加速比约 10 个点")
    lead = ("不跑 GPU，用已有日志和冻结模拟器补完第 2、3、6、10、11、13、14、15 项。模拟器里代价模型明显好于只开关投机，但对固定训练预算只在轨迹上更好；"
            "回放显示接受长度曲线准、第一次决策用的接受长度估计偏低，代码+对话上预测为正的三次实际都亏；校准前沿在 8B 代码上整体高估加速比约 10 个点，"
            "训练器在占空比 0.5 时实际只忙 35–39%。这些都只作描述，规则 3 的预注册判定见 E28。")

    def build(wb, ctx, e):
        s = Sheet(wb, e["sheet"], [24, 13, 11, 11, 11, 11, 12, 12, 12, 12])
        s.title_(f"{e['id']} · {e['name']}", lead)
        s.section("第 10 项：冻结模拟器里补 Switch 与固定训练预算（新文件 sim_arm.py，CPU）")
        g = gate["tables"] if "tables" in gate else []
        s.kv([("复现闸门", "冻结的 8 个策略在 4 张表上逐行逐列与冻结结果比对："
                         + "；".join(f"{t['table']} {t['rows_matched']}/{t['rows_frozen']} 行、{t['columns_checked']} 列、最大相对误差 {t['max_rel_err']:g}" for t in g)
                         + "。冻结的 V1 不过、V2 过、V3 不过、V4 过被原样复现。闸门脚本第一次因 pandas 3 的字符串类型崩溃，修复只动比较代码、写在评估之前。"),
              ("新策略", "switch_only（只开关）；round800_safe、budget_safe（训到 800 条 / 校准预算，饱和时 0.5、不饱和时取 SLO 内最大占空比）；naive、regime 两种计数方式作敏感性。"),
              ("判读规则", "比值 = 代价模型 goodput ÷ 对照，±0.1% 内算平；赢 ≥50% 且输 ≤10% = 更好，平 ≥80% = 分不开，输 ≥50% 且赢 ≤10% = 落后，否则混合（逐项列出）。")])
        rows = []
        for r in rd[rd.preregistered.astype(str) == "True"].itertuples():
            rows.append((f"{r.group}|{r.comparator}", {"g": GRP.get(r.group, r.group), "c": r.comparator, "n": V(int(r.n), fmt=INT),
                                                        "wtl": f"{int(r.win)} / {int(r.tie)} / {int(r.loss)}", "v": V(r.verdict, status=LAB.get(r.verdict, "参考")),
                                                        "med": V(float(r.ratio_median), fmt=F3), "mn": V(float(r.ratio_min), fmt=F3), "sum": V(float(r.sum_ratio), fmt=F4)}))
        s.table([("g", "组", None), ("c", "对照", None, 2), ("n", "场景数", INT), ("wtl", "赢 / 平 / 输", None), ("v", "判读", None), ("med", "比值中位数", F3),
                 ("mn", "最小比值", F3), ("sum", "总量比", F4)], rows, caption="表 1　预注册判读（模拟结果只说明负载覆盖范围，不等同于真实引擎验证）",
                note="混合组逐项清单见 results/sim_arm/sim_arm_mixed_items.csv 与预注册变更记录。I 组的输有三类：25 号是模拟器策略特有的\"不算已积队列的稳态检查\"（真实控制器没有这一分支）；"
                     "8 个 sql 长突发是估值不看积压对下一阶段的影响；6 个短突发是按时间窗计 goodput 的口径（推迟到突发后服务的 token 被计入）。II 组的输来自模拟器把 zh 训练按系数 1.0 迁移到 chat 的假设。")
        rows = []
        for k, frozen in [("V1", "不过"), ("V2", "过"), ("V3", "不过"), ("V4", "过")]:
            x = vj.get("extended", {}).get(k + "p", {}) if isinstance(vj, dict) else {}
            rows.append((k, {"k": k, "f": V(frozen, status="成立" if frozen == "过" else "不成立"),
                             "p": V("过" if x.get("passed") else "不过", status="成立" if x.get("passed") else "不成立")}))
        s.table([("k", "判据", None), ("f", "冻结记录", None), ("p", "加入新对照后（描述）", None, 2)], rows,
                caption="表 2　扩展判据 V1′–V4′：V2 在分母加入固定预算策略后由过变不过")
        s.section("第 14、15 项：规则 3 回放（replay_arm_value.py；只看决策）")
        s.kv([("回放检查", f"{rc['ticks']} 个规则 3 tick：占空比复现 {rc['duty_reproduced']} 个，H 全部复现（最大差 {rc['H_abs_diff_max']:g} 秒）；"
                         f"{rc['duty_mismatches']} 个不一致都在 V≈0 处，与日志只记 3 位小数一致。"),
              ("第 14 项读数", "学习曲线这一侧准：最后上线头的预测接受长度与 B 段末实测差 <0.03（9/9）。第一次决策的接受长度估计偏低（5/9 低 0.1 以上，最大 0.45），抬高预测 V。"
                             "代码+对话三次预测 V 为正（2,952–49,035 token），同一时域实际累计差为负（−32,267 到 −62,572）：盈亏边界附近的失败。"),
              ("第 15 项读数", "时域下限从 90 降到 30/60 秒，改变的决策全部落在 B 段前 30 秒；升到 180 秒，训练 tick 在 8B 代码变 2.9–3.5 倍、MoE 1.4–2.4 倍。")])
        rows = []
        for r in r15.itertuples():
            rows.append((f"{r.set}|{r.scenario}", {"set": {"abl_arm": "E28a", "abl": "冻结", "abl_mix": "冻结（混合）"}[r.set], "sc": dict(SC5E)[r.scenario],
                                                     "t": V(int(r.ticks), fmt=INT), "f": f"{int(r.train_f30)} / {int(r.train_f60)} / {int(r.train_f90)} / {int(r.train_f180)}",
                                                     "a": V(int(r.to_notrain_f30), fmt=INT), "b": V(int(r.to_notrain_f60), fmt=INT), "c": V(int(r.to_train_f180), fmt=INT)}))
        s.table([("set", "批次", None), ("sc", "场景", None), ("t", "规则 3 tick", INT), ("f", "训练 tick：下限 30 / 60 / 90 / 180 s", None, 3),
                 ("a", "30 s 下\n转为不训", INT), ("b", "60 s 下\n转为不训", INT), ("c", "180 s 下\n转为训", INT)], rows, caption="表 3　时域下限敏感性（第 15 项）")
        s.section("第 11、13 项：校准在真实饱和 B 段准不准（探索性，未预注册）")
        rows = []
        for r in fit.itertuples():
            rows.append((f"{r.set}|{r.scenario}", {"set": {"abl_arm": "E28a", "abl": "冻结", "abl_mix": "冻结（混合）"}[r.set], "sc": dict(SC5E)[r.scenario],
                                                     "n": V(int(r.n), fmt=INT), "rng": f"{r.acc_min:.2f}–{r.acc_max:.2f}", "d": V(float(r.mean_diff_pts), fmt=F1),
                                                     "fb": V(float(r.fit_breakeven), fmt=F2) if r.fit_breakeven >= r.acc_min - 0.3 else "低于观测范围",
                                                     "mb": V(float(r.model_breakeven), fmt=F2), "src": "冻结 nospec（跨批）" if r.plain_ref_source != r.set else "同批"}))
        s.table([("set", "批次", None), ("sc", "场景", None), ("n", "5 秒窗口", INT), ("rng", "接受长度范围", None), ("d", "前沿高估\n（点，均值）", F1),
                 ("fb", "实测盈亏\n接受长度", F2), ("mb", "前沿盈亏\n接受长度", F2), ("src", "纯解码参照", None, 3)], rows,
                caption="表 4　第 11 项：前沿预测的加速比减实测（训练器空闲、换头后 13 秒外、≥250 个运行中请求）",
                note="8B 代码上前沿整体高估约 10 个点，两批一致；实测盈亏接受长度约 2.54，前沿给 2.29。高估主要是整体偏移，影响\"这个接受长度下投机赚不赚\"（规则 1），对训练增益（两点之差）影响小。")
        rows = []
        for r in kap.itertuples():
            rows.append((r.scenario, {"sc": dict(SC5E)[r.scenario], "n": V(int(r.n_busy_windows), fmt=INT), "b": V(float(r.busy_share_mean), fmt=F3),
                                      "d": V(float(r.drop_mean_pct) / 100, fmt=PCTP), "p": V(float(r.pred_drop_mean_pct) / 100, fmt=PCTP),
                                      "kf": V(float(r.kappa_fit), fmt=F3), "km": V(float(r.kappa_model), fmt=F3)}))
        s.table([("sc", "场景", None), ("n", "训练窗口", INT), ("b", "平均忙碌比例", F3), ("d", "实测吞吐下降", PCTP), ("p", "模型 κ×忙碌", PCTP),
                 ("kf", "拟合 κ", F3), ("km", "校准 κ(256)", F3)], rows,
                caption="表 5　第 13 项：训练时少服务的吞吐（同接受长度箱里和训练器空闲窗口比）",
                note="占空比 0.5 时训练器实际只忙 35–39% 的时间，训练速度是模型假设的 29–51%（duty05 臂），计划的 τ 在真实运行里长约一倍；每忙碌一秒的损失比校准 κ 小 16–36%。")
        s.section("第 2、3 项：全生命周期与延迟（冻结主矩阵 81 次 + E28a）")
        rows = []
        for sc, nm in SC5E:
            first = True
            for b in ["nospec", "static", "switch", "fixed800", "fixed", "tide"]:
                x = P[(P.batch == "frozen") & (P.scenario == sc) & (P.a == "v6") & (P.b == b)]
                if not len(x):
                    continue
                d = {m_: x[x.metric == m_].iloc[0] for m_ in ["A1_thr", "B_thr", "A2_thr", "all_thr"] if len(x[x.metric == m_])}
                rows.append((f"{sc}|{b}", {"sc": nm if first else "", "b": {"nospec": "Plain", "static": "Static", "switch": "Switch", "fixed800": "Informed",
                                                                          "fixed": "Always", "tide": "TIDE-style"}[b],
                                           **{k: V(float(d[m_].mean_pct) / 100, fmt=PCT) for k, m_ in [("a1", "A1_thr"), ("bb", "B_thr"), ("a2", "A2_thr"), ("al", "all_thr")] if m_ in d},
                                           "ps": str(d["all_thr"].per_seed).replace(";", "；") if "all_thr" in d else "",
                                           "v": V(VERD[d["all_thr"].verdict], status=VERD[d["all_thr"].verdict]) if "all_thr" in d else ""}))
                first = False
        s.table([("sc", "场景", None), ("b", "Full 对", None), ("a1", "A1", PCT), ("bb", "B", PCT), ("a2", "A2", PCT), ("al", "全程", PCT), ("ps", "全程逐种子 %", None, 3),
                 ("v", "全程判定", None)], rows, caption="表 6　第 2 项：冻结主矩阵里 Full 相对各基线的分阶段与全程吞吐（全程含 A1 训练、换头与 A2）",
                note="E28a 同起点：A1 预适配在 8B 场景让 A1 吞吐降 10.4–10.6%（sw 对 switch），B 段不带来收益，全程 −0.3% 到 −2.2%；Full 对 switch 全程 8B 代码 +1.0%、MoE +6.7%（更好）、其余 −0.3% 到 +0.4%。")
        rows = []
        for sc, nm in SC5E:
            first = True
            for st, lab in [("nospec", "Plain"), ("switch", "Switch"), ("static", "Static"), ("fixed800", "Informed"), ("v6", "Full")]:
                x = M[(M.batch == "frozen") & (M.scenario == sc) & (M.strategy == st)]
                if not len(x):
                    continue
                x = x.iloc[0]
                rows.append((f"{sc}|{st}", {"sc": nm if first else "", "st": lab, "slo": V(float(x.slo_share), fmt=PCTP), "t50": V(float(x.tpot_p50), fmt=F1),
                                            "t99": V(float(x.tpot_p99), fmt=F1), "f50": V(float(x.ttft_p50), fmt=INT), "f99": V(float(x.ttft_p99), fmt=INT),
                                            "e99": V(float(x.e2e_p99) / 1000, fmt=F1)}))
                first = False
        s.table([("sc", "场景", None), ("st", "策略", None), ("slo", "TPOT 在 SLO 内\n的请求比例", PCTP), ("t50", "TPOT p50 ms", F1), ("t99", "TPOT p99 ms", F1),
                 ("f50", "TTFT p50 ms", INT), ("f99", "TTFT p99 ms", INT), ("e99", "端到端 p99 s", F1)], rows,
                caption="表 7　第 3 项：B 段请求级延迟（冻结主矩阵，3 次均值；SLO = 控制器的 2×T_base(1)：8B 16.69 ms、MoE 10.61 ms，事后口径，不宣称硬 SLO）",
                note="8B 上只要投机开着，TTFT 中位数就从约 50 ms 升到约 115 ms，Full 的 TPOT p99 比 Plain 高 7–41%；MoE 上投机让 TPOT 减半，TTFT 中位数从 103 ms 升到约 143 ms；MoE 在 C=256 下所有策略都不满足 10.6 ms。")
        s.section("第 6 项：时序图")
        s.para("results/abl_arm/row6/row6_timeseries.png（每个场景取 Full 吞吐居中的种子，与同种子 sw 对比：吞吐、接受长度、训练器忙碌比例与关投机区间、换头时刻、累计 token 差）。"
               "中文两场景 Full 与 sw 的决策相同，图上的差是运行间噪声（zh 种子 2 的 sw 比同组低约 7.5%）。")
        s.source("results/sim_arm（sim_arm.py、sim_arm_mech.py）；results/abl_arm/value_replay（replay_arm_value.py）；results/abl_arm/frontier_check（frontier_check.py）；"
                 "results/abl_arm/row13（interference_check.py）；results/abl_arm/plan_rows（plan_rows_tables.py）；results/abl_arm/row6（row6_timeseries.py）")
    return meta(ctx, "E29", short="离线整理与校准核对", name="离线整理：模拟器补对照、规则 3 回放、校准核对、全生命周期与延迟",
                q="不再跑 GPU，用已有日志和冻结模拟器能把计划里的第 2、3、6、10、11、13、14、15 项补到什么程度？",
                cfg="冻结模拟器 15 个策略 × 原 held-out 场景（CPU）；E28a 与冻结矩阵日志回放与核对", date="11-25（本机）",
                res=res, verdict="模拟器：对固定预算只在轨迹上更好；回放：代码+对话预测为正、实际为负；前沿在 8B 高估约 10 点", word="部分成立",
                status="有效（模拟）", build=build)


def build_experiments(ctx):
    fs = [e01, e02, e03, e04, e05, e06, e07, e08, e09, e10, e11, e12, e13, e14, e15, e16, e17, e18, e21]
    if os.path.exists(f"{ABL_DIR}/abl_runs.csv"):
        fs[-1:-1] = [e19, e20]
        if os.path.exists(E22_SCAN) and os.path.exists(E22_ANOM):
            fs.append(e22)
    if os.path.exists(f"{SGL_CHK}/m1/func/report.json"):
        fs.append(e23)
    if os.path.exists(f"{ABLATE_DIR}/cell_pairs.csv"):
        fs.append(e24)
    if os.path.exists(f"{BUD_DIR}/single/cell_pairs.csv"):
        fs.append(e25)
    if os.path.exists(f"{SEG_DIR}/seg_pairs.csv"):
        fs.append(e26)
    if os.path.exists(f"{WAIT_DIR}/wait_pairs.csv"):
        fs.append(e27)
    if os.path.exists(f"{ARM_DIR}/pairs_B_thr.csv") and os.path.exists(f"{ARM_DIR}/plan_rows/row19_aa.csv"):
        fs.append(e28)
    if os.path.exists(f"{SIM_ARM_DIR}/sim_arm_reading.csv") and os.path.exists(f"{ARM_DIR}/frontier_check/speedup_fit.csv"):
        fs.append(e29)
    out = [f(ctx) for f in fs]
    return sorted(out, key=lambda e: e["id"])


# =============================================================================================== summary sheets
def sheet_readme(wb, ctx, exps):
    jobs = ctx["jobs"]
    s = Sheet(wb, "说明", [22, 14, 52, 12, 12], back=False)
    n_sim = sum(1 for e in exps if "模拟" in e["status"])
    s.title_("在线训草稿头 × 投机解码：全部实验汇总",
             f"这个 idea 从 2026-10-04 到 10-24（悉尼时间）跑过的全部实验共 {len(exps)} 组：{len(exps) - n_sim} 组在 OSC Cardinal H100 上，对应 {len(jobs)} 个作业；"
             f"另外 {n_sim} 组是本机模拟。每组都写明配置、结果、结论和数据出处；作废的轮次单列，不混进正式结果。"
             "10-23 起的论文补充（E19–E22）全部在同一个冻结 checkpoint 上跑（E22 的 TIDE 式基线与重跑用按原版本重建的环境），之前的闭环数字只当开发记录。")
    s.section("这个 idea 是什么（大白话）")
    s.para("推理服务开着投机解码：小的草稿头（EAGLE-3）先猜几个 token，目标模型一次验证。草稿头猜得越准越快，但出厂的草稿头不一定对得上"
           "线上流量。想法是：服务一边跑，一边用自己刚服务过的请求在线训练草稿头，训好了不停服热替换上去；再由一个调度器决定什么时候训、"
           "训多少、投机该开还是该关，让同一张卡的吞吐最大。")
    s.section("怎么读这本表")
    s.bullets(["先看「核心结果」：一页里放了最重要的结论和数字。",
               f"「总览」每行一个实验（E01–E{len(exps):02d}）：要回答的问题、配置要点、关键结果、判定；点最后一列跳到明细表。",
               "每张明细表结构一样：顶部黄框是一句话结论 → 实验配置 → 结果表 → 怎么读 → 数据来源。",
               "「通用配置」写所有实验共用的硬件、软件、模型、数据、训练和测量口径；明细表只写和它不同的地方。",
               "「术语」解释接受长度、并发、占空比、frontier、热替换等说法。",
               "「作业清单」列出全部 OSC 作业（时间已换成悉尼时间）、卡时和对应实验；「作废与教训」列出不能用的结果、原因和文档勘误。",
               "实测值直接写入；比值、变化百分比、合计这类派生列是 Excel 公式，点单元格能看到算法。"])
    s.section("判定颜色")
    s.legend([("成立 / 通过 / 有效", "成立", "预注册假设成立，或功能验证通过，或结果有效"),
              ("部分成立 / 半过 / 边缘", "部分成立", "只在部分条件下成立，或差别落在刀口上，或推翻了原来的剧本"),
              ("不成立 / 作废", "不成立", "假设不成立，或这一轮结果不能用（原因见「作废与教训」）"),
              ("有效（模拟）", "有效（模拟）", "本机模拟器结果，参数来自实测曲线，不是真机测量"),
              ("对照 / 参考 / 取消", "对照", "对照组、参考信息，或作业被取消、没有结果")])
    s.section("四个阶段")
    rows = []
    for st, span, what in [("Rung 0", "10-04", "刻画冲突：投机收益与同卡训练代价随负载怎么变；在线更新多久回本（预注册 H1–H4）"),
                           ("Rung 1", "10-05 ~ 10-06", "frontier 闭式 + 最小调度器（模拟）+ 真机占空比控制 + 第一次热替换闭环"),
                           ("demo 冲刺", "10-08 ~ 10-10", "实时导出训练特征、控制器真管闭环、MoE、运行时投机开关、多卡热替换"),
                           ("论文补充", "10-23 ~ 10-24", "同一冻结 checkpoint：统一 v6 重测三场景 + 8 策略消融 + 每格 3 次重复（72 次闭环）；模拟器冻结后在新轨迹上验证")]:
        js = [j for j in jobs if STAGE.get(j["exp"]) == st]
        rows.append((st, {"st": st, "span": span, "what": what,
                          "n": F(f"=COUNTIF('作业清单'!$D:$D,\"{st}\")", len(js), INT),
                          "gh": F(f"=SUMIF('作业清单'!$D:$D,\"{st}\",'作业清单'!$J:$J)", sum(j["gh"] for j in js), F1)}))
    rows.append(("all", {"st": V("合计", bold=True), "span": "10-04 ~ 10-24", "what": "全部在 OSC Cardinal（账号 PCS0296）；debug 分区单卡为主，多卡走 gpu 分区",
                         "n": F("=COUNT('作业清单'!$H:$H)", len(jobs), INT, bold=True),
                         "gh": F("=SUM('作业清单'!$J:$J)", sum(j["gh"] for j in jobs), F1, bold=True)}))
    s.table([("st", "阶段", None), ("span", "日期（悉尼）", None), ("what", "做了什么", None), ("n", "OSC 作业数", INT), ("gh", "H100 卡时", F1)], rows,
            note="卡时 = 作业时长 × 分到的 GPU 数，按 sacct 记录汇总，含取消和作废的作业。本机模拟（E08、E09）不占 GPU。")
    s.section("所有实验共有的局限")
    s.bullets(["只有一个目标模型家族（Qwen3：8B 稠密、30B-A3B MoE），草稿长度 k=3 固定，贪心解码；没有采样解码。",
               "硬件只有 H100（单卡为主，TP=2/4 只验证了热替换）；没有 H20。",
               "流量是公开数据集的提示 + 目标模型自己的输出，不是真实线上日志；闭环实验的域切换是人为安排的 A→B→A，每段几分钟。",
               f"10-23 以前的测量只跑一次，靠跨实例噪声标定（变异系数 ≤{ctx['e05']['cv_max']:.1%}，E05）判断差别是否可信；E19 的矩阵每格独立重复 3 次，"
               "但 n=3 的统计检验只作参考。",
               "10-23 以前的闭环数字（E11–E16）来自控制器 v2–v6 多个版本和不同的提示池 / 段长，彼此不能直接相减；论文主表只用 E19（同一冻结 checkpoint）。"])
    s.section("文件与生成方式")
    s.kv([("原始数据", "data/runs/（OSC 结果的本地镜像；每组一个目录：summary.json / phase*.json / *.jsonl）"),
          ("结论文档", "results/summaries/2026-10-04_Rung0_结论.md、2026-10-05_Rung1_调度器.md、2026-10-08_demo冲刺_实时导出与控制器.md"),
          ("预注册", "ideas/2026-10-04_Rung0_characterization_预注册.md（H1–H4 与通过线，含变更记录）"),
          ("旧 Excel", "results/Rung0_数据总结.xlsx：逐条原始字段的导出（英文列名）；本表是它的可读版"),
          ("生成脚本", "code/specsched/build_summary_excel.py（重跑：python build_summary_excel.py <项目根目录>）"),
          ("作业记录", "data/osc_jobs_sacct.psv（从 OSC sacct 导出）"),
          ("冻结清单", "data/runs/abl/MANIFEST.json（E19 用的代码 / vLLM 补丁文件 / 头 / 提示池 / 环境的 hash）"),
          ("项目目录", ROOT),
          ("生成日期", datetime.now().strftime("%Y-%m-%d"))])


def sheet_overview(wb, ctx, exps):
    s = Sheet(wb, "总览", [6, 9, 11, 18, 26, 36, 40, 30, 12, 7, 7], back=False)
    s.title_("总览：每行一个实验", "点最后一列「明细 →」跳到对应的明细表；卡时由「作业清单」按实验汇总（公式），模拟实验不占 GPU。")
    rows = []
    for e in exps:
        rows.append((e["id"], {"id": V(e["id"], bold=True), "st": e["stage"], "date": e["date"], "name": V(e["name"], bold=True),
                               "q": e["q"], "cfg": e["cfg"], "res": e["res"], "verdict": V(e["verdict"], status=e["word"]),
                               "status": V(e["status"], status=("有效（模拟）" if "模拟" in e["status"] else "有效")),
                               "gh": F("=SUMIF('作业清单'!$C:$C,A{r},'作业清单'!$J:$J)", e["gh"], F1) if e["gh"] > 0 else V(0, fmt=F1),
                               "link": V("明细 →", link=e["sheet"])}))
    s.table([("id", "编号", None), ("st", "阶段", None), ("date", "日期（悉尼）", None), ("name", "实验", None), ("q", "要回答的问题", None),
             ("cfg", "配置要点", None), ("res", "关键结果", None), ("verdict", "结论 / 判定", None), ("status", "结果状态", None),
             ("gh", "卡时", F1), ("link", "明细", None)], rows)
    s.ws.freeze_panes = "E5"


def sheet_core(wb, ctx, exps):
    sheet_of = {e["id"]: e["sheet"] for e in exps}
    e2, e4, e5, e6 = ctx["e02"], ctx["e04"], ctx["e05"], ctx["e06"]
    e13, e14, e15, e16, e17 = ctx["e13"], ctx["e14"], ctx["e15"], ctx["e16"], ctx["e17"]
    gs = e2["gain"]
    s = Sheet(wb, "核心结果", [6, 30, 50, 30, 10])
    s.title_("核心结果（一页）", "冲突存在，但形状和最初设想不同：分时共置的代价是常数（约一半吞吐，可用占空比线性调小）；投机收益随负载的走向依情形而异；"
                            "在线更新便宜到\"什么时候训\"大体退化。真正值钱的是三个判别——按 frontier 开关投机、SLO 内免费训满并热替换、"
                            "识别结构天花板不白训。闭环里控制器用约 1/5–1/10 的训练量拿到与固定策略同量级的收益；但饱和时最大的杠杆是开关投机。")
    sat15 = [e15["sp"][h][256] for h in e15["sp"]]
    rows = [
        ("1", "投机收益随负载的走向分三种形状",
         f"8B 短输出 {gs['m1_short'][1]:+.0%} → {gs['m1_short'][256]:+.0%}（C=1→256）；8B 长输出 C=96 仍 {gs['m1_long'][96]:+.0%}；"
         f"MoE {gs['m2_short'][1]:+.0%} → {gs['m2_short'][64]:+.0%}（C=64）→ {gs['m2_short'][256]:+.0%}（C=256）", "H100 单卡，k=3，贪心", "E02"),
        ("2", "同卡训练（不限速）的代价与负载无关，但能线性调小",
         f"一直训：吃掉 {e2['cost'][0]:.0%}–{e2['cost'][1]:.0%} 吞吐，P99 TPOT 涨 {e2['p99'][0]:.0%}–{e2['p99'][1]:.0%}；"
         f"占空比 0.1 只损失 {ctx['e03']['loss'][0.1][0]:.0%}–{ctx['e03']['loss'][0.1][1]:.0%}，0.5 损失 "
         f"{ctx['e03']['loss'][0.5][0]:.0%}–{ctx['e03']['loss'][0.5][1]:.0%}",
         "两进程分时共享；MPS 不限额更糟（服务只剩约 1/5）", "E03"),
        ("3", "在线更新又便宜又有效，对话域正迁移",
         f"100 条请求 ≤1 GPU 秒；3200 条后代码 {e4['acc0']['code']:.2f}→{e4['acc3']['code']:.2f}、SQL {e4['acc0']['sql']:.2f}→{e4['acc3']['sql']:.2f}、"
         f"MoE 代码 {e15['a0']:.2f}→{e15['a3']:.2f}", "单遍在线，AdamW 5e-5，约 4096 token/步", "E04"),
        ("4", "中文是草稿词表天花板，不是漂移",
         f"词表只覆盖 42.6% 的输出 token；接受长度 {e4['acc0']['zh']:.2f}→{e4['acc3']['zh']:.2f} 就到顶", "AngelSlim 32k 草稿词表", "E04"),
        ("5", "稠密短输出在饱和时，学到头也反超不了不投机",
         f"固定长度口径：代码 N0 {e5['fx']['N0']:.2f}× → N3200 {e5['fx']['N3200']:.2f}×（相对不投机）", "变长口径含排空段、偏向投机", "E05"),
        ("6", "frontier 对接受长度线性，可用闭式标定",
         f"gain = α(C)·a + β(C)，残差 ≤{max(e6['res_max']['code'], e6['res_max']['zh']):.2f}（SQL C=16 为 {e6['res_max']['sql']:.2f}）；"
         "闭环里预测 +26%、实测 +24%", "T 曲线要按负载形状标定", "E06"),
        ("7", "\"什么时候训\"大体退化（模拟）",
         f"24 h trace：调度器比 OSD 式\"空闲就训\"只多 {ctx['e08']['vs_osd']:+.1%}，训练量为其 1/{ctx['e08']['ratio']:.0f}；"
         f"中文天花板场景调度器 goodput 是反应式策略的 {ctx['e09']['zh_cm'] / ctx['e09']['zh_osd']:.1f} 倍", "本机模拟，参数来自实测", "E09"),
        ("8", "热替换不停服、无损",
         f"单卡约 0.7 s、TP=2 {e17['tp'][2][1]['swap_s']:.2f} s、TP=4 {e17['tp'][4][1]['swap_s']:.2f} s、MoE 约 0.32 s；无操作重载 32/32 逐 token 一致",
         "reload_draft_weights（PR #60139）", "E17"),
        ("9", "实时导出训练特征的开销约 1.5%",
         f"饱和段吞吐 {ctx['e12']['d']['B']:+.1%}、TPOT {ctx['e12']['tB']:+.1%}", "live tap → /dev/shm 环形缓冲", "E12"),
        ("10", "闭环收益：稠密代码 +18%、中文 +11%、MoE +5%（控制器 v5.1）",
         f"代码 {e13['gx']['controller_live']:+.1%}（固定策略 {e13['gx']['train_swap_live']:+.1%}）、中文 {e14['gx']['controller_live']:+.1%}、"
         f"MoE {e15['gx']['controller_live']:+.1%}；控制器训练占空比均值 0.06–0.10", "B 段饱和 C=256、9000 条唯一请求；开发期数字，论文用 E19", "E13"),
        ("11", "饱和时最大的杠杆是开关投机",
         f"中文关投机 {e14['gx']['static_sched']:+.1%}、开关版控制器 {e14['gx']['controller_sw_live']:+.1%}；代码关投机 {e13['gx']['static_sched']:+.1%}；"
         f"运行时开关单测：代码 {e16['m1_code_v2']:+.1%}、中文 {e16['m1_zh']:+.0%}，切换 ≤10 ms", "需要引擎有运行时开关", "E16"),
        ("12", "MoE 永远不该关投机", f"饱和 C=256 仍 {min(sat15):.2f}–{max(sat15):.2f}×；训头再加 5–8%", "Qwen3-30B-A3B", "E15"),
    ]
    if "e19" in ctx and ctx["e19"].get("S") and all(sc in ctx["e19"]["S"] for sc in ("m1_code", "m1_zh", "m2_code")):
        e19 = ctx["e19"]
        S, ALL, P99 = e19["S"], e19["ALL"], e19["P99"]
        pm, cv = e19["pm"], e19["cv"]

        def f(x):
            return f"{x['m']:+.1f}%±{x['s']:.1f}" if finite(x["s"]) else f"{x['m']:+.1f}%"
        rows += [
            ("13", "同一冻结 checkpoint 上的终表：v6 在三个场景都成立（3 次重复）",
             f"B 段相对静态投机 代码 {f(S['m1_code']['v6'])}、中文 {f(S['m1_zh']['v6'])}、MoE {f(S['m2_code']['v6'])}（三次同号）；全程相对现网部署 "
             f"{f(ALL['m1_code']['v6'])} / {f(ALL['m1_zh']['v6'])} / {f(ALL['m2_code']['v6'])}；B 段 p99 {f(P99['m1_code'])} / {f(P99['m1_zh'])} / {f(P99['m2_code'])}",
             f"3 场景 × {len(set(e19['runs'].strategy))} 策略 × 3 次 = {len(e19['runs'])} 次闭环，B 段 16000 条", "E19"),
            ("14", "消融：8B 饱和段最大杠杆是关投机，MoE 不能关；frontier 判别是算法的独立贡献",
             f"纯解码相对静态投机：代码 {f(S['m1_code']['nospec'])}、中文 {f(S['m1_zh']['nospec'])}、MoE {f(S['m2_code']['nospec'])}；"
             f"阈值规则在 MoE {f(S['m2_code']['simple'])}，frontier 开关 {f(S['m2_code']['switch'])}；v6 − 仅开关 = 代码 {f(pm('m1_code', 'v6', 'switch'))}、"
             f"MoE {f(pm('m2_code', 'v6', 'switch'))}（训练换头的贡献）", "v6 在 8B 追平纯解码，在 MoE 靠训练 +8%", "E19"),
            ("15", "固定训练不是弱基线：已知切换点时在可适应域比 v6 高 1–2 点，但不会关投机",
             f"fixed800 相对静态投机 代码 {f(S['m1_code']['fixed800'])}、MoE {f(S['m2_code']['fixed800'])}、中文 {f(S['m1_zh']['fixed800'])}；"
             f"v6 − fixed800 = 代码 {f(pm('m1_code', 'v6', 'fixed800'))}、MoE {f(pm('m2_code', 'v6', 'fixed800'))}、中文 {f(pm('m1_zh', 'v6', 'fixed800'))}；"
             f"全程 0.5 占空比不停的固定训练 ≈ 静态（{f(S['m1_code']['fixed'])} / {f(S['m1_zh']['fixed'])} / {f(S['m2_code']['fixed'])}）",
             f"重复噪声：B 吞吐变异系数 静态 ≤{max(cv(sc, 'static') for sc in S):.1%}、v6 {cv('m1_code', 'v6'):.1%}–{cv('m1_zh', 'v6'):.1%}", "E19"),
        ]
    s.table([("n", "序号", None), ("f", "发现（大白话）", None), ("num", "关键数字", None), ("cond", "条件", None), ("lk", "明细", None)],
            [(n, {"n": n, "f": V(f, bold=True), "num": num, "cond": cond, "lk": V(f"{eid} →", link=sheet_of[eid])})
             for n, f, num, cond, eid in rows], caption=f"表 1　{len(rows)} 条结论（13–15 来自 10-23 的冻结 checkpoint 矩阵，论文主表以它们为准）")
    if "e19" in ctx and ctx["e19"].get("S") and all(sc in ctx["e19"]["S"] for sc in ("m1_code", "m1_zh", "m2_code")):
        e19 = ctx["e19"]
        mean = e19["mean"]
        rows3 = []
        for sc, lab in [("m1_code", "Qwen3-8B 代码"), ("m1_zh", "Qwen3-8B 中文"), ("m2_code", "Qwen3-30B-A3B 代码")]:
            b, m = mean(sc, "static", "B_thr"), mean(sc, "v6", "B_thr")
            rows3.append((sc, {"n": "", "f": V(lab, bold=True), "st": b, "v6": m, "x": F("={v6}{r}/{st}{r}-1", m / b - 1, PCT),
                               "lk": V("E19 →", link=sheet_of["E19"])}))
        s.table([("n", "", None), ("f", "场景", None), ("st", "静态投机 tok/s（3 次均值）", INT), ("v6", "完整 v6 tok/s", INT), ("x", "v6 相对静态投机", PCT),
                 ("lk", "明细", None)][:5], rows3, caption="表 2　终表（E19，同一冻结 checkpoint，B 段饱和 C=256、16000 条，三次均值）")
        rows4 = []
        for sc, lab in [("m1_code", "Qwen3-8B 代码"), ("m1_zh", "Qwen3-8B 中文"), ("m2_code", "Qwen3-30B-A3B 代码")]:
            S = e19["S"][sc]
            rows4.append((sc, {"n": "", "f": V(lab, bold=True),
                               "num": f"纯解码 {S['nospec']['m']:+.1f}%、仅开关 {S['switch']['m']:+.1f}%、简单规则 {S['simple']['m']:+.1f}%、"
                                      + (f"TIDE 式 {S['tide']['m']:+.1f}%、" if 'tide' in S else "") + f"固定训 N800 {S['fixed800']['m']:+.1f}%、"
                                      f"全程固定训 {S['fixed']['m']:+.1f}%、v6 {S['v6']['m']:+.1f}%（均相对静态投机，B 段，3 次配对均值）",
                               "cond": f"现网（无 tap、满显存）比静态投机高 {S['vanilla']['m']:+.1f}%", "lk": V("E19 →", link=sheet_of["E19"])}))
        s.table([("n", "", None), ("f", "场景", None), ("num", "各策略的 B 段吞吐（相对静态投机）", None), ("cond", "基础设施代价", None), ("lk", "明细", None)],
                rows4, caption="表 3　消融一览（E19）")
    rows2 = []
    for lab, h, eid, can_off in [("Qwen3-8B 代码", e13["h"], "E13", True), ("Qwen3-8B 中文", e14["h"], "E14", True),
                                 ("Qwen3-30B-A3B 代码", e15["h"], "E15", False)]:
        b = h["static_live"]["B"]["output_throughput"]
        fx, ct = h["train_swap_live"], h["controller_live"]
        f_thr, c_thr = fx["B"]["output_throughput"], ct["B"]["output_throughput"]
        train = (f"不训练 {b:,.0f} tok/s → 固定 0.5：{f_thr:,.0f}（{f_thr / b - 1:+.1%}，p99 {fx['B']['p99_tpot_ms']:.1f} ms）；"
                 f"控制器 v5.1：{c_thr:,.0f}（{c_thr / b - 1:+.1%}，占空比 {ct['duty']:.2f}，p99 {ct['B']['p99_tpot_ms']:.1f} ms）")
        if can_off:
            sc = h["static_sched"]["B"]
            off = f"{sc['output_throughput']:,.0f} tok/s（{sc['output_throughput'] / b - 1:+.1%}），p99 {sc['p99_tpot_ms']:.1f} ms"
        else:
            off = "不该关：饱和时投机仍快约 2 倍"
        rows2.append((eid, {"n": "", "f": V(lab, bold=True), "num": train, "cond": off, "lk": V(f"{eid} →", link=sheet_of[eid])}))
    s.table([("n", "", None), ("f", "场景", None), ("num", "训练 + 热替换（相对\"不训练、只导出\"）", None), ("cond", "饱和段关投机（静态表，不训）", None),
             ("lk", "明细", None)], rows2, caption="表 4　开发期汇总（10-08，控制器 v5.1，B 段 9000 条；版本混用，只作参考）")
    s.note("一句话（10-23 终表）：8B 稠密模型饱和时最大的杠杆是关投机（纯解码 +18% / +97%），v6 自动追平并在代码上靠训练换头保持开；MoE 不能关投机（−46%），"
           "v6 的 +8% 全部来自训练换头；阈值规则在 MoE 上 −46%，说明 frontier 判别是算法的独立贡献；已知切换点的固定训练在可适应域比 v6 高 1–2 点但不会关投机。"
           "vLLM 0.27 只有静态表，运行时开关是给引擎方的具体需求。")


def sheet_config(wb, ctx):
    s = Sheet(wb, "通用配置", [22, 18, 14, 12, 12, 12, 12, 12, 12])
    s.title_("通用配置：所有实验共用的设置", "明细表只写和这里不同的地方。")
    s.section("算力与软件")
    s.kv([("集群", "OSC Cardinal（Ohio Supercomputer Center），账号 PCS0296；工作区 /fs/scratch/PCS0296/specsched/{code,data,models,runs,logs,cache}。"),
          ("GPU", "NVIDIA H100 SXM 94 GB（显存总量 93.6 GB）。单卡作业走 debug 分区（每作业 ≤1 小时、每人同时 2 个）；多卡走 gpu 分区（TP=2/4）。"),
          ("推理引擎", "vLLM 0.27.1（复用 BURSAR 的 bsr312 环境），V1 引擎、V1 model runner（VLLM_USE_V2_MODEL_RUNNER=0），FlashAttention 3，CUDA 12.6.2；"
                    "VLLM_USE_FLASHINFER_SAMPLER=0（FlashInfer 采样核在该环境编不过，贪心用不到）；所有编译缓存放 scratch（家目录配额已满）。"),
          ("服务参数", "vllm serve --max-num-seqs 256 --no-enable-prefix-caching --seed 0，gpu_memory_utilization 0.90（同卡训练时扣掉训练进程预留）；"
                    "短输出 max_model_len 4096、长输出 6144。投机：--speculative-config {method: eagle3, num_speculative_tokens: 3}。"),
          ("压测", "vllm bench serve --backend openai-chat --max-concurrency C --request-rate inf（闭环并发）--temperature 0，每个条件先热身 8 条（C=4）。"),
          ("vLLM 改动", "reload_draft_weights（热替换，PR #60139）、live tap（实时导出）、set_speculation（运行时开关）；在 OSC 上以 apply_*_backport.py 打进 0.27.1，"
                       "主干版本在 feat/spec-live-tap 分支（见 E18）。")])
    s.section("模型")
    s.table([("k", "代号", None), ("t", "目标模型", None), ("h", "草稿头", None), ("aux", "aux 层\n（vLLM 注册号）", None), ("cal", "校准 top-1", F3),
             ("v", "草稿词表", None), ("mem", "训练进程\n预留 GB", INT), ("k2", "k", INT)], [
        ("m1", {"k": V("M1", bold=True), "t": "Qwen3-8B（稠密）", "h": "AngelSlim/Qwen3-8B_eagle3", "aux": "(2, 18, 33)", "cal": 0.588,
                "v": "32k", "mem": 12, "k2": 3}),
        ("m2", {"k": V("M2", bold=True), "t": "Qwen3-30B-A3B（MoE）", "h": "AngelSlim/Qwen3-a3B_eagle3", "aux": "(2, 24, 45)", "cal": 0.617,
                "v": "32k", "mem": 6, "k2": 3})],
        note="M2 的闭环实验预留 5 GB（gpu_memory_utilization 0.8375）。目标模型词表 151936；草稿头只在 32k 草稿词表上出 token。")
    s.section("流量（六个域）")
    rows = []
    for d, ds, how in [("chat", "HuggingFaceH4/ultrachat_200k（test_sft）", "取 20–2000 字符的提示"),
                       ("sql", "b-mc2/sql-create-context", "\"Database schema … Write a SQL query … Return only the SQL\""),
                       ("code", "sahil2801/CodeAlpaca-20k", "instruction（+ input）"),
                       ("zh", "shibing624/alpaca-zh（alpaca_gpt4_data_zh）", "instruction（+ input），4–1500 字符"),
                       ("fc", "glaiveai/glaive-function-calling-v2", "函数列表 + 用户请求，要求只回 JSON 调用"),
                       ("math", "HuggingFaceH4/MATH-500", "题目 + \"请逐步推理，答案放 \\boxed{}\"；思考模式，只用于 C1 长输出")]:
        row = {"d": V(DOM_ZH[d], bold=True), "ds": ds, "how": how}
        p = f"{ROOT}/data/traffic/m1/{d}.jsonl"
        if os.path.exists(p):
            tr = [json.loads(l) for l in open(p)]
            row.update(n=len(tr), pin=float(np.mean([len(r["prompt_token_ids"]) for r in tr])),
                       out=float(np.mean([r["n_out"] for r in tr])),
                       tr=sum(r["finish"] == "length" for r in tr) / len(tr), cov=COVER.get(d))
        else:
            row.update(n="500（只评测）", pin="—", out="≤4096", tr="—", cov="—")
        rows.append((d, row))
    s.table([("d", "域", None), ("ds", "数据集（Hugging Face）", None), ("how", "提示怎么构造", None), ("n", "条数", INT),
             ("pin", "平均提示\ntoken", INT), ("out", "平均输出\ntoken", INT), ("tr", "截断率\n（到 512）", PCTP), ("cov", "草稿词表\n覆盖率", PCTP)],
            rows, caption="表 1　六个域（统计为 M1 生成的流量）",
            note="每域去重后随机打乱（seed 0），取 3200 train（当作按到达顺序的线上流量）+ 300 eval + 300 reserve；目标模型对每条提示贪心生成（非思考、≤512 token），"
                 "输出 token id 落盘。M2 的流量（代码 / SQL / 中文 / 对话）同样生成，平均输出略短。闭环实验另用 make_bench_pool.py 从原始数据集建不重复提示池"
                 "（代码 16222、中文 44776、SQL 74776、函数调用 46982、对话 16383 条，排除了流量里的 3800 条）。")
    s.section("草稿头在线训练（E1 规则）")
    s.kv([("目标", "teacher-forced 在线蒸馏：目标 = 目标模型实际输出的下一个 token，在草稿词表上算交叉熵。"),
          ("优化器", "AdamW，lr 5e-5，betas (0.9, 0.95)，weight decay 0，warmup 10 步，梯度裁剪 1.0；每步约 4096 token，单条最长 2048。"),
          ("单遍在线（C2）", "按到达顺序，每条 train 请求只学一次；消费 N ∈ {100, 200, 400, 800, 1600, 3200} 条后导出 vLLM 可加载的头；记录只含头的 GPU 时间。"),
          ("同卡训练进程（C1 / 闭环）", "load 模式：在缓存的 hidden states 上反复训（C1、E10、E11）；live 模式：从 live tap 环形缓冲拼装完成的请求，"
                                    "CPU 侧后进先出缓冲 ≤60 万 token、60 秒保鲜（E12–E15）。占空比 d：每训完一步 sleep dt·(1−d)/d；d 可由 --duty-file 每步重读。")])
    s.section("测量口径")
    s.kv([("吞吐", "vllm bench serve 的 output_throughput（输出 token/s）；模拟器与 C2 另用 goodput / 每请求 GPU 秒。"),
          ("延迟", "TPOT（每输出 token 的时间）的 p50 / p99；TTFT 有记录但不作主指标。"),
          ("接受长度", "1 + 接受的草稿 token 数 ÷ 草稿次数（vLLM /metrics 的 spec_decode 计数器差分），k=3 时取值 1–4。"),
          ("SLO", "Rung 1 起：TPOT ≤ 16.7 ms（= 2 × 并发 1 时的不投机步时）且排队 ≤1 s；控制器规则 2 用 0.9 × SLO = 15 ms 作余量线。"),
          ("C2 饱和", "600 条（eval + reserve）一次提交，测 3 次取中位数；E05 另有固定长度口径。"),
          ("闭环剖面", "A1 对话 C=16 120 条 → B 新域 C=256 9000 条 → A2 对话 C=16 120 条；B 段吞吐为 bench 结果，接受长度来自 5 秒一次的 /metrics 采样。")])
    s.section("各阶段算力")
    rows = []
    for st in ["Rung 0", "Rung 1", "demo 冲刺"]:
        js = [j for j in ctx["jobs"] if STAGE.get(j["exp"]) == st]
        rows.append((st, {"st": st, "n": F(f"=COUNTIF('作业清单'!$D:$D,\"{st}\")", len(js), INT),
                          "gh": F(f"=SUMIF('作业清单'!$D:$D,\"{st}\",'作业清单'!$J:$J)", sum(j["gh"] for j in js), F1)}))
    s.table([("st", "阶段", None), ("n", "作业数", INT), ("gh", "H100 卡时", F1)], rows, caption="表 2　按阶段汇总（公式，来自「作业清单」）")


TERMS = [
    ("投机解码", "小模型（草稿）先猜几个 token，大模型（目标）一次并行验证，猜对的直接收下。输出与不投机逐 token 相同（贪心下），只是更快或更慢。", "—"),
    ("目标模型 / 草稿头", "目标模型 = 真正要服务的大模型；草稿头 = EAGLE-3 的小头，读目标模型几层的 hidden states 来猜下一个 token。", "M1、M2 见「通用配置」"),
    ("k（草稿长度）", "每步草稿头猜几个 token。本项目固定 k=3。", "—"),
    ("接受长度 a", "每步平均收下几个 token（含目标模型自己出的 1 个），k=3 时在 1–4 之间。越大越快。", "1 + 接受数 ÷ 草稿次数"),
    ("并发 C", "同时在跑的请求数（闭环压测：完成一个补一个）。C 大 = 负载高、batch 大。", "--max-concurrency"),
    ("饱和", "需求超过系统能力，请求开始排队；本项目里 C=256（max-num-seqs）就是饱和。", "—"),
    ("吞吐", "每秒输出多少 token（tok/s）。", "output_throughput"),
    ("TPOT / p50 / p99", "每输出一个 token 要多少毫秒；p50 是中位数，p99 是最慢的 1%（尾延迟）。", "bench 统计"),
    ("SLO / goodput", "服务承诺的延迟线（这里 TPOT ≤ 16.7 ms）；goodput = 满足 SLO 时服务的 token。", "见「通用配置」"),
    ("S0 / S1 / S2 / S3", "C1 的四个条件：不投机 / 投机 / 不投机 + 同卡训练 / 投机 + 同卡训练。", "E02"),
    ("同卡训练（共置）", "训练草稿头的进程和推理服务跑在同一张 GPU 上，抢 GPU 时间。", "E02、E03"),
    ("占空比 d", "训练进程占用 GPU 时间的比例（0 = 不训，1 = 一直训）。服务损失 ≈ κ(C) × d，κ≈0.5。", "sleep 时间片"),
    ("MPS", "NVIDIA 的多进程空间共享（多个进程同时用 SM），对比默认的时间片轮转。", "E03"),
    ("N（已学请求数）", "在线更新时已经学过多少条线上请求；N0 = 原装头。", "E04"),
    ("回本时间", "训练花掉的 GPU 时间要服务多少请求才能省回来。", "E04"),
    ("草稿词表覆盖率", "目标模型输出的 token 有多少落在草稿头的 32k 词表里；落在外面的草稿头永远猜不中。", "中文只有 42.6%"),
    ("天花板 / 结构天花板", "在线学习能把接受长度推到的上限；如果上限都到不了盈亏线（如中文），训了也白训。", "E06、E09、E14"),
    ("frontier / gain(a, C)", "投机比不投机快多少倍，作为接受长度 a 和并发 C 的函数；本项目里对 a 是线性的。", "E06"),
    ("盈亏接受长度 a*", "让投机刚好不亏的接受长度；a < a* 时投机比不投机还慢，该关。", "a* = (1−β)/α 或 T_spec/T_base"),
    ("T_base / T_spec", "不投机 / 投机时每步解码耗时；T_spec 与头无关。", "E02、E06"),
    ("热替换", "不停服务，把新训好的草稿头权重换进正在运行的引擎。", "reload_draft_weights"),
    ("live tap（实时导出）", "推理时顺手把训练草稿头需要的 hidden states + token 写到共享内存环形缓冲，供同卡训练进程实时读取。", "E12"),
    ("静态表", "vLLM 已有的按批大小设定投机长度的表（如并发 >64 时 k=0），启动时定死，不能随头变。", "num_speculative_tokens_per_batch_size"),
    ("运行时开关", "本项目给 vLLM 加的 set_speculation：服务中随时开 / 关投机，≤10 ms 生效。", "E16"),
    ("控制器规则 0–3", "0 停：学到头或结构天花板；1 按 frontier 开关投机；2 SLO 内取最大免费占空比；3 饱和时按价值选占空比。", "controller.py"),
    ("预测器 oracle / persist / lindy", "调度器猜\"当前状态还会持续多久\"：真未来 / 当前延续 / 再持续已持续的时长。", "E09"),
    ("A1 / B / A2", "闭环实验的三段流量：对话（低负载）→ 新域（饱和）→ 对话。", "E11–E15"),
    ("唯一池 vs 300 条循环池", "压测提示是否重复。循环池会让在线学习器\"背下\"提示，接受长度虚高（记忆泄漏）。", "见「作废与教训」"),
    ("TP（张量并行）", "一个模型切到多张卡上一起算；TP=2 / TP=4 = 2 / 4 张卡。", "E17"),
    ("batch-invariance", "同一个请求在不同批组合下算出完全相同的数。TP 内核不保证这一点，所以贪心输出可能在平局处翻一个 token。", "E17"),
]


def sheet_terms(wb):
    s = Sheet(wb, "术语", [24, 70, 30])
    s.title_("术语表", "按在表里出现的顺序排列。")
    s.table([("t", "术语", None), ("d", "大白话", None), ("h", "本表里怎么算 / 在哪", None)],
            [(t, {"t": V(t, bold=True), "d": d, "h": h}) for t, d, h in TERMS])


def sheet_jobs(wb, jobs):
    s = Sheet(wb, "作业清单", [11, 17, 7, 11, 7, 18, 10, 10, 8, 9, 8, 60])
    s.title_("作业清单：OSC 上的全部作业", f"共 {len(jobs)} 个作业（2026-10-04 至 10-10，悉尼时间），sacct 导出；卡时 = 时长（秒）× GPU 数 ÷ 3600（公式）。")
    rows = []
    for j in sorted(jobs, key=lambda x: x["id"]):
        rows.append((j["id"], {"id": j["id"], "name": j["name"], "e": j["exp"], "st": STAGE[j["exp"]], "p": j["part"],
                               "t": j["start"].strftime("%Y-%m-%d %H:%M") if j["start"] else "未运行", "el": j["el"], "s": j["secs"],
                               "g": j["ngpu"], "gh": F("={s}{r}*{g}{r}/3600", j["gh"], F2),
                               "state": V(j["state"], status=("有效" if j["state"] == "完成" else "取消")), "w": j["what"]}))
    s.table([("id", "作业号", None), ("name", "作业名", None), ("e", "实验", None), ("st", "阶段", None), ("p", "分区", None),
             ("t", "开始（悉尼）", None), ("el", "时长", None), ("s", "时长（秒）", INT), ("g", "GPU 数", INT), ("gh", "卡时", F2),
             ("state", "状态", None), ("w", "做了什么", None)], rows)
    s.ws.freeze_panes = "C5"
    s.ws.auto_filter.ref = f"A4:L{s.r - 1}"


VOID = [
    ("frontier 第一轮", "15263307、15263308", "E06", "OSC debug 分区两个单卡作业可能落在同一节点、共享网络命名空间；两个 vllm serve 都用 8123 端口时，"
     "一个作业的 bench 打到了另一个作业的服务上（结果 = 对方的曲线、没有投机指标）。", "端口改为 8000 + 作业号 % 1000，启动前探测 /health 必须不通；第一轮作废重跑（15263570/71）。"
     "10-04 的结果核对过未受影响。", "作废"),
    ("真机控制器 v1", "15263716（costmodel_v1_ungated）", "E10", "规则 3 移植时被简化成\"天花板超过盈亏线就训\"，C=256 时全速训：吞吐 −54%、TPOT 45 ms。"
     "对话天花板 2.37 在 256 路时 TPOT 仍超 SLO，且当时没有热替换，违约时训练永远兑现不了。", "门控：无热替换时违约不训；有热替换也只在天花板能让 TPOT 回到 SLO 内时训。重跑 15264390。"
     "v1 结果保留在 E10 表里作对照。", "作废"),
    ("热替换第一版", "15283467、15283468（*_v1_trainer_never_stops）", "E11", "训练进程吃完缓存后又从头开始、不退出：B 段 train_swap 12919、train_noswap 10626 tok/s，"
     "都低于 static 的 14461。", "训练进程最后一次导出后退出；\"训完就停\"写进调度器规则。重跑 15284127。", "作废"),
    ("实时闭环第一版（300 条循环池）", "15342983（*_v1_pool300）", "E13", "bench 提示池只有 300 条，B 段 9000 条请求循环 30 遍；在线学习器学的正是正在服务的请求，"
     "7 秒后就见过全部提示，之后接受长度 3.1+ 超过离线天花板 2.72——记忆泄漏。B 段 16190 tok/s（+12%）不能用。",
     "make_bench_pool.py 建不重复提示池（每域 1.6 万–7.5 万条），三种模式用同一池重跑。同一作业里的 static_live 不训练，仍用于导出开销（E12）。", "作废"),
    ("控制器 v2", "15343658（controller_live_v2）", "E13", "四个毛病：域标签比流量早到导致视界虚高；饱和判别 C<250 抖动；换域后缓冲里的旧域数据被吃掉、连发换头；"
     "按消费数定位学习曲线高估剩余增益。结果 +14.9%，换头 13 次。", "改成 v3/v4 重跑（15344614）。v2 结果保留在 E13 作版本对比。", "对照"),
    ("控制器 v5（MoE 轮）", "15345625", "E15", "v5 在代码域缓冲积多时选占空比 1.0 爆训，p99 变差；MoE 的 v5 轮在跑完前撤下。",
     "加饱和段占空比 ≤0.5 护栏成 v5.1，三组重跑（15345869/70/15345940）。", "取消"),
    ("运行时开关 worker 单侧版", "15347807、15347872（及 15347808/09/74）", "E16", "只让 worker 不出草稿，调度器仍按 padded 模式给每个请求预留 k 个草稿槽：关后吞吐 14168 → 7633（−46%），"
     "接受长度 1.00、草稿计数翻倍；第一次提交还触发引擎断言崩溃。", "调度器 + worker 两半同步切（EngineCore 拦截 set_speculation）；重跑 15348478、15349223。", "作废"),
    ("时间线采样器", "10-06 起的 hotswap 时间线", "E11–E15", "vllm:generation_tokens 实际名为 vllm:generation_tokens_total，早期 fallback 把直方图系列一起加总，"
     "时间线上的 gen_tok_s 约虚高 3 倍。", "只影响时间线绝对值（结论文档里的\"稳态 43.9k / 54.7k\"），bench 的吞吐不受影响；已改为优先取精确计数器。", "参考"),
    ("C2 饱和的变长口径", "E04", "E04", "600 条一次提交含排空段（后半段并发低、投机占优），偏向投机，且对输出长度分布敏感（代码 N800 的 +11% 是输出集效应）。",
     "补固定长度口径（E05）；变长口径里几个百分点的差别一律读作打平。", "参考"),
    ("预注册的加性交互口径", "E02", "E02", "H3 的加性交互项受尺度混淆（训练代价本身随负载变化），会误判。", "与乘法口径并列报告，结论改述为\"只在中高负载争\"。", "参考"),
    ("预注册的 goodput SLO（C1）", "E02", "E02", "TPOT ≤ 并发 1 时 S0 中位数的 2 倍，对 MoE 过苛（MoE 不投机的 TPOT 本身随 batch 陡升）。",
     "C1 主指标改用吞吐与 P99 TPOT，goodput 只作参考。", "参考"),
    ("中文\"拒训\"剧本", "E14", "E14", "原计划：中文天花板到不了盈亏线，控制器应拒训。实测引擎关不掉投机时照训也 +12.5%。",
     "控制器加 --spec-switchable：引擎能关投机时先关、再判定不训；不能关时走价值规则照训。", "参考"),
    ("SQL 饱和\"翻盘\"", "E09", "E09", "模拟器在看到结果后修过三处建模、三处规则；修正后 SQL 饱和翻盘取决于接受长度差 0.04。", "从\"稳健\"降为\"边缘\"，不作为结论。", "参考"),
    ("文档勘误：SQL 不投机单请求延迟", "E04", "E04", "Rung 0 结论文档 §2.1 表里写成 5.9 ms/token；原始数据 data/runs/c2/m1/sql/none/eval.json 为 8.70。",
     "本表以原始数据为准（N0 为 5.84、N3200 为 4.28 与文档一致）。", "参考"),
    ("文档勘误：函数调用 N3200 单请求延迟", "E04", "E04", "Rung 0 结论文档 §2.1 写成 4.3 ms/token；原始数据为 4.56。", "本表以原始数据为准。", "参考"),
    ("未运行即取消的作业", "15249092、15249093、15343331、15343332、15343234、15344905、15347809、15348491、15347875 等",
     "—", "重复提交或被后续方案替代，没有结果。", "见「作业清单」。", "取消"),
]


def sheet_void(wb, ctx):
    s = Sheet(wb, "作废与教训", [24, 22, 8, 50, 50, 9])
    s.title_("作废与教训", "这些结果不进正式结论，或需要带着限定条件读；文档里与原始数据不一致的地方也列在这里（以原始数据为准）。")
    e3, e4 = ctx["e03"], ctx["e04"]
    dfc, dco = e4["dfs"]["fc"], e4["dfs"]["code"]

    def at(df, N, col):
        return float(df[df.N == N][col].iloc[0])
    fc_peak = max((at(dfc, N, "acc_len"), N) for N in (100, 200, 400, 800, 1600, 3200))
    errata = [
        ("文档勘误：回本时间的口径", "E04", "E04",
         "Rung 0 结论文档 §2.3 里代码域\"0.07 → 0.97 分钟\"用的是\"两次饱和测量相减\"口径，其余三个域用的是接受长度模型口径，两种口径混在一张表里。",
         f"本表统一用接受长度模型口径（代码 {at(dco, 100, 'Tbe_min_accmodel'):.2f} → {at(dco, 3200, 'Tbe_min_accmodel'):.2f} 分钟）；"
         f"结论不变：N200 时域间最大/最小 {e4['r200']:.1f}×，H4 不成立。", "参考"),
        ("文档勘误：函数调用的数字", "E04", "E04",
         "Rung 0 结论文档写结论时函数调用只评测到 N800（\"2.27 → ~2.65\"、对话 2.04、回本 0.08 → 0.5 分钟、\"全部 2 分钟以内\"）。",
         f"补齐后的原始数据：N3200 接受长度 {at(dfc, 3200, 'acc_len'):.2f}（峰值 {fc_peak[0]:.2f}@N{fc_peak[1]}），对话迁移 "
         f"{at(dfc, 3200, 'acc_len_chat'):.2f}（比原装头 {e4['chat_d']['fc']:+.2f}），一直开口径回本 {at(dfc, 3200, 'Tbe_min_accmodel'):.2f} 分钟；"
         f"本表据此写成\"回本全在 {e4['tmax']:.1f} 分钟内\"。", "参考"),
        ("文档勘误：占空比代价", "E03", "E03",
         "结论文档写\"服务损失 ≈ 占空比 × 50%\"。实测只有占空比 0.5–1 时接近这个比例。",
         f"占空比 0.1 只损失 {e3['loss'][0.1][0]:.0%}–{e3['loss'][0.1][1]:.0%}、0.25 损失 {e3['loss'][0.25][0]:.0%}–{e3['loss'][0.25][1]:.0%}，"
         "低占空比比线性还便宜；本表按实测写，模拟器的 κ 拟合不受影响。", "参考"),
    ]
    if "e21" in ctx:
        v = ctx["e21"]
        errata += [
            ("模拟结论改弱：突发多久值得训", "E09 → E21", "E21",
             "Rung 1 结论文档写\"反应式策略 1 秒翻盘、≥3 s 就值得训\"，来自单一场景（A=代码已适应、B 饱和需求 = 不投机容量）。",
             f"冻结模拟器在 60 个随机突发上：≤20 s 的突发多数平手（严格赢 {min(v['V2']['by_dur'][k]['strict_win'] for k in ('5', '10', '20')):.0%}–"
             f"{max(v['V2']['by_dur'][k]['strict_win'] for k in ('5', '10', '20')):.0%}），60 s 中位 {v['V2']['by_dur']['60']['median']:+.1%}，180 s 起可靠（中位 "
             f"{v['V2']['by_dur']['180']['median']:.2f}×）。论文写\"训练不吃亏；≥60 s 的突发可靠回本\"。", "参考"),
            ("模拟结论改弱：中文天花板的幅度", "E09 → E21", "E21",
             "结论文档写中文场景\"反应式策略/投机常开 goodput 砍半（0.48 vs 0.96M）\"，是单一场景数字。",
             f"40 个随机中文场景排序全部成立，幅度：可行饱和 {v['V3']['feasible_sat']['min']:.2f}–{v['V3']['feasible_sat']['max']:.2f}×，1.1× 过载时只有 "
             f"{v['V3']['overload']['min']:.2f}–{v['V3']['overload']['max']:.2f}×（过载期谁都违约）。论文写\"1.4–2×（饱和且可行）\"。", "参考"),
            ("模拟结论改弱：抗浪涌不是独有", "E09 → E21", "E21",
             "结论文档写\"cost-model 平静期免费预训练能扛 1.1× 浪涌，OSD/accdrop 只能扛 0.95×\"。",
             f"20 个随机浪涌场景：accdrop 平均能扛 {v['V4']['max_surge_mean']['accdrop']:.2f}×，与 cost-model（{v['V4']['max_surge_mean']['costmodel_lindy']:.2f}×）相当，"
             f"OSD {v['V4']['max_surge_mean']['osd_idle']:.2f}×。cost-model 的独有之处是四条判据同时成立，不是抗浪涌。", "参考"),
            ("文档勘误：E08 cost-model 的 goodput", "E08", "E08",
             "Rung 1 结论文档 §2.4 写 cost-model 246.3M，是加\"按域 frontier\"之前的模拟器版本。",
             "冻结版 sim.py 重跑为 253.4M（静态 228.8 / 投机常开 215.6 / OSD 251.1 与文档一致）；E08 表已用 253.4。", "参考"),
        ]
    items = VOID[:-1] + errata + VOID[-1:]
    s.table([("k", "项目", None), ("j", "作业 / 位置", None), ("e", "实验", None), ("w", "问题", None), ("fix", "怎么处理", None), ("v", "状态", None)],
            [(str(i), {"k": V(k, bold=True), "j": j, "e": e, "w": w, "fix": fx, "v": V(v, status=v)}) for i, (k, j, e, w, fx, v) in enumerate(items)])
    s.section("方法论教训")
    s.bullets(["饱和吞吐\"一批请求一次提交\"的口径含排空段，偏向投机；长输出域用固定长度口径，短输出域两种口径都不干净，要并列报告。",
               "在线学习实验的压测提示必须不重复，而且要和训练数据分开，否则测到的是记忆。",
               "OSC 同节点作业共享网络命名空间：端口要按作业号错开，启动前探测。",
               "任何 \"训练进程\" 都要有退出条件；控制器要有\"学到头就停\"。",
               "预注册的判据要在看到数据前想清楚尺度问题（加性 vs 乘法）。",
               "模拟实验调过参的轨迹不能再当证据：冻结模拟器和规则后，在新轨迹上按预先写定的判据重跑，才能保留结论（E21）。",
               "一篇论文的所有实验用同一个冻结 checkpoint（代码、补丁、头、环境、提示池）；版本混用的数字只当开发记录（E19 之前的闭环）。"])


# =============================================================================================== formula check + cached values
def to_python(formula, sheet):
    """Translate one of our formulas (arithmetic, cell refs, ranges, SUM/AVERAGE/MAX/MIN/COUNT/COUNTIF/SUMIF) to Python."""
    store = []

    def keep(s):
        store.append(s)
        return f"\u00a7{len(store) - 1}\u00a7"
    e = formula[1:]
    e = re.sub(r"'([^']+)'!(\$?[A-Z]+\$?\d*:\$?[A-Z]+\$?\d*)", lambda m: keep(f"__R({m.group(1)!r},{m.group(2)!r})"), e)
    e = re.sub(r"'([^']+)'!(\$?[A-Z]+\$?\d+)", lambda m: keep(f"__C({m.group(1)!r},{m.group(2).replace('$', '')!r})"), e)
    e = re.sub(r'"[^"]*"', lambda m: keep(m.group(0)), e)
    e = re.sub(r"\$?[A-Z]+\$?\d+:\$?[A-Z]+\$?\d+", lambda m: keep(f"__R({sheet!r},{m.group(0)!r})"), e)
    e = re.sub(r"(?<![A-Za-z_])\$?[A-Z]{1,3}\$?\d+(?![\w(])", lambda m: keep(f"__C({sheet!r},{m.group(0).replace('$', '')!r})"), e)
    return re.sub("\u00a7(\\d+)\u00a7", lambda m: store[int(m.group(1))], e)


def evaluate_all(path):
    """Evaluate every formula independently (own evaluator over the stored inputs) and compare with REG."""
    wb = load_workbook(path)

    def cell_value(sheet, coord):
        if (sheet, coord) in REG:
            return REG[(sheet, coord)]
        v = wb[sheet][coord].value
        if isinstance(v, str) and v.startswith("="):
            raise ValueError(f"formula without registered value: {sheet}!{coord}")
        return v

    def rng(sheet, ref):
        a, b = ref.replace("$", "").split(":")
        if a.isalpha():   # whole column: every row, aligned across columns (SUMIF zips two of them)
            return [cell_value(sheet, f"{a}{r}") for r in range(1, wb[sheet].max_row + 1)]
        c0, r0 = re.match(r"([A-Z]+)(\d+)", a).groups()
        c1_, r1 = re.match(r"([A-Z]+)(\d+)", b).groups()
        return [cell_value(sheet, f"{get_column_letter(c)}{r}") for r in range(int(r0), int(r1) + 1)
                for c in range(column_index_from_string(c0), column_index_from_string(c1_) + 1)]

    def nums(xs):
        return [x for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]
    def stdev(xs):
        v = nums(xs)
        m = sum(v) / len(v)
        return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
    env = dict(STDEV=stdev, SUM=lambda xs: sum(nums(xs)), AVERAGE=lambda xs: sum(nums(xs)) / len(nums(xs)), MAX=lambda xs: max(nums(xs)),
               MIN=lambda xs: min(nums(xs)), COUNT=lambda xs: len(nums(xs)),
               COUNTIF=lambda xs, c: sum(1 for x in xs if x == c),
               SUMIF=lambda xs, c, ys: sum(y for x, y in zip(xs, ys) if x == c and isinstance(y, (int, float))))
    bad = []
    for (sheet, coord), want in REG.items():
        f = wb[sheet][coord].value
        assert isinstance(f, str) and f.startswith("="), (sheet, coord, f)
        expr = to_python(f, sheet)
        try:
            got = eval(expr, {"__builtins__": {}}, {**env, "__R": rng, "__C": cell_value})  # noqa: S307 - our own formulas
        except Exception as ex:   # noqa: BLE001
            bad.append((sheet, coord, f, expr, f"error {ex}"))
            continue
        if not math.isclose(float(got), want, rel_tol=1e-9, abs_tol=1e-9):
            bad.append((sheet, coord, f, expr, f"got {got} want {want}"))
    return bad


def write_cached_values(path):
    """openpyxl leaves <v></v> empty for formula cells; fill in the checked values so previews show numbers."""
    z = zipfile.ZipFile(path)
    names = z.namelist()
    files = {n: z.read(n) for n in names}
    z.close()
    wbx = files["xl/workbook.xml"].decode()
    rels = files["xl/_rels/workbook.xml.rels"].decode()
    rid2target = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', rels)) or \
        {m[1]: m[0] for m in re.findall(r'Target="([^"]+)"[^>]*Id="(rId\d+)"', rels)}
    for name, rid in re.findall(r'<sheet[^>]*name="([^"]+)"[^>]*r:id="(rId\d+)"', wbx):
        import html
        name = html.unescape(name)
        target = rid2target[rid].lstrip("/")
        target = target if target.startswith("xl/") else f"xl/{target}"
        xml = files[target].decode()
        todo = {coord: v for (sh, coord), v in REG.items() if sh == name}
        n = 0

        def fix(m):
            nonlocal n
            coord = m.group(2)
            if coord not in todo:
                return m.group(0)
            n += 1
            return f"{m.group(1)}<f>{m.group(3)}</f><v>{repr(float(todo[coord]))}</v></c>"
        xml = re.sub(r'(<c r="([A-Z]+\d+)"[^>]*>)<f>(.*?)</f><v\s*/?>(?:</v>)?</c>', fix, xml)
        assert n == len(todo), (name, n, len(todo))
        files[target] = xml.encode()
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:
        for n_ in names:
            zo.writestr(n_, files[n_])
    os.replace(tmp, path)


def check_and_cache(path):
    bad = evaluate_all(path)
    if bad:
        for b in bad[:20]:
            print("FORMULA MISMATCH", b)
        raise SystemExit(f"{len(bad)} formula(s) do not evaluate to the computed value")
    write_cached_values(path)
    print(f"checked {len(REG)} formulas; cached values written")


# =============================================================================================== main
def main():
    wb = Workbook()
    wb.remove(wb.active)
    ctx = dict(jobs=load_jobs())
    exps = build_experiments(ctx)
    sheet_readme(wb, ctx, exps)
    sheet_overview(wb, ctx, exps)
    sheet_core(wb, ctx, exps)
    sheet_config(wb, ctx)
    sheet_terms(wb)
    for e in exps:
        e["build"](wb, ctx, e)
    sheet_jobs(wb, ctx["jobs"])
    sheet_void(wb, ctx)
    wb.active = 0
    wb.save(OUT)
    check_and_cache(OUT)
    print("wrote", OUT, "|", len(wb.sheetnames), "sheets:", ", ".join(wb.sheetnames))


if __name__ == "__main__":
    main()
