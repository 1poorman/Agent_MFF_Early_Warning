"""接口文档 Markdown -> PDF 转换脚本（reportlab + 系统中文字体）。

用法：
    conda run -n mff_agent python scripts/gen_api_docs_pdf.py
输出：
    docs/中频炉预警智能体 OpenAPI 接口文档.pdf
    docs/中频炉预警智能体 MCP 接口文档.pdf

排版规范（按赛事文档要求）：
    - 全文档仅使用黑色字体
    - 中文正文：宋体（simsun）小四（12pt）
    - 中文标题/表头：黑体（simhei）
    - 英文/数字：Times New Roman（系统无微软字体，用度量完全兼容的开源替代
      Liberation Serif 注册为 Times New Roman 使用）

依赖：reportlab（pip install reportlab）；字体文件：
    /usr/share/fonts/chinese/simsun.ttc
    /usr/share/fonts/chinese/simhei.ttf
    /usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

# 中文字体（按顺序尝试注册）
FONT_CANDIDATES = [
    ("Song", "/usr/share/fonts/chinese/simsun.ttc", "宋体"),
    ("Song", "/usr/share/fonts/chinese/simhei.ttf", "黑体"),
    ("Hei", "/usr/share/fonts/chinese/simhei.ttf", "黑体"),
]
# 西文 Times New Roman：Liberation Serif（度量兼容替代）
SERIF_CANDIDATES = [
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
]
SERIF_BOLD_CANDIDATES = [
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
]

FONT = "Song"        # 中文正文字体（宋体）
FONT_BOLD = "Hei"    # 中文标题/表头字体（黑体）
FONT_EN = "TimesNewRoman"   # 英文/数字字体
FONT_EN_BOLD = "TimesNewRoman-Bold"


def register_fonts() -> bool:
    """注册中文字体与英文 Times New Roman（Liberation Serif 替代），成功返回 True。"""
    reg = {}
    for key, path, _ in FONT_CANDIDATES:
        if not Path(path).exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont(key, path))
            reg[key] = True
        except Exception:
            continue
    # 注册西文 Times New Roman（含粗体）
    for candidate, key in ((SERIF_CANDIDATES, FONT_EN),
                           (SERIF_BOLD_CANDIDATES, FONT_EN_BOLD)):
        for path in candidate:
            if Path(path).exists():
                try:
                    pdfmetrics.registerFont(TTFont(key, path))
                    reg[key] = True
                    break
                except Exception:
                    continue
    if "Song" not in reg and "Hei" not in reg:
        return False
    global FONT, FONT_BOLD
    FONT = "Song" if "Song" in reg else "Hei"
    FONT_BOLD = "Hei" if "Hei" in reg else "Song"
    return True


# ---------------- Markdown 解析 ----------------

def is_table_row(line: str) -> bool:
    return bool(re.match(r"^\s*\|", line)) and line.count("|") >= 2


def split_cells(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_markdown(text: str) -> List[Tuple[str, object]]:
    """解析 Markdown 为块列表：("h1"|"h2"|"h3"|"p"|"code"|"table"|"list"|"hr"|"quote", 内容)。"""
    blocks: List[Tuple[str, object]] = []
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        s = line.strip()

        # 代码块
        if s.startswith("```"):
            buf = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1  # 跳过结束 ```
            blocks.append(("code", "\n".join(buf)))
            continue

        # 表格
        if is_table_row(line) and i + 1 < n and re.match(r"^\s*\|[\s:\-|]+\|\s*$", lines[i + 1]):
            header = split_cells(line)
            i += 2
            rows = []
            while i < n and is_table_row(lines[i]):
                rows.append(split_cells(lines[i]))
                i += 1
            blocks.append(("table", (header, rows)))
            continue

        # 标题
        m = re.match(r"^(#{1,4})\s+(.*)", s)
        if m:
            level = len(m.group(1))
            blocks.append((f"h{level}", m.group(2).strip()))
            i += 1
            continue

        # 分隔线
        if re.match(r"^-{3,}$", s) or re.match(r"^\*{3,}$", s):
            blocks.append(("hr", None))
            i += 1
            continue

        # 引用
        if s.startswith(">"):
            blocks.append(("quote", s.lstrip("> ").strip()))
            i += 1
            continue

        # 列表
        if re.match(r"^[-*]\s+", s):
            items = []
            while i < n:
                t = lines[i].strip()
                if re.match(r"^[-*]\s+", t):
                    items.append(re.sub(r"^[-*]\s+", "", t))
                    i += 1
                else:
                    break
            blocks.append(("list", items))
            continue

        # 空行
        if not s:
            i += 1
            continue

        # 普通段落（合并连续非空行）
        buf = [line]
        i += 1
        while i < n and lines[i].strip() and not is_table_row(lines[i]) \
                and not lines[i].strip().startswith("```") \
                and not re.match(r"^(#{1,4})\s", lines[i]) \
                and not re.match(r"^[-*]\s+", lines[i].strip()):
            buf.append(lines[i])
            i += 1
        blocks.append(("p", "\n".join(buf)))

    return blocks


# ---------------- 行内格式 ----------------

INLINE_RE = re.compile(r"(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)")
# ASCII 片段（英文/数字/符号），用于自动切 Times New Roman
_EN_RE = re.compile(r"[A-Za-z0-9_./:@\-+%()\[\]{}|,;=<>\u00b5\u00b0]+")
# 占位符：每个 stash 元素用唯一的私有区字符 (0xE000+i)，不含 ASCII，不会被 _EN_RE 误匹配
_PH_BASE = 0xE000


def _ph(i: int) -> str:
    return chr(_PH_BASE + i)


def _wrap_en(text: str, bold: bool = False) -> str:
    """将连续英文/数字片段包成 Times New Roman 字体标签。"""
    face = FONT_EN_BOLD if bold else FONT_EN

    def _m(m):
        return f'<font face="{face}">{m.group(0)}</font>'
    return _EN_RE.sub(_m, text)


def inline_html(text: str, bold: bool = False) -> str:
    """行内 markdown 转 HTML；中文保持当前样式字体（宋体/黑体），英文自动 Times New Roman。

    关键：行内 code/bold/italic 结构内部先切英文，避免被后续外层 _wrap_en 二次包裹。
    """
    # 1. 解析结构标记：对每个匹配内部先切英文，组装为完整 font 标签作为占位符
    stash = []  # 每项 = (原始 markdown, 已处理好的 HTML 片段)

    def _stash(m):
        orig = m.group(0)
        if orig.startswith("`"):                          # 行内代码：外层 Times New Roman 即可
            html = f'<font face="{FONT_EN}">{orig[1:-1]}</font>'
        elif orig.startswith("**"):                       # 加粗：中文黑体 + 英文 TNR-Bold
            inner = _wrap_en(orig[2:-2], True)
            html = f'<font face="{FONT_BOLD}">{inner}</font>'
        else:                                             # 斜体：中文宋体 + 英文 TNR
            inner = _wrap_en(orig[1:-1], bold)
            html = f'<font face="{FONT}">{inner}</font>'
        stash.append((orig, html))
        return _ph(len(stash) - 1)

    t = INLINE_RE.sub(_stash, text)
    # 2. 对外层文本（不含结构标记）切英文
    t = _wrap_en(t, bold)
    # 3. 还原占位符为已处理好的 HTML
    for i, (_orig, html) in enumerate(stash):
        t = t.replace(_ph(i), html)
    return t


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------- 样式 ----------------

def build_styles():
    """全黑色字体；中文宋体小四(12pt)正文、黑体标题；英文/数字 Times New Roman。"""
    ss = getSampleStyleSheet()
    BLACK = colors.black
    st = {
        # 标题：黑体，黑色（英文自动用 Times New Roman-Bold）
        "h1": ParagraphStyle("h1", parent=ss["Title"], fontName=FONT_BOLD, fontSize=16,
                             leading=21, spaceAfter=6, spaceBefore=0, textColor=BLACK),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName=FONT_BOLD, fontSize=14,
                             leading=19, spaceBefore=12, spaceAfter=5, textColor=BLACK),
        "h3": ParagraphStyle("h3", parent=ss["Heading3"], fontName=FONT_BOLD, fontSize=12,
                             leading=16, spaceBefore=9, spaceAfter=4, textColor=BLACK),
        "h4": ParagraphStyle("h4", parent=ss["Heading4"], fontName=FONT_BOLD, fontSize=11,
                             leading=15, spaceBefore=7, spaceAfter=3, textColor=BLACK),
        # 正文：宋体小四(12pt)，黑色
        "p": ParagraphStyle("p", parent=ss["BodyText"], fontName=FONT, fontSize=12,
                            leading=19, spaceAfter=5, alignment=TA_LEFT, textColor=BLACK),
        "quote": ParagraphStyle("quote", parent=ss["BodyText"], fontName=FONT, fontSize=12,
                                leading=19, leftIndent=10, textColor=BLACK,
                                spaceAfter=5),
        # 代码块：宋体 9.5pt（保留中文）+ 浅灰底，边框黑色，自动换行
        "code": ParagraphStyle("code", parent=ss["Code"], fontName=FONT, fontSize=9.5,
                               leading=13.5, spaceAfter=6, backColor=colors.HexColor("#f5f7fa"),
                               borderColor=BLACK, borderWidth=0.4,
                               borderPadding=6, wordWrap="CJK", textColor=BLACK),
        # 表格：正文 11pt（略小于正文以容纳表格）
        "cell": ParagraphStyle("cell", parent=ss["BodyText"], fontName=FONT, fontSize=11,
                               leading=15, textColor=BLACK),
        "cellh": ParagraphStyle("cellh", parent=ss["BodyText"], fontName=FONT_BOLD, fontSize=11,
                                leading=15, textColor=BLACK),
    }
    return st


def render_blocks(blocks, styles) -> List:
    """块 -> Platypus 流对象。"""
    flow = []
    ST = styles

    for kind, content in blocks:
        if kind == "h1":
            flow.append(Paragraph(inline_html(esc(content), bold=True), ST["h1"]))
        elif kind == "h2":
            flow.append(Paragraph(inline_html(esc(content), bold=True), ST["h2"]))
        elif kind == "h3":
            flow.append(Paragraph(inline_html(esc(content), bold=True), ST["h3"]))
        elif kind == "h4":
            flow.append(Paragraph(inline_html(esc(content), bold=True), ST["h4"]))
        elif kind == "p":
            flow.append(Paragraph(inline_html(esc(content)), ST["p"]))
        elif kind == "quote":
            flow.append(Paragraph(inline_html(esc(content)), ST["quote"]))
        elif kind == "list":
            for item in content:
                flow.append(Paragraph("• " + inline_html(esc(item)), ST["p"]))
        elif kind == "code":
            # 代码块：中文字体 + Paragraph 自动换行（中文正常显示、长行不超界）
            code_txt = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            # 空格转 &nbsp; 保持缩进（Paragraph 会折叠连续空格）
            code_txt = re.sub(r"^ ", "&nbsp;", code_txt, flags=re.M)
            code_txt = re.sub(r"(?<=\S)  +", lambda m: "&nbsp;" * len(m.group(0)), code_txt)
            flow.append(Paragraph(code_txt, ST["code"]))
            flow.append(Spacer(1, 4))
        elif kind == "hr":
            flow.append(HRFlowable(width="100%", thickness=0.6, color=colors.black))
            flow.append(Spacer(1, 4))
        elif kind == "table":
            header, rows = content
            data = [[Paragraph(inline_html(esc(c), bold=True), ST["cellh"]) for c in header]]
            for r in rows:
                data.append([Paragraph(inline_html(esc(c)), ST["cell"]) for c in r])
            ncol = max(len(header), max((len(r) for r in rows), default=0))
            tbl = Table(data, colWidths=[172 * mm / max(ncol, 1)] * ncol, repeatRows=1)
            tbl.setStyle(TableStyle([
                # 表头浅灰底 + 黑色粗体字；表格线黑色；正文行白/极浅灰交替
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.black),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#f7f7f7")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            flow.append(tbl)
            flow.append(Spacer(1, 6))

    return flow


# ---------------- 生成 PDF ----------------

def md_to_pdf(md_path: Path, pdf_path: Path):
    text = md_path.read_text(encoding="utf-8")
    blocks = parse_markdown(text)
    styles = build_styles()

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=19 * mm, rightMargin=19 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title=md_path.stem, author="中频炉预警智能体",
    )

    # 页脚：黑色，宋体（含中文页码，数字为宋体自带字形）
    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont(FONT, 9)
        canvas.setFillColor(colors.black)
        canvas.drawCentredString(A4[0] / 2, 9 * mm, f"第 {doc_.page} 页")
        canvas.restoreState()

    flow = render_blocks(blocks, styles)
    doc.build(flow, onFirstPage=_footer, onLaterPages=_footer)
    print(f"[OK] {pdf_path}")


def main():
    if not register_fonts():
        print("错误：未找到中文字体，请安装 simsun/simhei 或 Noto CJK。")
        sys.exit(1)
    global HRFlowable
    from reportlab.platypus import HRFlowable

    jobs = [
        (DOCS / "中频炉预警智能体 OpenAPI 接口文档.md",
         DOCS / "中频炉预警智能体 OpenAPI 接口文档.pdf"),
        (DOCS / "中频炉预警智能体 MCP 接口文档.md",
         DOCS / "中频炉预警智能体 MCP 接口文档.pdf"),
    ]
    for md, pdf in jobs:
        if not md.exists():
            print(f"跳过（不存在）: {md}")
            continue
        md_to_pdf(md, pdf)


if __name__ == "__main__":
    main()
