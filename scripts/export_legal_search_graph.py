"""Export the legal search agent flow without using an external renderer."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import tools as _legal_tools  # noqa: E402,F401

sys.path.insert(0, str(ROOT / "src" / "agents"))

from legal_search_agent import legal_search_agent  # noqa: E402


PNG_PATH = ROOT / "legal_search_agent_graph.png"
MERMAID_PATH = ROOT / "legal_search_agent_graph.mmd"


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    path = Path("C:/Windows/Fonts") / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str,
) -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=text_font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        ((left + right - width) / 2, (top + bottom - height) / 2 - bounds[1]),
        text,
        font=text_font,
        fill=fill,
    )


def arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    *,
    fill: str = "#64748b",
    width: int = 5,
) -> None:
    draw.line(points, fill=fill, width=width, joint="curve")
    (x1, y1), (x2, y2) = points[-2], points[-1]
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 > x1 else -1
        head = [(x2, y2), (x2 - 16 * direction, y2 - 10), (x2 - 16 * direction, y2 + 10)]
    else:
        direction = 1 if y2 > y1 else -1
        head = [(x2, y2), (x2 - 10, y2 - 16 * direction), (x2 + 10, y2 - 16 * direction)]
    draw.polygon(head, fill=fill)


def main() -> None:
    graph = legal_search_agent.get_graph()
    MERMAID_PATH.write_text(graph.draw_mermaid(), encoding="utf-8")

    image = Image.new("RGB", (1400, 900), "#f8fafc")
    draw = ImageDraw.Draw(image)

    title_font = font(42, bold=True)
    subtitle_font = font(22)
    node_font = font(28, bold=True)
    detail_font = font(19)
    label_font = font(18, bold=True)

    draw.text((70, 48), "Legal Search Agent", font=title_font, fill="#0f172a")
    draw.text(
        (72, 104),
        "LangGraph control flow · exported locally",
        font=subtitle_font,
        fill="#64748b",
    )

    start = (90, 360, 270, 440)
    model = (410, 285, 790, 515)
    tools = (960, 285, 1330, 515)
    end = (610, 700, 790, 780)

    draw.rounded_rectangle(start, radius=40, fill="#e0f2fe", outline="#0284c7", width=4)
    draw.rounded_rectangle(model, radius=28, fill="#ede9fe", outline="#7c3aed", width=4)
    draw.rounded_rectangle(tools, radius=28, fill="#dcfce7", outline="#16a34a", width=4)
    draw.rounded_rectangle(end, radius=40, fill="#fee2e2", outline="#dc2626", width=4)

    centered_text(draw, start, "START", node_font, "#075985")
    centered_text(draw, (410, 310, 790, 375), "model", node_font, "#5b21b6")
    centered_text(draw, (960, 310, 1330, 375), "tools", node_font, "#166534")
    centered_text(draw, end, "END", node_font, "#991b1b")

    centered_text(draw, (445, 385, 755, 425), "call_model", detail_font, "#475569")
    centered_text(draw, (445, 427, 755, 475), "Gemini + tool decision", detail_font, "#475569")
    centered_text(draw, (985, 380, 1305, 420), "call_tools", detail_font, "#475569")
    centered_text(draw, (985, 422, 1305, 470), "validate IDs · search · count loop", detail_font, "#475569")

    arrow(draw, [(270, 400), (410, 400)])
    arrow(draw, [(790, 355), (960, 355)])
    centered_text(draw, (805, 305, 945, 345), "tool_calls", label_font, "#166534")

    arrow(draw, [(1145, 515), (1145, 600), (600, 600), (600, 515)])
    centered_text(draw, (790, 607, 1040, 648), "results → next loop", label_font, "#475569")

    arrow(draw, [(700, 515), (700, 700)])
    centered_text(draw, (710, 590, 845, 630), "done", label_font, "#991b1b")

    draw.rounded_rectangle((70, 825, 1330, 865), radius=18, fill="#e2e8f0")
    centered_text(
        draw,
        (70, 825, 1330, 865),
        "At LEGAL_SEARCH_MAX_LOOPS, model tools are disabled and the agent must conclude.",
        detail_font,
        "#334155",
    )

    image.save(PNG_PATH, format="PNG", optimize=True)
    print(PNG_PATH)
    print(MERMAID_PATH)


if __name__ == "__main__":
    main()
