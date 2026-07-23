# -*- coding: utf-8 -*-
"""Build the final eight-page PDF directly from the fresh result tables.

The editable Word source is produced by ``build_report.py``.  This renderer
uses the same tables and narrative, but avoids depending on an installed
office-suite PDF driver so that the final artifact is reproducible on Windows,
Linux and macOS.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from bidi.algorithm import get_display
from PIL import Image
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from final_validation import collect_key_results


REPORT_PATH = PROJECT_DIR / "report" / "final_report.pdf"
TABLES_DIR = PROJECT_DIR / "outputs" / "tables"
FIGURES_DIR = PROJECT_DIR / "outputs" / "figures"

PAGE_W, PAGE_H = LETTER
MARGIN = 38
CONTENT_W = PAGE_W - 2 * MARGIN
INK = HexColor("#172B4D")
ACCENT = HexColor("#315C8D")
ORANGE = HexColor("#D57837")
MUTED = HexColor("#5F6B7A")
LIGHT = HexColor("#E8EEF5")
PALE = HexColor("#F4F6F9")


def _font_paths() -> tuple[Path, Path]:
    candidates = [
        (
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf"),
        ),
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ),
        (
            Path("/Library/Fonts/Arial.ttf"),
            Path("/Library/Fonts/Arial Bold.ttf"),
        ),
    ]
    for regular, bold in candidates:
        if regular.is_file() and bold.is_file():
            return regular, bold
    raise FileNotFoundError("A Hebrew-capable Arial or DejaVu font is required.")


REGULAR_FONT, BOLD_FONT = _font_paths()
pdfmetrics.registerFont(TTFont("Report", str(REGULAR_FONT)))
pdfmetrics.registerFont(TTFont("Report-Bold", str(BOLD_FONT)))


def _visual(text: object) -> str:
    return get_display(str(text), base_dir="R")


def _wrap(text: str, width: float, font: str, size: float) -> list[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if pdfmetrics.stringWidth(_visual(candidate), font, size) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _shorten(text: object, width: float, font: str, size: float) -> str:
    logical = str(text)
    if pdfmetrics.stringWidth(_visual(logical), font, size) <= width:
        return logical
    suffix = "…"
    while logical and pdfmetrics.stringWidth(
        _visual(logical + suffix), font, size
    ) > width:
        logical = logical[:-1]
    return logical + suffix


class ReportCanvas:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.canvas = Canvas(str(path), pagesize=LETTER)
        self.canvas.setTitle(
            "תחום ההתמחות של מפתחי GitHub מתוך רשת מעקב הדדית"
        )
        self.canvas.setAuthor("Noa Yarin Levy; Jimi Halawi; Ofir Atias")
        self.y = PAGE_H - 58
        self.page_number = 0

    def begin_page(self, section: str) -> None:
        self.page_number += 1
        self.y = PAGE_H - 55
        c = self.canvas
        c.setStrokeColor(LIGHT)
        c.setLineWidth(0.8)
        c.line(MARGIN, PAGE_H - 40, PAGE_W - MARGIN, PAGE_H - 40)
        c.setFillColor(MUTED)
        c.setFont("Report", 7.4)
        c.drawRightString(
            PAGE_W - MARGIN,
            PAGE_H - 32,
            _visual("תחום התמחות ברשת GitHub | פרויקט גמר"),
        )
        c.setFillColor(ACCENT)
        c.setFont("Report-Bold", 7.4)
        c.drawString(MARGIN, PAGE_H - 32, _visual(section))

    def finish_page(self) -> None:
        c = self.canvas
        c.setStrokeColor(LIGHT)
        c.line(MARGIN, 30, PAGE_W - MARGIN, 30)
        c.setFillColor(MUTED)
        c.setFont("Report", 7.5)
        c.drawCentredString(
            PAGE_W / 2,
            18,
            _visual(f"עמוד {self.page_number} מתוך 8"),
        )
        c.showPage()

    def heading(self, text: str, *, size: float = 14, gap: float = 5) -> None:
        self.canvas.setFillColor(ACCENT)
        self.canvas.setFont("Report-Bold", size)
        self.canvas.drawRightString(PAGE_W - MARGIN, self.y, _visual(text))
        self.y -= size + gap

    def paragraph(
        self,
        text: str,
        *,
        size: float = 9.0,
        leading: float = 11.5,
        gap: float = 5,
        bold: bool = False,
        color=INK,
        indent: float = 0,
    ) -> None:
        font = "Report-Bold" if bold else "Report"
        self.canvas.setFillColor(color)
        self.canvas.setFont(font, size)
        for line in _wrap(text, CONTENT_W - indent, font, size):
            self.canvas.drawRightString(
                PAGE_W - MARGIN - indent, self.y, _visual(line)
            )
            self.y -= leading
        self.y -= gap

    def bullet(self, text: str) -> None:
        self.paragraph(f"• {text}", size=8.4, leading=10.2, gap=1.5, indent=8)

    def table(
        self,
        headers: list[str],
        rows: list[list[object]],
        widths: list[float],
        *,
        font_size: float = 7.6,
        row_height: float = 21,
        gap: float = 6,
    ) -> None:
        if abs(sum(widths) - CONTENT_W) > 0.1:
            raise ValueError("Table widths must sum to the content width.")
        c = self.canvas
        x0 = MARGIN
        y_top = self.y
        all_rows = [headers, *rows]
        for row_index, values in enumerate(all_rows):
            y = y_top - row_height * (row_index + 1)
            c.setFillColor(LIGHT if row_index == 0 else (PALE if row_index % 2 == 0 else white))
            c.rect(x0, y, CONTENT_W, row_height, fill=1, stroke=0)
            x = x0
            for width, value in zip(widths, values):
                c.setStrokeColor(white)
                c.rect(x, y, width, row_height, fill=0, stroke=1)
                font = "Report-Bold" if row_index == 0 else "Report"
                c.setFont(font, font_size)
                c.setFillColor(INK)
                shown = _shorten(value, width - 8, font, font_size)
                c.drawRightString(
                    x + width - 4,
                    y + (row_height - font_size) / 2 + 1.5,
                    _visual(shown),
                )
                x += width
        self.y = y_top - row_height * len(all_rows) - gap

    def figure(
        self,
        filename: str,
        caption: str,
        *,
        box_height: float,
        gap: float = 6,
    ) -> None:
        path = FIGURES_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            image_w, image_h = image.size
        scale = min(CONTENT_W / image_w, box_height / image_h)
        draw_w = image_w * scale
        draw_h = image_h * scale
        x = MARGIN + (CONTENT_W - draw_w) / 2
        y = self.y - draw_h
        self.canvas.drawImage(
            str(path),
            x,
            y,
            width=draw_w,
            height=draw_h,
            preserveAspectRatio=True,
            mask="auto",
        )
        self.y = y - 10
        self.canvas.setFillColor(MUTED)
        self.canvas.setFont("Report", 7.2)
        self.canvas.drawCentredString(PAGE_W / 2, self.y, _visual(caption))
        self.y -= 8 + gap

    def save(self) -> None:
        if self.page_number != 8:
            raise AssertionError(f"Expected eight pages, got {self.page_number}.")
        self.canvas.save()


def _r(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def main() -> None:
    key = collect_key_results(PROJECT_DIR)
    basic = pd.read_csv(TABLES_DIR / "network_basic_stats.csv").iloc[0]
    centrality = pd.read_csv(TABLES_DIR / "centrality_top_developers.csv")
    overlap = pd.read_csv(TABLES_DIR / "centrality_top_overlap.csv")
    community = pd.read_csv(TABLES_DIR / "community_metrics.csv").iloc[0]
    classification = pd.read_csv(TABLES_DIR / "classification_summary.csv")
    thresholds = pd.read_csv(TABLES_DIR / "threshold_tuning_summary.csv").set_index(
        "evaluation_type"
    )
    ablation = pd.read_csv(TABLES_DIR / "ablation_model_results_summary.csv")
    embeddings = pd.read_csv(TABLES_DIR / "embedding_classification_summary.csv")
    reconstruction = pd.read_csv(TABLES_DIR / "edge_reconstruction_summary.csv")
    recon_groups = pd.read_csv(
        TABLES_DIR / "edge_reconstruction_same_vs_cross_summary.csv"
    )
    null_h = pd.read_csv(TABLES_DIR / "null_model_homophily.csv").set_index(
        "null_model"
    )
    null_m = pd.read_csv(TABLES_DIR / "null_model_modularity.csv").iloc[0]
    networks = pd.read_csv(TABLES_DIR / "network_vs_random_models.csv")
    powerlaw = pd.read_csv(TABLES_DIR / "powerlaw_fit.csv").iloc[0]

    report = ReportCanvas(REPORT_PATH)

    # Page 1 — title, abstract, research questions and data.
    report.begin_page("תקציר ושאלות מחקר")
    report.canvas.setFillColor(INK)
    report.canvas.setFont("Report-Bold", 18)
    report.canvas.drawCentredString(
        PAGE_W / 2,
        report.y,
        _visual("תחום ההתמחות של מפתחי GitHub מתוך רשת מעקב הדדית"),
    )
    report.y -= 23
    report.canvas.setFillColor(ACCENT)
    report.canvas.setFont("Report-Bold", 10)
    report.canvas.drawCentredString(
        PAGE_W / 2, report.y, _visual("פרויקט גמר — אלגוריתמים ברשתות")
    )
    report.y -= 16
    report.canvas.setFillColor(MUTED)
    report.canvas.setFont("Report", 8.2)
    report.canvas.drawCentredString(
        PAGE_W / 2,
        report.y,
        _visual(
            "נועה ירין לוי (208101196) · ג'ימי חלאוי (208301176) · אופיר אטיאס (211446935)"
        ),
    )
    report.y -= 12
    report.canvas.drawCentredString(
        PAGE_W / 2, report.y, _visual('מרצה: ד"ר אבנר פריאל')
    )
    report.y -= 20
    report.heading("תקציר", size=13)
    report.paragraph(
        f"העבודה מנתחת רשת לא־מכוונת של {key['network_nodes']:,} מפתחי GitHub "
        f"ו־{key['network_edges']:,} קשרי מעקב הדדיים. נבדקו סיווג תחום מקצועי "
        "ממבנה הרשת, הערך המשלים של תוכן, שחזור מדויק של קשתות שהוסתרו ומובהקות "
        "מבנית מול מודלי אפס. כל ניסויי החיזוי חזרו בחמש חלוקות קבועות, ללא שימוש "
        "בתוויות הבדיקה לבניית מאפיינים או לכיול. "
        f"שיעור הקשתות התוך־תחומיות הוא {100 * key['homophily_rate']:.1f}%; "
        f"המסווג הידני הטוב הגיע ל־ROC-AUC {key['best_classification_roc_auc']:.3f}; "
        f"DeepWalk ו־Node2Vec הגיעו ל־{key['deepwalk_roc_auc']:.3f} "
        f"ו־{key['node2vec_roc_auc']:.3f}. בשחזור קשתות הובילה "
        f"{key['best_reconstruction_heuristic']} עם ROC-AUC "
        f"{key['best_reconstruction_roc_auc']:.3f}. אלו קשרים חיזויים בתנאי "
        "הניסוי ואינם ראיה לסיבתיות."
    )
    report.heading("שאלות המחקר", size=12)
    report.paragraph(
        "RQ1: באיזו מידה שכנות, מרכזיות, קהילות וייצוגי הליכות מאפשרים לסווג את "
        "תחום המפתח כאשר תוויות הבדיקה מוסתרות?",
        size=8.6,
        leading=10.6,
        gap=2,
    )
    report.paragraph(
        "RQ2: מהו הערך השולי של מבנה הרשת מול מאפייני תוכן, ומה מוסיף השילוב?",
        size=8.6,
        leading=10.6,
        gap=2,
    )
    report.paragraph(
        "RQ3: האם ניתן לשחזר קשתות שהוסתרו אקראית תוך שמירת קישוריות, והאם "
        "same-field קל יותר לשחזור מ־cross-field?",
        size=8.6,
        leading=10.6,
        gap=2,
    )
    report.paragraph(
        "RQ4: האם הומופיליה ומודולריות חריגות מול מודלי אפס, ואילו מאפיינים "
        "משחזרים ER, BA, Small-World ורשת משמרת־דרגה?",
        size=8.6,
        leading=10.6,
        gap=5,
    )
    report.table(
        ["צמתים", "קשתות", "דרגה ממוצעת", "צפיפות", "רכיבי קישוריות"],
        [[
            f"{int(basic['num_nodes']):,}",
            f"{int(basic['num_edges']):,}",
            _r(basic["average_degree"]),
            _r(basic["density"], 6),
            str(int(basic["num_connected_components"])),
        ]],
        [92, 100, 115, 110, CONTENT_W - 417],
        font_size=7.7,
    )
    report.finish_page()

    # Page 2 — methodology and exact logical model inventory.
    report.begin_page("1 · מתודולוגיה ושיטות")
    report.heading("מסגרת מתודולוגית ושיטות")
    report.paragraph(
        "בסיווג הצמתים מבנה הגרף גלוי (transductive), אך תוויות test מוסתרות. "
        "מאפייני שכנים וקהילה מחושבים מתוויות train בלבד. בכל seed בוצעה חלוקה "
        "מרובדת 70/30, וסף ההחלטה כויל רק על validation פנימי מתוך train."
    )
    report.paragraph(
        "ניתוח הרשת כולל Degree, Clustering, PageRank, Betweenness בקירוב, HITS "
        "Hub יחיד, Louvain ומודולריות. התפלגות הדרגות נבחנת בזהירות מול חלופות. "
        "לשחזור קשתות מוגן עץ פורש; non-edges מותאמים לפי Web-Web, ML-ML ו־Web-ML."
    )
    model_rows = [
        ["בסיס", "Majority", "שכיחות train"],
        ["בסיס", "Neighbor Majority", "תוויות שכני train"],
        ["ידני", "LogReg Graph", "מרכזיות ומבנה"],
        ["ידני", "LogReg Graph + Community", "מבנה וקהילה"],
        ["תוכן", "LogReg Content", "מאפייני MUSAE"],
        ["היברידי", "LogReg Hybrid", "רשת ותוכן"],
        ["הטמעה", "LogReg DeepWalk", "הליכות אחידות"],
        ["הטמעה", "DeepWalk + ידני", "הטמעה ומבנה"],
        ["הטמעה", "LogReg Node2Vec", "הליכות מוטות"],
        ["הטמעה", "Node2Vec + ידני", "הטמעה ומבנה"],
    ]
    report.table(
        ["משפחה", "מודל", "קלט מרכזי"],
        model_rows,
        [90, 225, CONTENT_W - 315],
        font_size=7.5,
        row_height=19,
    )
    report.heading("שחזור קשתות ומודלי אפס", size=11)
    report.paragraph(
        "השיטות הן Common Neighbors, Jaccard, Adamic-Adar, Preferential "
        "Attachment, Same Community, DeepWalk ו־Node2Vec. ההטמעות והקהילות "
        "נלמדות מחדש על גרף האימון בלבד. ערכי p אמפיריים מחושבים כ־(b+1)/(B+1), "
        "ולכן אינם אפס. ההשוואות כוללות 1,000 ערבובי תוויות, 20 חיווטים "
        "משמרי־דרגה ומודלי ER, BA ו־Watts-Strogatz."
    )
    report.finish_page()

    # Page 3 — structure and communities.
    report.begin_page("2 · מבנה, מרכזיות וקהילות")
    report.heading("מבנה הרשת, מרכזיות וקהילות")
    report.paragraph(
        f"הרשת קשירה וצפופה מעט בלבד ({basic['density']:.6f}). מדדי Degree "
        "ו־PageRank מדגישים מוקדי קשרים, בעוד Betweenness מדגיש תיווך בין אזורים; "
        "לכן רשימות המובילים אינן זהות."
    )
    top_rows: list[list[object]] = []
    for metric in ("Degree Centrality", "PageRank", "Betweenness Centrality"):
        row = (
            centrality[centrality["metric"].str.startswith(metric)]
            .sort_values("rank")
            .iloc[0]
        )
        top_rows.append(
            [
                metric.replace(" Centrality", ""),
                row["developer_name"],
                int(row["node_id"]),
                int(row["degree"]),
                f"{float(row['score']):.6f}",
            ]
        )
    report.table(
        ["מדד", "מפתח מוביל", "Node ID", "דרגה", "ציון"],
        top_rows,
        [100, 190, 76, 60, CONTENT_W - 426],
        font_size=7.2,
        row_height=20,
    )
    degree_pr = overlap[
        overlap.apply(
            lambda row: {
                row["metric_a"].replace(" Centrality", ""),
                row["metric_b"].replace(" Centrality", ""),
            }
            == {"Degree", "PageRank"},
            axis=1,
        )
    ]["jaccard_overlap"].iloc[0]
    degree_bet = overlap[
        overlap.apply(
            lambda row: "Degree" in row["metric_a"]
            + row["metric_b"]
            and "Betweenness" in row["metric_a"] + row["metric_b"],
            axis=1,
        )
    ]["jaccard_overlap"].iloc[0]
    report.paragraph(
        f"חפיפת Jaccard בין 20 המובילים ב־Degree וב־PageRank היא {degree_pr:.3f}, "
        f"לעומת {degree_bet:.3f} בין Degree ל־Betweenness. הפער ממחיש שמתווכים "
        "אינם בהכרח בעלי מספר הקשרים הגדול ביותר.",
        size=8.5,
    )
    report.figure(
        "community_size_and_composition.png",
        "איור 1. קהילות Louvain הגדולות: גודל והרכב תחומי.",
        box_height=260,
    )
    report.paragraph(
        f"Louvain מצא {int(community['number_of_communities'])} קהילות עם "
        f"מודולריות {community['modularity']:.3f}; הקהילה הגדולה מכילה "
        f"{int(community['largest_community_size']):,} צמתים והטוהר המשוקלל "
        f"הוא {community['size_weighted_mean_purity']:.3f}. קהילות מעורבות "
        "מונעות פירוש דטרמיניסטי של השתייכות קהילתית.",
        size=8.5,
    )
    report.finish_page()

    # Page 4 — node classification.
    report.begin_page("3 · RQ1 — סיווג צמתים")
    report.heading("RQ1 — סיווג צמתים")
    class_rows = [
        [
            str(row["model"]).replace("Logistic Regression - ", "LogReg "),
            _r(row["macro_f1_mean"]),
            _r(row["macro_f1_std"]),
            _r(row["roc_auc_mean"]),
            _r(row["roc_auc_std"]),
        ]
        for _, row in classification.sort_values("roc_auc_mean").iterrows()
    ]
    report.table(
        ["מודל", "Macro F1", "SD", "ROC-AUC", "SD"],
        class_rows,
        [230, 82, 58, 82, CONTENT_W - 452],
        font_size=7.4,
        row_height=20,
    )
    report.figure(
        "classification_comparison.png",
        "איור 2. סיווג צמתים בחמש חלוקות; פסי השגיאה הם סטיית תקן.",
        box_height=315,
    )
    default = thresholds.loc["default_threshold_0.5"]
    tuned = thresholds.loc["tuned_threshold_validation_f1"]
    report.paragraph(
        f"הוספת קהילה העלתה את ROC-AUC ל־{key['best_classification_roc_auc']:.3f}. "
        f"הסף שנבחר על validation בלבד היה {tuned['selected_threshold_mean']:.3f} "
        f"בממוצע. Precision השתנה מ־{default['precision_mean']:.3f} ל־"
        f"{tuned['precision_mean']:.3f}, ו־Recall מ־{default['recall_mean']:.3f} "
        f"ל־{tuned['recall_mean']:.3f}; זהו trade-off תפעולי ולא שיפור בכל מדד."
    )
    report.paragraph(
        "מסקנת RQ1: מבנה הרשת מספק אות חיזויי מעל קווי הבסיס, וקהילה מוסיפה "
        "מידע מעבר לשכנות ולמרכזיות. המסקנה מוגבלת לפרוטוקול transductive "
        "ולתווית הבינארית שנבדקה.",
        bold=True,
        size=8.6,
    )
    report.finish_page()

    # Page 5 — content ablation and learned representations.
    report.begin_page("4 · RQ2 — רשת, תוכן והטמעות")
    report.heading("RQ2 — רשת, תוכן וייצוגים נלמדים")
    rows: list[list[object]] = []
    for _, row in ablation.iterrows():
        rows.append(
            [
                str(row["model"]).replace("Logistic Regression - ", "LogReg "),
                _r(row["macro_f1_mean"]),
                _r(row["roc_auc_mean"]),
                _r(row["roc_auc_std"]),
            ]
        )
    for _, row in embeddings.iterrows():
        rows.append(
            [
                str(row["model"]).replace("Logistic Regression - ", "LogReg "),
                _r(row["macro_f1_mean"]),
                _r(row["roc_auc_mean"]),
                _r(row["roc_auc_std"]),
            ]
        )
    report.table(
        ["ייצוג / מקור מידע", "Macro F1", "ROC-AUC", "SD AUC"],
        rows,
        [295, 84, 84, CONTENT_W - 463],
        font_size=7.1,
        row_height=18,
    )
    report.figure(
        "representation_comparison.png",
        "איור 3. פיצ'רים ידניים, תוכן, DeepWalk, Node2Vec ושילובים.",
        box_height=275,
    )
    hybrid = ablation[ablation["model"].str.contains("Hybrid", regex=False)].iloc[0]
    content = ablation[
        ablation["model"].str.contains("Content Features Only", regex=False)
    ].iloc[0]
    network = ablation[
        ablation["model"].str.contains("Network Only", regex=False)
    ].iloc[0]
    report.paragraph(
        f"תוכן לבדו הגיע ל־AUC {content['roc_auc_mean']:.3f}, מול "
        f"{network['roc_auc_mean']:.3f} ברשת ידנית ו־{hybrid['roc_auc_mean']:.3f} "
        f"בשילוב. DeepWalk ו־Node2Vec לבדם הגיעו ל־{key['deepwalk_roc_auc']:.3f} "
        f"ו־{key['node2vec_roc_auc']:.3f}. הדמיון אינו מוכיח שקילות בין מקורות מידע."
    )
    report.paragraph(
        "מסקנת RQ2: התוכן חזק מן הייצוג הידני של הרשת, אך השילוב מוסיף ביצועים; "
        "הטמעות הליכות מרכזות מידע מבני שימושי ומצמצמות חלק מן הפער.",
        bold=True,
        size=8.6,
    )
    report.finish_page()

    # Page 6 — connected edge reconstruction.
    report.begin_page("5 · RQ3 — שחזור קשתות")
    report.heading("RQ3 — שחזור קשתות קשיר")
    report.paragraph(
        "בכל seed הוגנו קשתות עץ פורש, והוסרו 10% רק מן הקשתות האחרות. בכל חמש "
        "החלוקות גרף האימון נשאר רכיב קשיר יחיד וללא צמתים מבודדים. לכל positive "
        "הותאם non-edge מאותו סוג קצוות; Louvain וההטמעות חושבו על train בלבד."
    )
    recon_rows = [
        [
            row["heuristic"],
            _r(row["roc_auc_mean"]),
            _r(row["roc_auc_std"]),
            _r(row["average_precision_mean"]),
            _r(row["average_precision_std"]),
        ]
        for _, row in reconstruction.sort_values(
            "roc_auc_mean", ascending=False
        ).iterrows()
    ]
    report.table(
        ["שיטה", "ROC-AUC", "SD", "Average Precision", "SD"],
        recon_rows,
        [210, 77, 55, 130, CONTENT_W - 472],
        font_size=7.0,
        row_height=18,
    )
    report.figure(
        "edge_reconstruction_same_vs_cross.png",
        "איור 4. השוואה ישירה same-field מול cross-field בדגימה מאוזנת.",
        box_height=255,
    )
    best_group = recon_groups[
        recon_groups["heuristic"] == key["best_reconstruction_heuristic"]
    ].set_index("edge_group")
    same_auc = best_group.loc["same-field", "roc_auc_mean"]
    cross_auc = best_group.loc["cross-field", "roc_auc_mean"]
    same_ap = best_group.loc["same-field", "average_precision_mean"]
    cross_ap = best_group.loc["cross-field", "average_precision_mean"]
    report.paragraph(
        f"{key['best_reconstruction_heuristic']} הוביל עם AUC "
        f"{key['best_reconstruction_roc_auc']:.3f}. עבורו same-field קיבל AUC "
        f"{same_auc:.3f} ו־AP {same_ap:.3f}, לעומת AUC {cross_auc:.3f} ו־AP "
        f"{cross_ap:.3f} ב־cross-field. הפערים בכל השיטות קטנים ותלויי שיטה; "
        "הממצא מתאר שחזור קשתות שהוסתרו, לא התפתחות קשרים בזמן."
    )
    report.finish_page()

    # Page 7 — null and generative models.
    report.begin_page("6 · RQ4 — מודלי אפס ורשת")
    report.heading("RQ4 — מודלי אפס ומודלי רשת")
    report.table(
        ["מודל אפס", "נצפה", "ממוצע אפס", "SD", "p אמפירי"],
        [
            [
                "ערבוב תוויות",
                _r(null_h.loc["label_shuffle", "observed"]),
                _r(null_h.loc["label_shuffle", "null_mean"]),
                _r(null_h.loc["label_shuffle", "null_std"]),
                _r(null_h.loc["label_shuffle", "empirical_p"], 6),
            ],
            [
                "חיווט משמר־דרגה",
                _r(null_h.loc["degree_preserving_rewire", "observed"]),
                _r(null_h.loc["degree_preserving_rewire", "null_mean"]),
                _r(null_h.loc["degree_preserving_rewire", "null_std"]),
                _r(null_h.loc["degree_preserving_rewire", "empirical_p"], 6),
            ],
        ],
        [150, 76, 105, 66, CONTENT_W - 397],
        font_size=7.2,
        row_height=19,
    )
    report.figure(
        "homophily_null_distribution.png",
        "איור 5. הומופיליה נצפית מול ערבוב תוויות וחיווטים משמרי־דרגה.",
        box_height=235,
        gap=3,
    )
    network_rows = [
        [
            row["network"],
            f"{int(row['edges']):,}",
            _r(row["avg_clustering"], 4),
            _r(row["avg_shortest_path_sampled"]),
            int(row["max_degree"]),
        ]
        for _, row in networks.iterrows()
    ]
    report.table(
        ["רשת", "קשתות", "Clustering", "מרחק ממוצע", "דרגת שיא"],
        network_rows,
        [205, 82, 83, 92, CONTENT_W - 462],
        font_size=6.7,
        row_height=17,
    )
    report.paragraph(
        f"המודולריות הנצפית היא {null_m['observed_modularity']:.3f}, מול "
        f"{null_m['rewired_null_mean']:.3f} בחיווטים משמרי־דרגה "
        f"(p={null_m['empirical_p']:.6f}). התאמת Power Law נתנה "
        f"α={powerlaw['alpha']:.2f}; ההשוואה ללוג־נורמלית אינה מצדיקה הכרזה "
        "גורפת על scale-free. ER אינו משחזר התקבצות או דרגת שיא; BA משחזר חלק "
        "מן הזנב; Small-World משחזר התקבצות ומרחקים קצרים.",
        size=8.2,
        leading=10.2,
    )
    report.finish_page()

    # Page 8 — synthesis, limitations and references.
    report.begin_page("7 · דיון, מגבלות ומסקנות")
    report.heading("דיון מסכם ומגבלות")
    report.paragraph(
        "התמונה המצטברת עקבית: קיים קשר תיאורי חזק בין מבנה הרשת לתחום; קהילות "
        "והטמעות מוסיפות כוח חיזויי; תוכן מספק אות משלים. מודלי האפס מראים שרצף "
        "הדרגות מסביר חלק מן ההומופיליה אך לא את כולה. הצלחת Preferential "
        "Attachment תואמת את חשיבות הדרגות, אך אינה מזהה מנגנון סיבתי."
    )
    report.heading("מגבלות", size=11)
    for limitation in [
        "הנתונים הם תמונת מצב מיוני 2019 ואינם מייצגים בהכרח את GitHub כיום.",
        "תווית היעד בינארית ונגזרה ממידע פרופיל; ייתכנו רעש וחסר.",
        "הסיווג transductive ואינו מדמה צומת חדש ללא קשרים ידועים.",
        "Betweenness מחושב בקירוב וההטמעות נבדקו בתקציב קבוע ללא חיפוש רחב.",
        "שחזור קשתות שהוסתרו אינו תחזית זמן ואינו מזהה סיבתיות או כיווניות.",
        "מודלי האפס והמודלים הגנרטיביים מבודדים מאפיינים ואינם תהליך מלא.",
    ]:
        report.bullet(limitation)
    report.heading("מסקנות", size=11)
    report.paragraph(
        "RQ1: מבנה הרשת מאפשר סיווג מעל קווי הבסיס. RQ2: יש חפיפה חלקית ומידע "
        "משלים בין רשת לתוכן. RQ3: שחזור קשתות אפשרי תוך שמירת קישוריות, והפער "
        "same/cross תלוי בשיטה. RQ4: ההומופיליה והמודולריות חריגות ביחס למודלי "
        "האפס שנבדקו. הממצאים תומכים בערך האנליטי של רשתות מקצועיות, תוך שמירה "
        "על ניסוח תיאורי וזהיר."
    )
    report.heading("מקורות", size=11)
    references = [
        "1. Rozemberczki, Allen & Sarkar (2019), MUSAE.",
        "2. Blondel et al. (2008), Fast unfolding of communities.",
        "3. Perozzi, Al-Rfou & Skiena (2014), DeepWalk.",
        "4. Grover & Leskovec (2016), node2vec.",
        "5. Liben-Nowell & Kleinberg (2003), Link Prediction.",
        "6. Clauset, Shalizi & Newman (2009), Power-law distributions.",
        "7. Page et al. (1999), PageRank.",
        "8. מצגות ותרגולי הקורס: מרכזיות, קהילות, מודלי רשת, שחזור והטמעות.",
    ]
    for reference in references:
        report.paragraph(reference, size=7.4, leading=8.6, gap=0.2)
    report.save()
    print(f"Built final PDF: {REPORT_PATH}")


if __name__ == "__main__":
    main()
