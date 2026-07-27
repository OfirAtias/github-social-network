"""Build the ordered, fully documented research-notebook series.

The generator deliberately keeps every ``src/*.py`` source character in code
cells.  Source cells are tagged with their original path and line interval so
that the final validation notebook can prove byte-for-byte coverage without
changing the authoritative Python files.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from textwrap import dedent

import nbformat as nbf
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
NOTEBOOKS = ROOT / "notebooks"
TABLES = ROOT / "outputs" / "tables"


NOTEBOOK_SPECS = [
    (
        "01_environment_and_reproducibility.ipynb",
        "סביבת עבודה, התקנות ושחזוריות",
        "מוודאת שהריפוזיטורי, הנתונים והתלויות זמינים לפני כל ניתוח.",
        "תיקיית הפרויקט, requirements.txt וקובצי data/.",
        "גרסאות חבילות, מלאי נתונים ובדיקות קדם להרצה.",
        "כל שאלות המחקר: זהו בסיס השחזוריות המשותף ל-RQ1-RQ4.",
    ),
    (
        "02_data_loading_and_preparation.ipynb",
        "טעינת הנתונים, אימותם והכנת אובייקטי הבסיס",
        "טוענת את קשתות MUSAE ואת תוויות המפתחים, מאמתת אותן ובונה גרף לא-מכוון.",
        "musae_git_edges.csv, musae_git_target.csv וקובץ המטא-דאטה.",
        "G, edges_df, labels_df ו-labels_dict, לצד תצוגות איכות ודוגמאות.",
        "מגדירה את יחידת הניתוח המשותפת ל-RQ1-RQ4 ומונעת כשלים סמויים בנתונים.",
    ),
    (
        "03_cleaning_eda_and_graph_statistics.ipynb",
        "ניקוי, EDA וסטטיסטיקה תיאורית של הרשת",
        "מחשבת איכות נתונים, התפלגות תוויות, הומופיליה, מרכזיות ותרשימי בסיס.",
        "אובייקטי הגרף והתוויות שנוצרו במחברת 02.",
        "טבלאות CSV ותרשימים תחת outputs/, כולל מרכזיות וקהילות ראשוניות.",
        "מספקת את הראיות המבניות שמניעות את RQ1 ואת השערת ההומופיליה של RQ4.",
    ),
    (
        "04_communities_and_problem_definition.ipynb",
        "קהילות Louvain והגדרה תפעולית של בעיות המחקר",
        "מאתרת קהילות, מודדת מודולריות וטוהר ומגדירה תכונות קהילה ללא דליפת תוויות.",
        "גרף מלא ותוויות; עבור תכונות מפוקחות נדרשת חלוקת train בלבד.",
        "partition, modularity, community_summary ותכונות קהילה לצמתים.",
        "מבססת את רכיב הקהילה ב-RQ1 ואת השוואת המודולריות ב-RQ4.",
    ),
    (
        "05_node_classification_algorithms.ipynb",
        "מימוש אלגוריתמי סיווג צמתים וניסוי חוזר",
        "מגדירה baselines, מאפייני גרף/שכנים/קהילה, רגרסיה לוגיסטית וכיוון סף ללא leakage.",
        "הגרף, התוויות והקהילות; seeds קבועים לחלוקות stratified.",
        "תוצאות חמש ריצות, ממוצעים, סטיות תקן, מדדי סיווג ותרשימי השוואה.",
        "עונה ישירות על RQ1 ומכינה baseline הוגן להשוואות RQ2.",
    ),
    (
        "06_embeddings_content_and_ablation.ipynb",
        "DeepWalk, Node2Vec, תוכן וניסויי אבלציה",
        "לומדת ייצוגי צמתים מהגרף ומשווה רשת, תוכן ושילוב תחת אותו פרוטוקול.",
        "גרף, תוויות ו-musae_git_features.json; אותן חלוקות train/test.",
        "embedding files, טבלאות ביצועים ותרשימי ייצוג/אבלציה/t-SNE.",
        "עונה על RQ2 ומרחיבה את RQ1 בייצוגים נלמדים בלתי-מפוקחים.",
    ),
    (
        "07_link_reconstruction_experiments.ipynb",
        "שחזור קשתות: פרוטוקול, אלגוריתמים וניסויים",
        "מסתירה קשתות תוך שמירת קישוריות ומשווה heuristics מבניים ל-embeddings.",
        "גרף מלא, תוויות ו-seeds; negatives אמיתיים ומותאמים לפי סוג קשת.",
        "ROC-AUC/AP כולל, לפי סוג תווית ובין same-field ל-cross-field.",
        "עונה על RQ3; מדובר בשחזור קשתות סטטי ולא בתחזית זמן.",
    ),
    (
        "08_null_models_results_and_research_answers.ipynb",
        "מודלי אפס, מובהקות ותשובות לשאלות המחקר",
        "משווה את הרשת לערבובי תוויות, rewiring משמר-דרגה ומודלים גנרטיביים ומרכזת מסקנות.",
        "פלטי כל הניסויים הקודמים והגרף המקורי.",
        "ערכי p אמפיריים, השוואת רשתות, התאמת power law וטבלת תשובות RQ1-RQ4.",
        "עונה ישירות על RQ4 ומסכמת באופן כמותי את RQ1-RQ3.",
    ),
    (
        "09_EXTENTIONS_and_validation.ipynb",
        "EXTENTIONS — הרחבות, הפקת תוצרים ואימות סופי",
        "מפרידה הרחבות ניסיוניות מן הליבה, מציגה builders ומריצה בדיקות שחזוריות וכיסוי.",
        "כל פלטי הפרויקט, קוד builders/validation ותיקיית extensions/.",
        "אימות סופי, כיסוי מלא של src, ומפת הרחבות מעבר לדרישות הבסיס.",
        "מבהירה מהו המימוש המרכזי ומהן הרחבות עתידיות, כנדרש בהנחיות הפרויקט.",
    ),
]


MODULE_ROLE = {
    "data_loader.py": "שכבת קלט ואימות סכמות; היא הופכת קובצי CSV לאובייקטים עקביים שעליהם נשען כל המחקר.",
    "graph_analysis.py": "מדדי תיאור בסיסיים, התפלגות מחלקות, סוגי קשתות, הומופיליה ואסורטטיביות.",
    "visualization.py": "פונקציות תרשים אחידות ששומרות PNG עם כותרות, צירים ומקרא לפי הצורך.",
    "run_core_analysis.py": "תזמור EDA, בדיקות איכות, מרכזיות, Louvain ושמירת טבלאות/תרשימים.",
    "community_analysis.py": "Louvain, מודולריות, סיכום הרכב קהילות ותכונות קהילה שמחושבות מתוויות train בלבד.",
    "modeling.py": "פרוטוקול הסיווג: חלוקות, baselines, feature engineering, Logistic Regression, metrics וכיוון סף.",
    "run_classification.py": "הרצה חוזרת בחמישה seeds, אגרגציה ותרשימי ביצועים עבור RQ1.",
    "content_features.py": "טעינת מאפייני התוכן הדלילים ומודלי content-only/hybrid עבור האבלציה של RQ2.",
    "embeddings.py": "DeepWalk ו-Node2Vec: random walks ו-skip-gram לייצוג מבני בלתי-מפוקח.",
    "run_ablation.py": "השוואה מבוקרת בין רשת בלבד, תוכן בלבד ושילובם.",
    "run_embeddings.py": "אימון embeddings, סיווג חוזר, t-SNE והשוואה לייצוגים ידניים.",
    "link_prediction.py": "פיצול קשתות חסין leakage, דגימת non-edges והיוריסטיקות שחזור.",
    "run_link_prediction.py": "ניסוי RQ3 החוזר, כולל ביצוע לפי סוג קשת ו-same/cross field.",
    "run_null_models.py": "ערבובי תוויות, rewiring משמר-דרגה, ER/BA/WS/Configuration ו-power-law.",
    "final_validation.py": "בדיקות fail-fast לתוצרים, פרוטוקולים, טווחי metrics ודוח סופי.",
    "run_all.py": "התזמור הסמכותי של כל הצנרת, מניקוי outputs ועד דוחות ואימות.",
    "build_notebook.py": "ה-builder ההיסטורי למחברת היחידה; נשמר במלואו לצורך עקיבות לאחור.",
    "build_readme.py": "יצירת README מתוך טבלאות התוצאות, כדי למנוע מספרים מיושנים בתיעוד.",
    "build_report.py": "יצירת מקור DOCX עריך לדוח האקדמי.",
    "build_report_pdf.py": "רינדור PDF שחזורי ישירות מהטבלאות והתרשימים, ללא תלות ב-Office.",
}


FUNCTION_NOTES = {
    "load_edges": "קוראת רשימת קשתות, בודקת עמודות, nulls, לולאות וכפילויות. הסריקה ליניארית במספר הרשומות O(E).",
    "load_targets": "מאחדת את סכמת התוויות ובודקת מזהים כפולים/חסרים; העלות O(V).",
    "build_graph": "בונה גרף NetworkX לא-מכוון. זמן וזיכרון הם O(V+E).",
    "detect_louvain_communities": "ממקסמת מודולריות באופן היררכי. זוהי היוריסטיקה יעילה, לרוב קרובה לליניארית בכל מעבר, אך רגישה ל-seed ובעלת resolution limit.",
    "summarize_communities": "מחשב גודל, הרכב, מחלקה דומיננטית וטוהר לכל קהילה.",
    "add_community_features": "מייצר תכונות קהילה; סטטיסטיקות תווית מסתמכות רק על train וכך מונעות leakage.",
    "generate_random_walks": "DeepWalk דוגם הליכות אחידות; העלות בקירוב O(V·num_walks·walk_length).",
    "generate_node2vec_walks": "Node2Vec מטה את המעבר מסדר שני בעזרת p ו-q ומאזן חזרה מול חקירת שכונות.",
    "train_deepwalk": "Skip-gram Word2Vec לומד וקטורים מצמדי צמתים בהליכות; workers=1 משפר דטרמיניזם.",
    "train_node2vec": "מאמן את אותו יעד skip-gram על ההליכות המוטות של Node2Vec.",
    "prepare_train_test_split": "חלוקה stratified שומרת יחס מחלקות ומקבעת seed להשוואה הוגנת.",
    "neighbor_majority_classifier": "baseline מקומי לפי שכני train בלבד; צומת ללא שכנים מתויגים חוזר למחלקת הרוב.",
    "extract_graph_neighbor_features": "מחשב degree, clustering, PageRank ומאפייני שכנים בלי לגעת בתוויות test.",
    "train_logistic_regression_graph_features": "Pipeline של קידוד/סטנדרטיזציה ורגרסיה לוגיסטית; האימון תלוי במספר הדוגמאות, התכונות ואיטרציות האופטימיזציה.",
    "evaluate_predictions": "מחשב Accuracy, Balanced Accuracy, Precision, Recall, F1, Macro/Weighted F1, ROC-AUC ומטריצת בלבול.",
    "run_repeated_threshold_tuning": "בוחר סף רק על validation פנימי ומעריך פעם אחת על test, לכן אינו מכוונן על נתוני הבחינה.",
    "sample_link_prediction_split": "מגן על עץ פורש, מסתיר רק קשתות אחרות ומתאים negatives לפי Web-Web/ML-ML/Web-ML.",
    "score_heuristics": "מחשב Common Neighbors, Jaccard, Adamic-Adar, Preferential Attachment ו-Same Community.",
    "auc_and_ap": "ROC-AUC מודד דירוג גלובלי; Average Precision רגישה לביצוע בראש הדירוג ומתאימה ל-retrieval.",
    "empirical_upper_tail_p": "משתמש בתיקון (b+1)/(B+1), ולכן ערך p לעולם אינו אפס במדגם סופי.",
    "main": "פונקציית התזמור של הקובץ. היא מוגדרת כאן אך אינה מופעלת אוטומטית; תא הרצה מפורש בהמשך מפעיל אותה ושומר פלטים.",
}


def md(source: str, **metadata):
    cell = nbf.v4.new_markdown_cell(dedent(source).strip())
    cell.metadata.update(metadata)
    return cell


def code(source: str, **metadata):
    cell = nbf.v4.new_code_cell(dedent(source).strip("\n"))
    cell.metadata.update(metadata)
    return cell


def new_notebook(spec, number: int):
    filename, title, purpose, inputs, outputs, research = spec
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "project": {
            "order": number,
            "title": title,
            "generated_by": "scripts/build_research_notebooks.py",
        },
    }
    nb.cells.append(md(f"""
    # {number:02d}. {title}

    **צוות.** נועה ירין לוי · ג'ימי חלאוי · אופיר אטיאס

    **מטרה.** {purpose}

    **מקום בתהליך.** מחברת {number} מתוך 9 בסדרת המחקר. יש להריץ את המחברות לפי הסדר המופיע ב-`notebooks/README.md`; עם זאת, קוד האתחול מאתר את שורש הפרויקט גם בהרצה עצמאית.

    **קלטים.** {inputs}

    **פלטים.** {outputs}

    **קשר לשאלות המחקר.** {research}

    **עקרונות שחזוריות.** `random_state` קבוע היכן שנדרש, כל התוצאות נשמרות תחת `outputs/`, מאפיינים מבוססי תווית נבנים מתוויות train בלבד, וכל קוד המקור הרלוונטי מוצג בתאי קוד עם נתיב ושורות מקור.
    """))
    nb.cells.append(md("""
    ## אתחול בטוח ובלתי תלוי בתיקיית ההרצה

    התא הבא מאתר את שורש הריפוזיטורי דרך `requirements.txt`, עובר אליו, מוסיף את `src` לנתיב הייבוא ומגדיר תצוגה עקבית. שינוי `__name__` מונע מבלוקי `if __name__ == \"__main__\"` שהועתקו כלשונם לפעול לפני תא ההרצה המתועד שלהם. הפלט הצפוי הוא הנתיב המוחלט של הפרויקט.
    """))
    nb.cells.append(code(f"""
    from pathlib import Path
    import os
    import sys
    import warnings

    def _find_project_root(start: Path) -> Path:
        for candidate in [start, *start.parents]:
            if (candidate / "requirements.txt").exists() and (candidate / "data").exists():
                return candidate.resolve()
        raise FileNotFoundError("לא נמצא שורש הפרויקט המכיל requirements.txt ו-data/")

    ROOT = _find_project_root(Path.cwd())
    os.chdir(ROOT)
    SRC_DIR = ROOT / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    os.environ.setdefault("PYTHONHASHSEED", "0")
    warnings.filterwarnings("ignore", category=FutureWarning)
    __name__ = "research_notebook_{number:02d}"
    print(f"Project root: {{ROOT}}")
    """))
    nb.cells.append(md("""
    **פלט ומשמעות.** הופעת הנתיב `github-social-network` מאשרת שכל הנתיבים היחסיים בהמשך יופנו לפרויקט הנכון. אם התא נכשל, אין להמשיך: הכשל מצביע על הרצה מחוץ לעץ הפרויקט או על עותק חסר.
    """))
    return nb


def _chunk_boundaries(tree: ast.Module) -> list[tuple[int, str, str | None]]:
    boundaries: list[tuple[int, str, str | None]] = [(1, "prologue", None)]
    for node in tree.body:
        kind = None
        name = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            name = node.name
            decorator_lines = [item.lineno for item in getattr(node, "decorator_list", [])]
            start = min([node.lineno, *decorator_lines])
        elif isinstance(node, ast.If) and "__name__" in ast.unparse(node.test):
            kind = "main_guard"
            name = "__main__ guard"
            start = node.lineno
        else:
            continue
        boundaries.append((start, kind, name))
    unique = {}
    for start, kind, name in boundaries:
        unique[start] = (kind, name)
    return [(start, *unique[start]) for start in sorted(unique)]


def source_chunks(path: Path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    bounds = _chunk_boundaries(tree)
    chunks = []
    for idx, (start, kind, name) in enumerate(bounds):
        end = bounds[idx + 1][0] - 1 if idx + 1 < len(bounds) else len(lines)
        chunks.append((start, end, kind, name, "".join(lines[start - 1 : end])))
    assert "".join(chunk[-1] for chunk in chunks) == source
    return source, tree, chunks


def _doc_for_node(tree: ast.Module, start: int, name: str | None) -> str:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            text = (ast.get_docstring(node) or "").strip().replace("\n", " ")
            return " ".join(text.split())
    return ""


def add_source_file(nb, relative_path: str):
    path = ROOT / relative_path
    source, tree, chunks = source_chunks(path)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    filename = path.name
    nb.cells.append(md(f"""
    ## קוד המקור: `{relative_path}`

    {MODULE_ROLE.get(filename, 'קוד תשתית של הפרויקט המועבר למחברת לצורך עצמאות, קריאות ועקיבות.')}

    לפני טעינת הקטעים נגדיר עבורם `__file__` וירטואלי התואם למיקומם המקורי. הדבר משמר את לוגיקת חישוב `PROJECT_DIR` בלי לשנות שורה בקוד המקור. SHA-256 של המקור: `{digest}`.
    """))
    nb.cells.append(md("""
    התא הבא מכין את הקשר הנתיבים עבור קובץ המקור. הוא אינו משנה את הקובץ; הפלט הצפוי הוא נתיב המקור שממנו הועתק הקוד.
    """))
    nb.cells.append(code(f"""
    __file__ = str(ROOT / {relative_path!r})
    print(f"Source context: {{__file__}}")
    """))
    nb.cells.append(md("""
    **פלט ומשמעות.** הנתיב המודפס מאפשר לבקר שהקבועים בקוד יחושבו ביחס לריפוזיטורי הנוכחי בלבד.
    """))

    for start, end, kind, name, chunk in chunks:
        label = "ייבואים, קבועים והגדרות מודול" if kind == "prologue" else f"`{name}`"
        doc = _doc_for_node(tree, start, name)
        note = FUNCTION_NOTES.get(name or "", "")
        if kind == "prologue":
            why = "הבלוק מרכז את התלויות, הקבועים והנחות ברירת המחדל של המודול; בלעדיו הפונקציות הבאות אינן עצמאיות."
        elif kind == "main_guard":
            why = "זהו שער ההרצה המקורי של קובץ CLI. במחברת הוא נשמר כלשונו, אך אינו מופעל כי `__name__` הוגדר כשם מחברת; ההרצה נעשית בתא מפורש ומתועד."
        else:
            why = note or "הבלוק מגדיר יחידת חישוב חוזרת בצנרת ומפריד בין לוגיקה לבין תזמור הניסוי."
        expected = "הגדרה בזיכרון ללא פלט חזותי; התוצאה תופיע כאשר הפונקציה תיקרא בתא הניסוי." if kind != "main_guard" else "אין פלט, משום ששער ה-CLI מושבת בכוונה בסביבת המחברת."
        doc_line = f" תיעוד המקור מסביר: {doc}" if doc else ""
        nb.cells.append(md(f"""
        ### {filename}, שורות {start}-{end}: {label}

        **מה ומדוע.** {why}{doc_line}

        **שיטה/אלגוריתם.** הקוד נשמר בדיוק לפי המקור; ההסבר האלגוריתמי המלא ניתן בפרק המתודולוגי של המחברת.

        **פלט צפוי.** {expected}
        """))
        cell = nbf.v4.new_code_cell(chunk)
        cell.metadata.update(
            {
                "source_file": relative_path.replace("\\", "/"),
                "source_start_line": start,
                "source_end_line": end,
                "source_sha256": digest,
            }
        )
        nb.cells.append(cell)
        nb.cells.append(md(f"""
        **פלט ומשמעות.** בלוק `{label}` נטען בהצלחה. היעדר פלט הוא תקין עבור הגדרה; המשמעות המחקרית שלו נבחנת בתאי ההרצה והטבלאות שבהמשך, ולא מעצם הגדרת הפונקציה.
        """))


def add_run(nb, alias: str, invocation: str, description: str, expected: str):
    nb.cells.append(md(f"""
    ## הרצת השלב

    **מה הקוד מבצע.** {description}

    **מדוע נחוץ.** הגדרת פונקציות בלבד אינה מייצרת ראיות. התא מפעיל את אותו entry point של קובץ המקור ומחדש את הטבלאות/התרשימים בפועל.

    **פלט צפוי.** {expected}
    """))
    nb.cells.append(code(f"""
    {alias} = main
    {invocation}
    """))
    nb.cells.append(md("""
    **פלט ומשמעות.** הודעות ההתקדמות שמעל מתעדות שהשלב רץ עד הסוף. הקבצים שנכתבו נטענים שוב בתאים הבאים, כך שממצא מספרי אינו נשען על טקסט מודפס בלבד.
    """))


def add_table(nb, relative_csv: str, variable: str, intro: str, interpretation: str, expression: str | None = None):
    expr = expression or variable
    nb.cells.append(md(f"""
    ### טבלה: `{relative_csv}`

    {intro} הקוד טוען מחדש את קובץ ה-CSV מהדיסק, ולכן התצוגה מאמתת גם שהפלט נשמר ולא רק חושב בזיכרון.
    """))
    nb.cells.append(code(f"""
    {variable} = pd.read_csv(ROOT / {relative_csv!r})
    display({expr})
    """))
    nb.cells.append(md(f"""
    **פלט, ממצא ומשמעות.** {interpretation}
    """))


def add_figures(nb, filenames: list[str], intro: str, interpretation: str):
    paths = ",\n        ".join(repr(f"outputs/figures/{name}") for name in filenames)
    nb.cells.append(md(f"""
    ### תרשימים שמורים

    {intro} כל התרשימים נוצרו בקוד המקור עם כותרת, שמות צירים ומקרא כאשר יש יותר מסדרה אחת.
    """))
    nb.cells.append(code(f"""
    from IPython.display import Image, display
    for relative_figure in [
        {paths}
    ]:
        print(relative_figure)
        display(Image(filename=str(ROOT / relative_figure)))
    """))
    nb.cells.append(md(f"""
    **פלט, ממצאים ופרשנות.** {interpretation}
    """))


def add_summary(nb, completed: str, findings: str, next_step: str):
    nb.cells.append(md(f"""
    ## סיכום המחברת

    **מה בוצע.** {completed}

    **ממצאים מרכזיים.** {findings}

    **חיבור למחברת הבאה.** {next_step}
    """))


def build_01():
    nb = new_notebook(NOTEBOOK_SPECS[0], 1)
    nb.cells += [
        md("""
        ## בדיקת תלויות והתקנה מותנית

        התא בודק את החבילות המופיעות ב-`requirements.txt`. רק אם חסרה תלות הוא מפעיל `pip install -r requirements.txt`, ולכן הוא מתאים גם לסביבה חדשה וגם להרצה חוזרת. הפלט המצופה הוא `All required imports are available` או פירוט ההתקנה שבוצעה.
        """),
        code("""
        import importlib.util
        import subprocess

        required_imports = {
            "pandas": "pandas", "numpy": "numpy", "networkx": "networkx",
            "matplotlib": "matplotlib", "sklearn": "scikit-learn",
            "seaborn": "seaborn", "community": "python-louvain",
            "scipy": "scipy", "gensim": "gensim", "powerlaw": "powerlaw",
            "nbformat": "nbformat", "nbconvert": "nbconvert", "docx": "python-docx",
            "pypdf": "pypdf", "bidi": "python-bidi", "reportlab": "reportlab",
        }
        missing = [package for module, package in required_imports.items() if importlib.util.find_spec(module) is None]
        if missing:
            print("Missing packages:", missing)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
        else:
            print("All required imports are available")
        """),
        md("""
        **פלט ומשמעות.** הודעת הזמינות מאשרת שאין צורך בתיקון ידני. התקנה מותנית שומרת על עצמאות המחברות בלי להתקין מחדש בכל ריצה.
        """),
        md("""
        ## גרסאות, קובצי נתונים וחתימות

        התא מתעד גרסאות חבילות, גודל קבצים ו-SHA-256. החתימות מאפשרות לזהות שינוי בנתונים שעלול להסביר תוצאה שונה. צפויות שלוש רשומות נתוני MUSAE ורשומת גרסה לכל חבילה מרכזית.
        """),
        code("""
        import hashlib
        import importlib.metadata as metadata
        import pandas as pd

        packages = ["pandas", "numpy", "networkx", "matplotlib", "scikit-learn", "seaborn", "python-louvain", "gensim", "powerlaw"]
        versions = pd.DataFrame({"package": packages, "version": [metadata.version(p) for p in packages]})
        display(versions)

        data_rows = []
        for path in sorted((ROOT / "data").glob("musae_git_*")):
            data_rows.append({
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        data_inventory = pd.DataFrame(data_rows)
        display(data_inventory)
        """),
        md("""
        **פלט ומשמעות.** טבלאות הגרסאות והחתימות הן provenance של ההרצה. צד שלישי יכול להשוות אותן לפני פירוש פערים מספריים, במיוחד ב-Word2Vec ובאלגוריתמים סטוכסטיים.
        """),
    ]
    add_summary(nb, "אותר שורש הפרויקט, נבדקו/הותקנו תלויות ותועדו גרסאות וחתימות נתונים.", "סביבת ההרצה שלמה ושלושת קובצי MUSAE נגישים ישירות, בהתאם לדרישת ה-PDF.", "מחברת 02 טוענת ומאמתת את הנתונים ובונה את אובייקט הגרף.")
    return nb


def build_02():
    nb = new_notebook(NOTEBOOK_SPECS[1], 2)
    nb.cells.append(md("""
    ## מודל הנתונים והנחות הניקוי

    כל שורה ב-edges היא קשר מעקב הדדי, ולכן הגרף לא-מכוון. `ml_target=0` מייצג Web ו-`1` מייצג Machine Learning. הניקוי כאן הוא אימות שמרני: אין השלמת תוויות, אין מחיקת צמתים שקטה ואין שינוי קשתות; קלט לא תקין גורם לכשל ברור. כך נשמרת התאמה מלאה לדאטה המקורי.
    """))
    add_source_file(nb, "src/data_loader.py")
    nb.cells.append(md("""
    ## טעינה מלאה ותצוגת בדיקות

    התא קורא את שני קובצי ה-CSV דרך ה-API שהוגדר לעיל, בונה `networkx.Graph` ומציג גדלים, קישוריות, דוגמאות ורשימת עמודות. מצופה לקבל 37,700 צמתים ו-289,003 קשתות ללא תוויות חסרות.
    """))
    nb.cells.append(code("""
    G, edges_df, labels_df, labels_dict = load_dataset(ROOT / "data")
    dataset_summary = pd.DataFrame([{
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "connected": __import__("networkx").is_connected(G),
        "label_rows": len(labels_df),
        "missing_graph_labels": len(set(G.nodes()) - set(labels_dict)),
    }])
    display(dataset_summary)
    display(edges_df.head())
    display(labels_df.head())
    """))
    nb.cells.append(md("""
    **פלט, ממצאים ומשמעות.** הגרף כולל 37,700 צמתים ו-289,003 קשתות, הוא קשיר וכל צומת מתויג. לכן אין צורך ב-imputation או בבחירת רכיב ענק, וכל ארבע שאלות המחקר פועלות על אותה אוכלוסייה בדיוק.
    """))
    add_summary(nb, "נטענו הקשתות והתוויות, אומתו הסכמות ונבנה גרף לא-מכוון.", "הנתונים שלמים, קשירים ונקיים מליקויי סכימה שמחייבים תיקון ידני.", "מחברת 03 מחשבת EDA, מרכזיות ותרשימים על הגרף המאומת.")
    return nb


def build_03():
    nb = new_notebook(NOTEBOOK_SPECS[2], 3)
    nb.cells.append(md("""
    ## מתודולוגיה: מדדי מבנה ומורכבות

    - **Degree / Degree Centrality:** מספר שכנים (או חלוקה ב-|V|-1), בחישוב O(V+E). מודד חשיפה מקומית אך מוטה כלפי hubs.
    - **PageRank:** איטרציית power על מעבר אקראי, בערך O(k(V+E)); נותן משקל לקשרים מצמתים חשובים.
    - **Betweenness:** שיעור מסלולים קצרים העוברים בצומת. חישוב מלא יקר; הפרויקט משתמש בקירוב עם 100 מקורות, בערך O(100(V+E)).
    - **Clustering coefficient:** צפיפות משולשים בשכונה; מדגיש לכידות מקומית אך נוטה להיות גבוה בצמתים קטני דרגה.
    - **HITS hub:** איטרציה ספקטרלית O(kE). בגרף לא-מכוון מתקבלת פרשנות מרכזיות ספקטרלית.
    - **Homophily:** חלק הקשתות שקצותיהן בעלי אותה תווית. זהו תיאור, לא הוכחה סיבתית; מובהקות נבדקת במחברת 08.
    """))
    add_source_file(nb, "src/graph_analysis.py")
    add_source_file(nb, "src/visualization.py")
    add_source_file(nb, "src/run_core_analysis.py")
    add_run(nb, "_run_core_analysis_main", "_run_core_analysis_main()", "מריץ את כל ניתוח הליבה: איכות, סטטיסטיקה, מרכזיות, Louvain ותרשימים.", "קובצי CSV תחת outputs/tables ושישה תרשימי PNG לפחות.")
    add_table(nb, "outputs/tables/data_quality_checks.csv", "quality_df", "בדיקות איכות פיזיות ולוגיות.", "כל תשע הבדיקות עוברות: אין לולאות/כפילויות/תוויות חסרות והרשת רכיב קשיר יחיד.")
    add_table(nb, "outputs/tables/network_basic_stats.csv", "stats_df", "סטטיסטיקות הרשת המרכזיות.", "הצפיפות כ-0.000407 והדרגה הממוצעת 15.33, אך שיעור הקשתות מאותה מחלקה 0.845 — סימן מבני חזק שייבחן מול null.")
    add_table(nb, "outputs/tables/centrality_top_overlap.csv", "overlap_df", "חפיפת top-20 בין מדדי מרכזיות.", "Degree ו-PageRank חופפים לחלוטין ב-top-20, בעוד Clustering אינו חופף עמם; המדדים תופסים תפקידים רשתיים שונים.")
    add_figures(nb, ["label_distribution.png", "degree_distribution.png", "degree_distribution_loglog.png", "edge_type_distribution.png", "sampled_network_by_label.png"], "התרשימים מציגים imbalance, זנב דרגות, סוגי קשתות ודגימת רשת צבעונית.", "התוויות אינן מאוזנות (כ-74.2% Web), התפלגות הדרגות heavy-tailed ורוב הקשתות הן תוך-תחומיות. לכן Accuracy לבדה אינה מספיקה, ובהמשך מדווחים Balanced Accuracy, F1 ו-ROC-AUC.")
    add_summary(nb, "בוצעו בדיקות איכות, EDA, מדדי מרכזיות ותרשימי מבנה.", "הרשת דלילה, heavy-tailed, בלתי מאוזנת בתוויות ובעלת הומופיליה תיאורית גבוהה.", "מחברת 04 בודקת אם מבנה זה מתארגן לקהילות Louvain שימושיות.")
    return nb


def build_04():
    nb = new_notebook(NOTEBOOK_SPECS[3], 4)
    nb.cells.append(md("""
    ## Louvain, מודולריות וטוהר

    Louvain מחפש חלוקה המגדילה את המודולריות

    \[Q=\\frac{1}{2m}\\sum_{ij}\\left(A_{ij}-\\frac{k_i k_j}{2m}\\right)\\mathbf{1}[c_i=c_j].\]

    האלגוריתם מעביר צמתים בין קהילות לפי שיפור מקומי ב-Q ואז מכווץ קהילות לגרף-על וחוזר. יתרונו הוא מהירות טובה ברשתות גדולות; מגבלותיו הן אקראיות, optimum מקומי ו-resolution limit. טוהר קהילה מודד רוב תוויתי אך מושפע מ-imbalance, ולכן אינו תחליף למדדי סיווג על test.
    """))
    add_source_file(nb, "src/community_analysis.py")
    nb.cells.append(md("""
    ## איתור קהילות והדגמת מניעת leakage

    התא טוען את הגרף, מריץ Louvain ב-seed 42 ומציג את הקהילות הגדולות. לאחר מכן הוא בונה דוגמת תכונות על חלוקת train/test; היחסים המבוססים על תוויות מחושבים מ-train בלבד. מצופה לקבל 31 קהילות ומודולריות סביב 0.453.
    """))
    nb.cells.append(code("""
    from data_loader import load_dataset
    from modeling import prepare_train_test_split
    import pandas as pd

    G, _, labels_df, labels_dict = load_dataset(ROOT / "data")
    partition, modularity = detect_louvain_communities(G, random_state=42)
    community_df = summarize_communities(G, labels_dict, partition)
    train_nodes, test_nodes, _, _ = prepare_train_test_split(labels_df, random_state=42)
    leakage_safe_features = add_community_features(
        G, labels_dict, train_nodes, test_nodes[:10], partition, community_df
    )
    display(pd.DataFrame([{"communities": len(community_df), "modularity": modularity, "weighted_purity": (community_df["size"] * community_df["purity"]).sum() / community_df["size"].sum()}]))
    display(community_df.head(10))
    display(leakage_safe_features)
    """))
    nb.cells.append(md("""
    **פלט, ממצאים ומשמעות.** נמצאו 31 קהילות, Q≈0.453 וטוהר משוקלל≈0.830. הקהילה הגדולה Web-dominant והקהילה השנייה ML-dominant; לכן community membership עשוי להוסיף אות ל-RQ1, אך הקהילות אינן חופפות לחלוטין לתוויות.
    """))
    add_figures(nb, ["community_size_distribution.png", "community_size_and_composition.png"], "תרשים אחד מציג את התפלגות גדלי הקהילות והשני את הרכבן התוויתי.", "מספר קטן של קהילות גדולות מכיל את רוב הצמתים, ולצדן זנב של קהילות קטנות. הרכב מעורב בחלק מהקהילות מסביר מדוע מודל קהילה בלבד אינו צפוי להיות מושלם.")
    add_summary(nb, "הוגדרו Louvain, מודולריות, טוהר ותכונות קהילה leakage-safe.", "הקהילות מבטאות מבנה מקצועי חלקי וחזק אך לא זהות בינארית מלאה.", "מחברת 05 משלבת אותות גרף וקהילה במודלי סיווג חוזרים.")
    return nb


def build_05():
    nb = new_notebook(NOTEBOOK_SPECS[4], 5)
    nb.cells.append(md("""
    ## מתודולוגיית הסיווג

    ארבע רמות נבדקות: Majority (רף imbalance), Neighbor Majority (אות מקומי), Logistic Regression על תכונות גרף, ו-Logistic Regression עם תכונות קהילה. Logistic Regression ממדלת `P(y=1|x)=σ(wᵀx+b)` וממזערת log-loss עם רגולריזציה; היא פרשנית ומהירה יחסית אך מניחה גבול ליניארי במרחב התכונות. כל תוצאה היא ממוצע וסטיית תקן על seeds 42, 7, 21, 84, 123. תוויות test אינן נכנסות לתכונות, ו-threshold נבחר על validation פנימי בלבד.
    """))
    add_source_file(nb, "src/modeling.py")
    add_source_file(nb, "src/run_classification.py")
    add_run(nb, "_run_classification_main", "_run_classification_main()", "מריץ ארבעה מודלים על חמש חלוקות stratified ומבצע ניסוי threshold נוסף עם validation.", "20 שורות תוצאות בסיס, 10 שורות threshold, שתי טבלאות סיכום ושני תרשימים.")
    add_table(nb, "outputs/tables/classification_summary.csv", "classification_summary", "ממוצעים וסטיות תקן לכל מודל.", "Graph+Community הוא המודל הידני החזק ביותר: ROC-AUC≈0.869 ו-F1≈0.672. Majority משיג Accuracy גבוהה לכאורה אך F1=0 למחלקת ML, ולכן אינו פתרון.", "classification_summary[[\"model\", \"accuracy_mean\", \"balanced_accuracy_mean\", \"f1_mean\", \"macro_f1_mean\", \"roc_auc_mean\"]].round(4)")
    add_table(nb, "outputs/tables/threshold_tuning_summary.csv", "threshold_summary", "השוואת threshold=0.5 לסף שנבחר ב-validation.", "הסף הנבחר הוא 0.55±0.05. F1 כמעט שאינו משתנה, בעוד precision עולה ו-recall יורד; כיוון סף משנה trade-off ולא את ROC-AUC.", "threshold_summary[[\"evaluation_type\", \"selected_threshold_mean\", \"precision_mean\", \"recall_mean\", \"f1_mean\", \"roc_auc_mean\"]].round(4)")
    add_figures(nb, ["classification_comparison.png", "threshold_tradeoff.png"], "התרשים הראשון משווה metrics בין המודלים; השני מציג את trade-off של הסף.", "תכונות קהילה משפרות באופן עקבי את המודל הידני. התרשים מדגיש מדוע יש להבחין בין יכולת דירוג (ROC-AUC) לבין החלטה בינארית תלויה-סף.")
    add_summary(nb, "מומשו ונבדקו ארבעה מודלי סיווג וכיוון סף נטול leakage.", "מבנה הגרף מאפשר סיווג משמעותי; Graph+Community מוביל ב-baseline הידני עם ROC-AUC≈0.869.", "מחברת 06 בודקת אם embeddings ותוכן מוסיפים מידע מעבר לתכונות הידניות.")
    return nb


def build_06():
    nb = new_notebook(NOTEBOOK_SPECS[5], 6)
    nb.cells.append(md("""
    ## מתודולוגיה: ייצוגים נלמדים ואבלציה

    DeepWalk יוצר הליכות אחידות ומתייחס אליהן כמשפטים; Node2Vec משתמש בפרמטרים p,q להטיית מעבר מסדר שני. בשניהם skip-gram מקרב צמתים שמופיעים בהקשרים דומים. היתרון הוא למידת מבנה לא-ליניארי בלי תוויות; המגבלות הן עלות הליכות/Word2Vec, רגישות hyperparameters והיעדר אינדוקטיביות טבעית לצמתים חדשים. ניסוי האבלציה מחזיק seeds וחלוקות קבועים ומשנה רק את משפחת המאפיינים: content-only, network-only או hybrid. כך ההפרש הוא הערכה הוגנת לערך השולי של כל מקור.
    """))
    add_source_file(nb, "src/content_features.py")
    add_source_file(nb, "src/embeddings.py")
    add_source_file(nb, "src/run_ablation.py")
    add_run(nb, "_run_ablation_main", "_run_ablation_main()", "מריץ את שלושת מודלי האבלציה על אותן חמש חלוקות.", "15 שורות תוצאות, סיכום שלושה מודלים ותרשים F1/AUC.")
    add_source_file(nb, "src/run_embeddings.py")
    add_run(nb, "_run_embeddings_main", "_run_embeddings_main()", "לומד DeepWalk ו-Node2Vec, שומר embeddings ומעריך ארבעה מודלי ייצוג.", "שני קובצי NPZ, 20 שורות תוצאה, סיכום, t-SNE ותרשים ייצוגים.")
    add_table(nb, "outputs/tables/ablation_model_results_summary.csv", "ablation_summary", "השוואת מקורות מידע.", "Content-only מגיע ל-ROC-AUC≈0.889, Network-only≈0.869 וה-Hybrid≈0.898. התוכן חזק יותר לבדו, אך הרשת מוסיפה אות משלים של כ-0.009 AUC מעל תוכן בלבד.", "ablation_summary[[\"model\", \"f1_mean\", \"macro_f1_mean\", \"roc_auc_mean\", \"roc_auc_std\"]].round(4)")
    add_table(nb, "outputs/tables/embedding_classification_summary.csv", "embedding_summary", "ביצועי DeepWalk/Node2Vec לבד ובשילוב ידני.", "DeepWalk ו-Node2Vec לבדם סביב ROC-AUC 0.891; שילוב Graph+Community מעלה אותם לכ-0.900. דמיון במספרים אינו הוכחת שקילות ייצוגים.", "embedding_summary[[\"model\", \"f1_mean\", \"macro_f1_mean\", \"roc_auc_mean\", \"roc_auc_std\"]].round(4)")
    add_figures(nb, ["ablation_f1_auc_comparison.png", "representation_comparison.png", "embedding_tsne.png"], "התרשימים משווים מקורות מידע, משפחות ייצוג ומרחב t-SNE מדגמי.", "Hybrid מוביל באבלציה, והוספת תכונות ידניות ל-embeddings משפרת מעט. t-SNE מציג חפיפה לצד אזורים תוויתיים; הוא כלי המחשה דו-ממדי ולא מדד סיווג.")
    add_summary(nb, "אומנו embeddings ונערכה אבלציה מבוקרת של רשת/תוכן/שילוב.", "התוכן חזק, אך המבנה מוסיף מידע משלים; ה-Hybrid הטוב ביותר עם ROC-AUC≈0.898, וייצוגי graph+handcrafted סביב 0.900.", "מחברת 07 משתמשת במבנה ובהטמעות למשימת שחזור קשתות נפרדת.")
    return nb


def build_07():
    nb = new_notebook(NOTEBOOK_SPECS[6], 7)
    nb.cells.append(md("""
    ## פרוטוקול שחזור והיוריסטיקות

    נבחר עץ פורש מוגן O(V+E), ומסתירים 10% רק מן הקשתות שאינן בעץ; לכן גרף האימון נשאר קשיר וללא מבודדים. לכל positive נדגם non-edge אמיתי באותו סוג תוויתי, כדי למנוע confounding בהרכב Web/ML. Common Neighbors הוא |Γ(u)∩Γ(v)|; Jaccard מחלק באיחוד; Adamic-Adar משקלל שכן משותף ב-1/log d; Preferential Attachment הוא d(u)d(v); Same Community בינארי; embedding dot מודד דמיון וקטורי. עלות scoring המקומי תלויה בחיתוך שכנויות, בעוד embedding דורש קודם אימון הליכות.
    """))
    add_source_file(nb, "src/link_prediction.py")
    add_source_file(nb, "src/run_link_prediction.py")
    add_run(nb, "_run_link_prediction_main", "_run_link_prediction_main()", "מריץ חמישה splits, מחשב שבע שיטות ושומר תוצאות כולל stratification.", "35 תוצאות כוללות, diagnostics לכל split ותוצאות לפי edge type ו-same/cross field.")
    add_table(nb, "outputs/tables/edge_split_diagnostics.csv", "split_diagnostics", "בדיקת תקינות הפיצול.", "בכל seed מוסתרות 28,900 קשתות, train נשאר קשיר, אין isolated nodes ומספר ה-positives וה-negatives זהה בכל strata.")
    add_table(nb, "outputs/tables/edge_reconstruction_summary.csv", "link_summary", "סיכום ROC-AUC ו-AP על חמש ריצות.", "Preferential Attachment מוביל עם ROC-AUC≈0.924 ו-AP≈0.931; Node2Vec ו-DeepWalk אחריו. המשמעות היא שדרגות הקצוות הן אות חזק מאוד בשחזור הסטטי.", "link_summary.sort_values(\"roc_auc_mean\", ascending=False).round(4)")
    add_table(nb, "outputs/tables/edge_reconstruction_same_vs_cross_gaps.csv", "link_gaps", "פער same-field פחות cross-field לכל שיטה.", "הפער אינו אחיד: ל-Preferential Attachment יתרון קטן ל-same-field (≈0.008 AUC), אך embeddings מעט טובים יותר ב-cross-field. לכן אין להסיק שכל קשת תוך-תחומית קלה יותר בכל שיטה.")
    add_figures(nb, ["edge_reconstruction_auc.png", "edge_reconstruction_same_vs_cross.png"], "תרשים כולל ותרשים מפוצל same/cross עם error bars.", "הדירוג הכולל יציב על פני seeds. הפיצול מראה אינטראקציה בין סוג הקשת לשיטת הניקוד ומונע מסקנה גורפת המבוססת רק על ממוצע.")
    add_summary(nb, "בוצע ניסוי שחזור leakage-safe עם שבע שיטות וחמש חלוקות.", "Preferential Attachment מוביל; same-field קל מעט יותר עבורו, אך לא עבור כל השיטות.", "מחברת 08 בודקת אם ההומופיליה והמודולריות חריגות מול מודלי אפס ומרכזת תשובות RQ1-RQ4.")
    return nb


def build_08():
    nb = new_notebook(NOTEBOOK_SPECS[7], 8)
    nb.cells.append(md("""
    ## מתודולוגיית מודלי האפס

    ערבוב תוויות שומר את הגרף ואת שכיחות המחלקות ובודק אם שיוך התוויות לקודקודים מסביר הומופיליה. Rewiring משמר-דרגה שומר degree sequence ותוויות ובודק מבנה מעבר לקורלציית דרגה-תווית. ER משמר V,E בממוצע; BA מייצר preferential attachment; Watts-Strogatz משלב clustering ומרחקים קצרים; Configuration משמר את רצף הדרגות. ערך p הוא `(b+1)/(B+1)` ומונע p=0. עם 20 rewires הרזולוציה המינימלית היא 1/21≈0.0476, מגבלה שיש לציין.
    """))
    add_source_file(nb, "src/run_null_models.py")
    add_run(nb, "_run_null_models_main", "_run_null_models_main()", "מריץ 1,000 ערבובי תוויות, 20 rewires, מודלים גנרטיביים והתאמת power law.", "טבלאות null, השוואת רשתות, power-law ותרשים התפלגות null.")
    add_table(nb, "outputs/tables/null_model_homophily.csv", "homophily_null", "בדיקות הומופיליה מול שני nulls.", "הנצפה 0.845 לעומת 0.617 בערבוב תוויות (p≈0.001) ו-0.751 ב-rewiring (p≈0.0476). גם לאחר שליטה בדרגות נשאר עודף הומופיליה.")
    add_table(nb, "outputs/tables/null_model_modularity.csv", "modularity_null", "מודולריות Louvain מול גרפים משמרי-דרגה.", "Q הנצפה≈0.453 לעומת ≈0.189 ב-null, עם p≈0.0476; מבנה הקהילות חזק מכפי שמסביר רצף הדרגות לבדו.")
    add_table(nb, "outputs/tables/network_vs_random_models.csv", "random_models", "השוואת מדדי הרשת למודלים גנרטיביים.", "אף מודל יחיד אינו משחזר הכול: ER מפספס hubs/clustering, BA יוצר זנב אך clustering נמוך, WS משחזר clustering גבוה מדי ומרחק גדול, ו-Configuration קרוב בדרגות אך לא בכל הסגירות המקומית.")
    add_table(nb, "outputs/tables/powerlaw_fit.csv", "powerlaw_fit", "התאמת זנב הדרגות.", "α≈2.54 מעל xmin=55. Power law עדיף משמעותית על exponential, אך אינו מובחן בבירור מ-lognormal (p≈0.783); לכן אין לטעון power law ייחודי.")
    add_figures(nb, ["homophily_null_distribution.png"], "הקו הנצפה מושווה להתפלגות של 1,000 ערבובי תוויות.", "הערך הנצפה נמצא הרחק בזנב העליון. זה תומך במובהקות סטטיסטית של הקשר בין תווית למבנה, לא בסיבתיות.")
    nb.cells.append(md("""
    ## תשובות מרוכזות לשאלות המחקר

    התא הבא טוען את טבלאות הסיכום מכל השלבים ומפיק טבלת תשובה אחת. הוא אינו מקליד מספרים ידנית, ולכן נשאר מסונכרן עם ההרצה האחרונה.
    """))
    nb.cells.append(code("""
    classification = pd.read_csv(ROOT / "outputs/tables/classification_summary.csv")
    ablation = pd.read_csv(ROOT / "outputs/tables/ablation_model_results_summary.csv")
    embeddings = pd.read_csv(ROOT / "outputs/tables/embedding_classification_summary.csv")
    links = pd.read_csv(ROOT / "outputs/tables/edge_reconstruction_summary.csv")
    gaps = pd.read_csv(ROOT / "outputs/tables/edge_reconstruction_same_vs_cross_gaps.csv")
    null_h = pd.read_csv(ROOT / "outputs/tables/null_model_homophily.csv")
    null_q = pd.read_csv(ROOT / "outputs/tables/null_model_modularity.csv")

    rq_answers = pd.DataFrame([
        {"RQ": "RQ1", "evidence": f"Best handcrafted AUC={classification.roc_auc_mean.max():.3f}; best embedding+handcrafted AUC={embeddings.roc_auc_mean.max():.3f}", "answer": "מבנה הרשת מסווג תחום היטב אך לא באופן מושלם."},
        {"RQ": "RQ2", "evidence": f"Network={ablation.loc[ablation.model.str.contains('Network Only'), 'roc_auc_mean'].iloc[0]:.3f}; Content={ablation.loc[ablation.model.str.contains('Content Features Only'), 'roc_auc_mean'].iloc[0]:.3f}; Hybrid={ablation.loc[ablation.model.str.contains('Hybrid'), 'roc_auc_mean'].iloc[0]:.3f}", "answer": "התוכן חזק יותר לבדו, והרשת מוסיפה מידע משלים."},
        {"RQ": "RQ3", "evidence": f"Best={links.loc[links.roc_auc_mean.idxmax(), 'heuristic']} AUC={links.roc_auc_mean.max():.3f}; same-cross gap={gaps.loc[gaps.heuristic.eq('Preferential Attachment'), 'roc_auc_same_minus_cross'].iloc[0]:.3f}", "answer": "שחזור סטטי אפשרי; יתרון same-field תלוי שיטה ואינו אוניברסלי."},
        {"RQ": "RQ4", "evidence": f"Homophily p={null_h.loc[null_h.null_model.eq('label_shuffle'), 'empirical_p'].iloc[0]:.6f}; rewired homophily p={null_h.loc[null_h.null_model.eq('degree_preserving_rewire'), 'empirical_p'].iloc[0]:.6f}; modularity p={null_q.empirical_p.iloc[0]:.6f}", "answer": "הומופיליה ומודולריות חריגות ביחס למודלי האפס שנבדקו."},
    ])
    display(rq_answers)
    """))
    nb.cells.append(md("""
    **פלט, ממצאים ומשמעות.** RQ1 מקבל תשובה חיובית מתונה; RQ2 מראה השלמה בין רשת לתוכן; RQ3 חיובי לשחזור סטטי עם הסתייגות לפי סוג שיטה; RQ4 מראה חריגות מובהקת ברזולוציית הסימולציות. המסקנות אסוציאטיביות ואינן מזהות מנגנון סיבתי.
    """))
    add_summary(nb, "הושוו null models ומודלים גנרטיביים ונבנתה טבלת תשובות דינמית לכל RQ.", "המבנה המקצועי ברשת מובהק מעבר למחלקות/דרגות, אך אף מודל גנרטיבי פשוט אינו מסביר את כל התכונות.", "מחברת 09 מפרידה הרחבות, מציגה את קוד הפקת התוצרים ומבצעת validation וכיסוי סופיים.")
    return nb


def build_09():
    nb = new_notebook(NOTEBOOK_SPECS[8], 9)
    nb.cells.append(md("""
    ## מה נחשב EXTENTIONS בפרויקט הזה

    הליבה המחקרית היא כל מה שרץ דרך `src/run_all.py` ומדווח ב-README ובדוח. תיקיית `extensions/` מכילה כיוונים מעבר לדרישות הבסיס: Label Propagation, GCN/GraphSAGE וגילוי תתי-תחומים בתוכן באמצעות SVD+KMeans. הם חשובים משום שהם בודקים message passing, semi-supervision ורזולוציית תוויות עשירה יותר; הם אינם מעורבבים בטבלאות הליבה, מפני שחלקם דורשים PyTorch/תקציב ניסוי נוסף וחלקם מחוץ להיקף האלגוריתמים שנבחר להגשה. ההפרדה מונעת הצגת תוצאה לא מאומתת כטענת פרויקט.

    שיפורים עתידיים מוצעים: nested cross-validation להיפרפרמטרים, יותר rewires לשיפור רזולוציית p, בדיקה טמפורלית אמיתית ל-link prediction, calibration למדדי הסתברות ובדיקת fairness מעבר לתווית הבינארית.

    **הערת הגשה לפי מסמך ההנחיות.** הדוח הסופי חייב לכלול חלוקת תרומות אישית בין נועה ירין לוי, ג'ימי חלאוי ואופיר אטיאס. חלוקה זו אינה ניתנת להסקה אמינה מקוד Git בלבד ולכן אינה מומצאת במחברות; יש לוודא שסעיף התרומות העובדתי מופיע בגרסת הדוח המוגשת.
    """))
    nb.cells.append(md("""
    ## מלאי הרחבות קיים

    התא קורא את README של ההרחבות ומציג את קובצי Python בלי להפעיל ניסויים שאינם חלק מהצנרת המאומתת. הפלט הצפוי הוא ארבעה קבצים ותיאור ההפרדה מהליבה.
    """))
    nb.cells.append(code("""
    import pandas as pd

    extension_files = sorted((ROOT / "extensions").glob("*.py"))
    extension_inventory = pd.DataFrame([
        {"file": p.name, "lines": len(p.read_text(encoding="utf-8").splitlines()), "bytes": p.stat().st_size}
        for p in extension_files
    ])
    display(extension_inventory)
    print((ROOT / "extensions" / "README.md").read_text(encoding="utf-8"))
    """))
    nb.cells.append(md("""
    **פלט ומשמעות.** ההרחבות שמורות ונגישות, אך אינן משנות את תשובות RQ1-RQ4. זוהי הבחנה מפורשת בין contribution מרכזי לבין exploration עתידי.
    """))

    add_source_file(nb, "src/final_validation.py")
    nb.cells.append(code("_run_all_validations = run_all_validations"))
    add_source_file(nb, "src/run_all.py")
    add_source_file(nb, "src/build_notebook.py")
    add_source_file(nb, "src/build_readme.py")
    add_source_file(nb, "src/build_report.py")
    add_source_file(nb, "src/build_report_pdf.py")

    nb.cells.append(md("""
    ## אימות פלטי הפרויקט

    התא מפעיל את מערכת ה-validation המקורית: קבצים נדרשים, CSV תקינים, גרף ופיצולי קשתות, p-values, קבוצות מודלים, מחברת legacy ודוח עד שמונה עמודים. מצופה לקבל `ALL FINAL PROJECT VALIDATIONS PASSED`.
    """))
    nb.cells.append(code("_run_all_validations(ROOT, verify_notebook_execution=True)"))
    nb.cells.append(md("""
    **פלט ומשמעות.** הודעת המעבר מתקבלת רק אם כל תנאי האיכות המקוריים מתקיימים; כשל מצביע על תוצר חסר או סתירה במספרים ולא מוסתר.
    """))

    nb.cells.append(md("""
    ## הוכחת כיסוי byte-for-byte של כל `src/*.py`

    לכל תא מקור נשמרים נתיב ושורות. התא מאחד את התאים לפי סדרם, משווה לטקסט המקורי ומוודא שכל 20 הקבצים מכוסים בדיוק פעם אחת בסדרה. זהו test אוטומטי לדרישת "ללא השמטות".
    """))
    nb.cells.append(code("""
    import nbformat
    from collections import defaultdict

    reconstructed = defaultdict(list)
    series_paths = sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb"))
    for notebook_path in series_paths:
        notebook = nbformat.read(notebook_path, as_version=4)
        for cell in notebook.cells:
            source_file = cell.metadata.get("source_file")
            if source_file:
                reconstructed[source_file].append((cell.metadata["source_start_line"], cell.source))

    coverage_rows = []
    for source_path in sorted((ROOT / "src").glob("*.py")):
        relative = source_path.relative_to(ROOT).as_posix()
        combined = "".join(text for _, text in sorted(reconstructed.get(relative, [])))
        original = source_path.read_text(encoding="utf-8")
        coverage_rows.append({
            "source_file": relative,
            "source_lines": len(original.splitlines()),
            "covered_chars": len(combined),
            "source_chars": len(original),
            "exact_match": combined == original,
        })
    coverage_df = pd.DataFrame(coverage_rows)
    display(coverage_df)
    assert len(coverage_df) == 20
    assert coverage_df["exact_match"].all(), coverage_df.loc[~coverage_df["exact_match"]]
    print("ALL 20 SRC FILES ARE COVERED EXACTLY")
    """))
    nb.cells.append(md("""
    **פלט ומשמעות.** כל שורה עם `exact_match=True` מוכיחה שהקוד הועבר בשלמותו בלי לשנות את קובצי המקור. הבדיקה משלימה את ה-validation הניסויי בבדיקת תיעוד וארגון.
    """))
    add_summary(nb, "הופרדו הרחבות, הוצג כל קוד ה-build/validation, הורץ validation ונבדק כיסוי byte-for-byte.", "ליבת הפרויקט משוחזרת ועקבית; ההרחבות מסומנות בבירור ואינן מנפחות את טענות הליבה.", "הסדרה מסתיימת כאן. להרצה חוזרת יש להשתמש בפקודות שב-notebooks/README.md לפי הסדר 01-09.")
    return nb


def write_index():
    rows = []
    for idx, spec in enumerate(NOTEBOOK_SPECS, 1):
        filename, title, purpose, _inputs, _outputs, research = spec
        rows.append(f"| {idx:02d} | [`{filename}`]({filename}) | {title} | {purpose} | {research} |")
    readme = f"""# סדרת מחברות המחקר

סדרה זו היא הנתיב המתועד והמורץ של הפרויקט של נועה ירין לוי, ג'ימי חלאוי ואופיר אטיאס. היא נבנתה מחדש מתוך **כל 20 קובצי `src/*.py`** ובהתאם ל-`Graph_Algo_project_guidelines_2026 (3).pdf`. קובצי המקור אינם משתנים; קוד המקור מופיע בתאי קוד עם נתיב וטווח שורות, ומחברת 09 מוכיחה התאמה byte-for-byte.

## סדר ההרצה

| סדר | מחברת | מטרה | תפקיד | קשר למחקר |
|---:|---|---|---|---|
{chr(10).join(rows)}

המחברת הישנה `GitHub_Developer_Field_Prediction_Network_Only.ipynb` נשמרת לצורך עקיבות, אך הסדרה הממוספרת היא התיעוד המחקרי המפורט.

## הרצה מלאה ב-Windows PowerShell

```powershell
Set-Location (git rev-parse --show-toplevel)
$env:PYTHONHASHSEED="0"
if (-not (Test-Path .\\.venv\\Scripts\\python.exe)) {{
    py -3.11 -m venv .venv
}}
.\\.venv\\Scripts\\python.exe -m pip install -r .\\requirements.txt
New-Item -ItemType Directory -Force .\\tmp\\jupyter-runtime | Out-Null
.\\.venv\\Scripts\\python.exe -m ipykernel install --prefix .\\tmp\\jupyter-runtime --name github-social-network --display-name "GitHub Social Network (.venv)"
$env:JUPYTER_PATH = (Resolve-Path .\\tmp\\jupyter-runtime\\share\\jupyter).Path
$notebooks = Get-ChildItem .\\notebooks -Filter "[0-9][0-9]_*.ipynb" | Sort-Object Name
foreach ($notebook in $notebooks) {{
    .\\.venv\\Scripts\\python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=github-social-network --ExecutePreprocessor.timeout=-1 $notebook.FullName
}}
```

כל מחברת מאתרת את שורש הפרויקט גם אם Jupyter פתח אותה מתוך `notebooks/`. מחברת 01 מתקינה תלויות חסרות באופן מותנה. להרצה מלאה מומלץ להשתמש ב-Python 3.11 ובסביבה הווירטואלית של הפרויקט.

## מפת תלות

`01 סביבה` → `02 נתונים` → `03 EDA` → `04 קהילות` → `05 סיווג` → `06 embeddings/ablation` → `07 שחזור קשתות` → `08 nulls ותשובות` → `09 EXTENTIONS ואימות`.

הפלטים נשמרים ב-`outputs/tables`, `outputs/figures` ו-`outputs/embeddings`. בכל מחברת מופיעים הסברים לפני קוד, פרשנות אחרי פלטים וסיכום המקשר לשלב הבא.
"""
    (NOTEBOOKS / "README.md").write_text(readme, encoding="utf-8")


def main():
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    builders = [build_01, build_02, build_03, build_04, build_05, build_06, build_07, build_08, build_09]
    for spec, builder in zip(NOTEBOOK_SPECS, builders):
        filename = spec[0]
        notebook = builder()
        nbf.write(notebook, NOTEBOOKS / filename)
        print(f"Wrote {filename}: {len(notebook.cells)} cells")
    write_index()
    print("Wrote notebooks/README.md")
    # Apply the permanent English documentation and legacy-content merge after
    # generation so rebuilt notebooks do not regress to the old Hebrew text.
    from englishize_notebooks import englishize_all

    englishize_all(ROOT)


if __name__ == "__main__":
    main()
