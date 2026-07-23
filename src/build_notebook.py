# -*- coding: utf-8 -*-
"""Build the final, ordered notebook from freshly generated CSV results."""

from pathlib import Path
import sys

import nbformat as nbf
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from final_validation import collect_key_results


NOTEBOOK_PATH = (
    PROJECT_DIR
    / "notebooks"
    / "GitHub_Developer_Field_Prediction_Network_Only.ipynb"
)
TABLES_DIR = PROJECT_DIR / "outputs" / "tables"


def _f(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _table_rows(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    rows = [
        "| "
        + " | ".join(str(row[column]) for column in columns)
        + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, rule, *rows])


def _result_context() -> dict:
    key = collect_key_results(PROJECT_DIR)
    centrality = pd.read_csv(
        TABLES_DIR / "centrality_top_developers.csv"
    )
    overlap = pd.read_csv(TABLES_DIR / "centrality_top_overlap.csv")
    community_metrics = pd.read_csv(
        TABLES_DIR / "community_metrics.csv"
    ).iloc[0]
    communities = pd.read_csv(TABLES_DIR / "community_summary.csv")
    classification = pd.read_csv(
        TABLES_DIR / "classification_summary.csv"
    )
    threshold = pd.read_csv(TABLES_DIR / "threshold_tuning_summary.csv")
    ablation = pd.read_csv(
        TABLES_DIR / "ablation_model_results_summary.csv"
    )
    embeddings = pd.read_csv(
        TABLES_DIR / "embedding_classification_summary.csv"
    )
    reconstruction = pd.read_csv(
        TABLES_DIR / "edge_reconstruction_summary.csv"
    )
    reconstruction_groups = pd.read_csv(
        TABLES_DIR / "edge_reconstruction_same_vs_cross_summary.csv"
    )
    reconstruction_gaps = pd.read_csv(
        TABLES_DIR / "edge_reconstruction_same_vs_cross_gaps.csv"
    )
    null_homophily = pd.read_csv(
        TABLES_DIR / "null_model_homophily.csv"
    )
    null_modularity = pd.read_csv(
        TABLES_DIR / "null_model_modularity.csv"
    ).iloc[0]
    networks = pd.read_csv(
        TABLES_DIR / "network_vs_random_models.csv"
    )
    powerlaw = pd.read_csv(TABLES_DIR / "powerlaw_fit.csv").iloc[0]
    return locals()


def _top_name(centrality: pd.DataFrame, metric_prefix: str) -> str:
    row = centrality[
        centrality["metric"].str.startswith(metric_prefix)
    ].sort_values("rank").iloc[0]
    return (
        f"{row['developer_name']} (צומת {int(row['node_id'])}, "
        f"דרגה {int(row['degree']):,})"
    )


def _overlap_value(
    overlap: pd.DataFrame, metric_a: str, metric_b: str
) -> float:
    mask = (
        overlap["metric_a"].str.startswith(metric_a)
        & overlap["metric_b"].str.startswith(metric_b)
    ) | (
        overlap["metric_a"].str.startswith(metric_b)
        & overlap["metric_b"].str.startswith(metric_a)
    )
    return float(overlap.loc[mask, "jaccard_overlap"].iloc[0])


def main() -> None:
    context = _result_context()
    key = context["key"]
    centrality = context["centrality"]
    overlap = context["overlap"]
    community_metrics = context["community_metrics"]
    communities = context["communities"]
    classification = context["classification"]
    threshold = context["threshold"]
    ablation = context["ablation"]
    embeddings = context["embeddings"]
    reconstruction = context["reconstruction"]
    reconstruction_groups = context["reconstruction_groups"]
    reconstruction_gaps = context["reconstruction_gaps"]
    null_homophily = context["null_homophily"]
    null_modularity = context["null_modularity"]
    networks = context["networks"]
    powerlaw = context["powerlaw"]

    degree_top = _top_name(centrality, "Degree Centrality")
    pagerank_top = _top_name(centrality, "PageRank")
    betweenness_top = _top_name(centrality, "Betweenness")
    degree_pr_overlap = _overlap_value(
        overlap, "Degree Centrality", "PageRank"
    )
    degree_bet_overlap = _overlap_value(
        overlap, "Degree Centrality", "Betweenness"
    )

    top_communities = communities.head(3).copy()
    top_communities["summary"] = top_communities.apply(
        lambda row: (
            f"קהילה {int(row['community_id'])}: {int(row['size']):,} צמתים, "
            f"{100 * row['ml_ratio']:.1f}% ML"
        ),
        axis=1,
    )
    community_lines = "\n".join(
        f"- {text}" for text in top_communities["summary"]
    )

    classification_display = classification[
        ["model", "macro_f1_mean", "macro_f1_std", "roc_auc_mean", "roc_auc_std"]
    ].copy()
    classification_display.columns = [
        "מודל",
        "Macro F1 ממוצע",
        "Macro F1 סטיית תקן",
        "ROC-AUC ממוצע",
        "ROC-AUC סטיית תקן",
    ]
    for column in classification_display.columns[1:]:
        classification_display[column] = classification_display[column].map(
            lambda value: _f(value)
        )

    ablation_display = ablation[
        ["model", "macro_f1_mean", "roc_auc_mean", "roc_auc_std"]
    ].copy()
    ablation_display.columns = [
        "מקור מידע",
        "Macro F1",
        "ROC-AUC",
        "סטיית תקן AUC",
    ]
    for column in ablation_display.columns[1:]:
        ablation_display[column] = ablation_display[column].map(
            lambda value: _f(value)
        )

    embedding_display = embeddings[
        ["model", "macro_f1_mean", "roc_auc_mean", "roc_auc_std"]
    ].copy()
    embedding_display.columns = [
        "מודל",
        "Macro F1",
        "ROC-AUC",
        "סטיית תקן AUC",
    ]
    for column in embedding_display.columns[1:]:
        embedding_display[column] = embedding_display[column].map(
            lambda value: _f(value)
        )

    best_heuristic = key["best_reconstruction_heuristic"]
    best_groups = reconstruction_groups[
        reconstruction_groups["heuristic"] == best_heuristic
    ].set_index("edge_group")
    best_gap = reconstruction_gaps[
        reconstruction_gaps["heuristic"] == best_heuristic
    ].iloc[0]
    same_auc = float(best_groups.loc["same-field", "roc_auc_mean"])
    cross_auc = float(best_groups.loc["cross-field", "roc_auc_mean"])
    same_ap = float(
        best_groups.loc["same-field", "average_precision_mean"]
    )
    cross_ap = float(
        best_groups.loc["cross-field", "average_precision_mean"]
    )
    relationship = "גבוהה" if same_auc > cross_auc else "נמוכה"

    label_null = null_homophily.set_index("null_model")
    network_rows = networks.set_index("network")
    observed_network = network_rows.loc["GitHub observed"]
    er_network = network_rows.loc["Erdos-Renyi"]
    ba_name = next(name for name in network_rows.index if name.startswith("Barabasi"))
    ws_name = next(name for name in network_rows.index if name.startswith("Watts"))
    config_network = network_rows.loc[
        "Degree-preserving configuration null"
    ]

    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3"}
    nb.metadata["final_project_expected_results"] = key

    cells = []
    cells.append(
        nbf.v4.new_markdown_cell(
            f"""# תחום התמחות של מפתחי GitHub מתוך רשת מעקב הדדית

**פרויקט גמר — אלגוריתמים ברשתות**

נועה ירין לוי · ג'ימי חלאוי · אופיר אטיאס<br>
מרצה: ד"ר אבנר פריאל

העבודה בוחנת רשת לא־מכוונת של {key['network_nodes']:,} מפתחים ו־{key['network_edges']:,} קשרי מעקב הדדיים. הגרסה הסופית מבוססת על שיטות הקורס בלבד; מאפייני תוכן משמשים נקודת ייחוס מבוקרת ולא תחליף לניתוח הרשת."""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 1. מוטיבציה ושאלות מחקר

קשרי מעקב ב־GitHub עשויים לשאת מידע תיאורי על תחומי עניין מקצועיים. מטרת הניתוח אינה לטעון לסיבתיות, אלא למדוד אילו מאפיינים מבניים קשורים לתוויות Web ו־Machine Learning וכמה מידע שימושי הם מכילים.

1. **RQ1 — סיווג צמתים:** באיזו מידה שכנות, מרכזיות, קהילות וייצוגים נלמדים מאפשרים לסווג את תחום המפתח ללא שימוש בתוויות הבדיקה?
2. **RQ2 — רשת מול תוכן:** מהו הערך התיאורי השולי של מאפייני הרשת מול מאפייני התוכן, ומה מוסיף שילובם?
3. **RQ3 — שחזור קשתות:** האם ניתן לשחזר קשתות מעקב שהוסתרו אקראית באמצעות היוריסטיקות מבניות וייצוגים נלמדים, והאם קשתות תוך־תחומיות קלות יותר לשחזור מקשתות בין־תחומיות?
4. **RQ4 — מודלי אפס:** האם ההומופיליה והמודולריות הנצפות חריגות ביחס לערבוב תוויות ולגרפים משמרי־דרגות, ואילו מאפייני רשת משוחזרים על ידי מודלי ER, BA, Small-World ו־Configuration?

RQ3 עוסק בשחזור קשתות שהוסתרו מתוך תמונת מצב סטטית — לא בתחזית זמן ולא במנגנון היווצרות קשרים."""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 2. סביבת הרצה ושחזוריות

כל האקראיות נשלטת באמצעות seeds קבועים: 42, 7, 21, 84 ו־123. אימון Word2Vec מתבצע על worker יחיד. להרצה ביט־ל־ביט מומלץ להגדיר `PYTHONHASHSEED=0` לפני פתיחת ה־kernel."""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """from pathlib import Path
import os, random, sys
import numpy as np
import pandas as pd
from IPython.display import Image, display

PROJECT_DIR = Path.cwd().resolve()
if not (PROJECT_DIR / "src").exists():
    PROJECT_DIR = PROJECT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
TABLES_DIR = PROJECT_DIR / "outputs" / "tables"
FIGURES_DIR = PROJECT_DIR / "outputs" / "figures"
SEEDS = (42, 7, 21, 84, 123)
random.seed(42)
np.random.seed(42)
print(f"Project: {PROJECT_DIR}")
print(f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED', 'not set (set to 0 before launch)')}")"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 3. דאטה, הכנה ובדיקות איכות

הקלט הוא MUSAE GitHub Social Network (יוני 2019): רשימת קשתות, תווית בינארית לכל צומת ומאפייני תוכן דלילים. לפני כל ניתוח נבדקים כפילויות, לולאות עצמיות, תוויות חסרות, התאמת קבוצת הצמתים וקשירות."""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """from run_core_analysis import main as run_core_analysis
run_core_analysis()
display(pd.read_csv(TABLES_DIR / "data_quality_checks.csv"))
display(pd.read_csv(TABLES_DIR / "network_basic_stats.csv").round(5))
display(Image(filename=str(FIGURES_DIR / "label_distribution.png")))"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            f"""### תיאור הרשת

הרשת כוללת {key['network_nodes']:,} צמתים ו־{key['network_edges']:,} קשתות, ללא לולאות או כפילויות ובמרכיב קשיר יחיד. שיעור הקשתות המחברות צמתים מאותו תחום הוא {_f(100 * key['homophily_rate'], 1)}%. זהו קשר תיאורי משמעותי, אך לבדו אינו מפריד בין השפעת שכיחות התוויות, התפלגות הדרגות ומבנה קהילתי."""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """display(Image(filename=str(FIGURES_DIR / "degree_distribution.png")))
display(Image(filename=str(FIGURES_DIR / "edge_type_distribution.png")))
display(Image(filename=str(FIGURES_DIR / "sampled_network_by_label.png")))"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            f"""## 4. מרכזיות: פופולריות, חשיבות רקורסיבית ותיווך

- **Degree Centrality** מדגישה קשרים ישירים; המוביל הוא {degree_top}.
- **PageRank** משקללת גם את חשיבות השכנים; המוביל הוא {pagerank_top}.
- **Betweenness Centrality** מדגישה צמתים על מסלולים רבים; בקירוב המבוסס על 100 מקורות, המוביל הוא {betweenness_top}.
- חפיפת Jaccard בין 20 המובילים ב־Degree וב־PageRank היא {_f(degree_pr_overlap)}, לעומת {_f(degree_bet_overlap)} בין Degree וב־Betweenness. לכן המדדים אינם חליפיים: הראשון והשני מדגישים מוקדי קשרים, בעוד Betweenness עשוי לחשוף מתווכים בין אזורים.

ב־HITS נשמר ציון Hub יחיד בלבד, משום שבגרף לא־מכוון אין הצדקה לשני פיצ'רים חופפים."""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """centrality_top = pd.read_csv(TABLES_DIR / "centrality_top_developers.csv")
display(centrality_top.groupby("metric", group_keys=False).head(5).round(6))
display(pd.read_csv(TABLES_DIR / "centrality_top_overlap.csv").round(3))"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            f"""## 5. קהילות Louvain

Louvain מצא {int(community_metrics['number_of_communities'])} קהילות עם מודולריות {_f(community_metrics['modularity'])}. הקהילה הגדולה מכילה {int(community_metrics['largest_community_size']):,} מפתחים, והטוהר הממוצע המשוקלל בגודל הוא {_f(community_metrics['size_weighted_mean_purity'])}.

{community_lines}

הבדלי היחסים בין הקהילות תומכים בכך שהמבנה הקהילתי קשור לתחום, אך קהילות מעורבות מונעות זיהוי מושלם ואין לפרש את החלוקה כזהות מקצועית קשיחה."""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """display(pd.read_csv(TABLES_DIR / "community_metrics.csv").round(4))
display(pd.read_csv(TABLES_DIR / "community_summary.csv").head(12).round(3))
display(Image(filename=str(FIGURES_DIR / "community_size_and_composition.png")))
display(Image(filename=str(FIGURES_DIR / "community_size_distribution.png")))"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 6. RQ1 — סיווג צמתים ללא דליפת מידע

בכל אחת מחמש החלוקות נשמר יחס 70/30 מרובד. תוויות test אינן משמשות באימון, בחישוב מאפייני שכנים/קהילה או בכיול הסף. מודל הקהילה משתמש רק בספירות תוויות מתוך train."""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """from run_classification import main as run_classification
run_classification()
display(pd.read_csv(TABLES_DIR / "classification_summary.csv").round(4))
display(Image(filename=str(FIGURES_DIR / "classification_comparison.png")))
display(pd.read_csv(TABLES_DIR / "threshold_tuning_summary.csv").round(4))
display(Image(filename=str(FIGURES_DIR / "threshold_tradeoff.png")))"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            f"""### תוצאות RQ1

{_table_rows(classification_display, list(classification_display.columns))}

המודל החזק בקבוצה זו הוא **{key['best_classification_model']}** עם ROC-AUC ממוצע {_f(key['best_classification_roc_auc'])}. הפער מול שכנות בלבד מצביע על ערך מצטבר של מאפייני מרכזיות וקהילה. בחירת סף נעשית על validation פנימי בלבד; היא משנה את האיזון בין Precision ל־Recall ואינה משנה את ROC-AUC."""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 7. RQ2 — Ablation: רשת, תוכן ושילוב

שלושת מקורות המידע נבחנים באותו מסווג ובאותן חלוקות. כך ההבדלים מיוחסים לייצוג הקלט ולא לפרוטוקול הערכה שונה."""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """from run_ablation import main as run_ablation
run_ablation()
display(pd.read_csv(TABLES_DIR / "ablation_model_results_summary.csv").round(4))
display(Image(filename=str(FIGURES_DIR / "ablation_f1_auc_comparison.png")))"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            f"""### תוצאות RQ2

{_table_rows(ablation_display, list(ablation_display.columns))}

מאפייני התוכן הם נקודת ייחוס חזקה, והרשת לבדה חלשה מהם; השילוב משפר את הממוצע ומצביע על מידע משלים. זו עדות חיזויית בתנאי החלוקות שנבדקו, לא הוכחה לכך שמקור מידע אחד גורם לתווית."""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 8. ייצוגים נלמדים: DeepWalk ו־Node2Vec

שני הייצוגים נלמדים ללא תוויות וללא תוכן, עם אותו תקציב של 3 הילוכים לצומת, אורך 40, 64 ממדים ו־5 epochs. ב־Node2Vec נקבעו \(p=1, q=2\), הטיה מתונה לשכונות מקומיות. כל ייצוג נבדק לבדו ובשילוב הפיצ'רים הידניים."""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """from run_embeddings import main as run_embeddings
run_embeddings()
display(pd.read_csv(TABLES_DIR / "embedding_classification_summary.csv").round(4))
display(Image(filename=str(FIGURES_DIR / "representation_comparison.png")))
display(Image(filename=str(FIGURES_DIR / "embedding_tsne.png")))"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            f"""### השוואת הייצוגים

{_table_rows(embedding_display, list(embedding_display.columns))}

DeepWalk לבדו השיג ROC-AUC {_f(key['deepwalk_roc_auc'])}, ו־Node2Vec לבדו {_f(key['node2vec_roc_auc'])}. ההשוואה למאפיינים ידניים ולתוכן מראה עד כמה ההילוכים האקראיים מרכזים מידע שימושי במבנה הרשת. ביצועים דומים אינם הוכחת שקילות בין מקורות המידע; הם מתארים רק את החלוקות, הפרמטרים והמסווג שנבדקו."""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 9. RQ3 — שחזור קשתות תוך שמירת קשירות

לכל seed נבנה עץ פורש, קשתותיו מוגנות, וקשתות test נדגמות רק מן הקשתות שאינן בעץ. לאחר ההסרה נבדקים במפורש `nx.is_connected(G_train)` והיעדר צמתים מבודדים. שמירת הקשירות חיונית למדדים מבוססי מסלולים, לקהילות ולהילוכים של DeepWalk/Node2Vec.

הדוגמאות השליליות הן non-edges אמיתיים מן הגרף המלא ומותאמות במספרן לכל אחת מקבוצות Web-Web, ML-ML ו־Web-ML. לכן same-field ו־cross-field נבדקות תחת כלל דגימה זהה."""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """from run_link_prediction import main as run_edge_reconstruction
run_edge_reconstruction()
display(pd.read_csv(TABLES_DIR / "edge_split_diagnostics.csv"))
display(pd.read_csv(TABLES_DIR / "edge_reconstruction_summary.csv").round(4))
display(Image(filename=str(FIGURES_DIR / "edge_reconstruction_auc.png")))
display(pd.read_csv(TABLES_DIR / "edge_reconstruction_same_vs_cross_summary.csv").round(4))
display(pd.read_csv(TABLES_DIR / "edge_reconstruction_same_vs_cross_gaps.csv").round(4))
display(Image(filename=str(FIGURES_DIR / "edge_reconstruction_same_vs_cross.png")))"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            f"""### תוצאות RQ3

**{best_heuristic}** היה החזק ביותר בשחזור הכולל עם ROC-AUC {_f(key['best_reconstruction_roc_auc'])}. באותו מדד, AUC של same-field הוא {_f(same_auc)} ושל cross-field {_f(cross_auc)} (פער {_f(best_gap['roc_auc_same_minus_cross'])}); Average Precision הוא {_f(same_ap)} מול {_f(cross_ap)}.

לכן, עבור ההיוריסטיקה המובילה ובפרוטוקול המאוזן שנבדק, יכולת השחזור של קשתות תוך־תחומיות **{relationship}** מזו של קשתות בין־תחומיות. התוצאה אינה מוכיחה מנגנון סיבתי של יצירת קשרים ואינה ניתנת להכללה לתחזית זמן."""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 10. RQ4 — מודלי אפס ומודלי רשת

נבדקים 1,000 ערבובי תוויות ו־20 גרפים משמרי־דרגות. ערכי p אמפיריים מחושבים עם תיקון finite-simulation: \((b+1)/(B+1)\), ולכן אינם יכולים להיות אפס. בנוסף מושווים מדדי הרשת למודלי ER, BA, Watts-Strogatz ולגרף קונפיגורציה פשוט משמר־דרגות."""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """from run_null_models import main as run_null_models
run_null_models()
display(pd.read_csv(TABLES_DIR / "null_model_homophily.csv").round(6))
display(pd.read_csv(TABLES_DIR / "null_model_modularity.csv").round(6))
display(Image(filename=str(FIGURES_DIR / "homophily_null_distribution.png")))
display(pd.read_csv(TABLES_DIR / "network_vs_random_models.csv").round(4))
display(pd.read_csv(TABLES_DIR / "powerlaw_fit.csv").round(4))
display(Image(filename=str(FIGURES_DIR / "degree_distribution_loglog.png")))"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            f"""### תוצאות RQ4

- ההומופיליה הנצפית ({_f(key['homophily_rate'])}) גבוהה מממוצע ערבוב התוויות ({_f(label_null.loc['label_shuffle', 'null_mean'])}) ומממוצע הגרפים משמרי־הדרגות ({_f(label_null.loc['degree_preserving_rewire', 'null_mean'])}). ערכי p המתוקנים הם {_f(key['label_shuffle_empirical_p'], 6)} ו־{_f(key['degree_rewire_empirical_p'], 6)}.
- המודולריות הנצפית היא {_f(null_modularity['observed_modularity'])}, מול {_f(null_modularity['rewired_null_mean'])} בממוצע בגרפים משמרי־הדרגות; p אמפירי {_f(null_modularity['empirical_p'], 6)}.
- ההתקבצות הממוצעת ברשת היא {_f(observed_network['avg_clustering'])}, מול {_f(er_network['avg_clustering'], 4)} ב־ER, {_f(network_rows.loc[ba_name, 'avg_clustering'], 4)} ב־BA ו־{_f(network_rows.loc[ws_name, 'avg_clustering'])} ב־Small-World. מודל הקונפיגורציה משמר את הדרגות אך משחזר התקבצות של {_f(config_network['avg_clustering'])}.
- התאמת הזנב נותנת \\(\\alpha={_f(powerlaw['alpha'], 2)}\\), אך ההשוואה ללוג־נורמלית אינה מצדיקה הכרזה גורפת על רשת scale-free. המסקנה הבטוחה היא זנב כבד."""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 11. דיון, מגבלות ומסקנות

**RQ1:** מבנה הרשת מספק אות חיזויי עקבי; קהילות וייצוגים נלמדים משפרים את קווי הבסיס.<br>
**RQ2:** תוכן חזק מן הפיצ'רים הידניים של הרשת, אך השילוב מוסיף מידע — חפיפה חלקית, לא שקילות.<br>
**RQ3:** שחזור קשתות אפשרי מעל רמת אקראי תוך שמירת קשירות. ההבדל same/cross תלוי בהיוריסטיקה ובפרוטוקול, ולכן מנוסח כתוצאה תיאורית.<br>
**RQ4:** ההומופיליה והמודולריות חריגות ביחס למודלי האפס שנבדקו; הבדלי דרגות מסבירים חלק ניכר אך לא את כל הפער.

### מגבלות

- הנתונים הם תמונת מצב מיוני 2019 ותווית היעד בינארית ורועשת אפשרית.
- זהו ניסוי transductive; מבנה הגרף כולו גלוי בסיווג הצמתים.
- Betweenness מחושב בקירוב בגלל גודל הרשת.
- פרמטרי ההטמעה נקבעו מראש ולא בוצע חיפוש נרחב.
- שחזור קשתות שהוסתרו אינו תחזית זמן; לא נבדקו כיווניות, עוצמת קשר או סיבתיות.
- מודלי האפס והמודלים הגנרטיביים הם השוואות ממוקדות, לא תיאור מלא של כל תהליכי הרשת.

בסך הכול נבדקו **10 מודלי סיווג לוגיים**: ארבעה מודלי בסיס/פיצ'רים ידניים, תוכן והיברידי, וארבעה מודלים המבוססים על DeepWalk/Node2Vec לבדם ובשילוב ידני."""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 12. מקורות

1. Rozemberczki, B., Allen, C., & Sarkar, R. (2019). *Multi-scale Attributed Node Embedding (MUSAE)*.
2. Blondel, V. D., et al. (2008). *Fast unfolding of communities in large networks*.
3. Perozzi, B., Al-Rfou, R., & Skiena, S. (2014). *DeepWalk*.
4. Grover, A., & Leskovec, J. (2016). *node2vec: Scalable Feature Learning for Networks*.
5. Liben-Nowell, D., & Kleinberg, J. (2003). *The Link Prediction Problem for Social Networks*.
6. Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). *Power-law distributions in empirical data*.
7. Page, L., et al. (1999). *The PageRank Citation Ranking*.
8. מצגות ותרגולי הקורס בנושאי מרכזיות, קהילות, מודלי רשת, שחזור קשתות והטמעות גרף."""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 13. Validation סופי

התא הבא נכשל אם חסר פלט, CSV ריק או לא־סופי, מדד מחוץ לטווח, מודל לא מסונכרן, ערך p לא תקין, גרף אימון מנותק, צומת מבודד, חוסר התאמה בין ערכי המחברת לקובצי ה־CSV, או דוח החורג משמונה עמודים."""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """from final_validation import run_all_validations
run_all_validations(PROJECT_DIR, verify_notebook_execution=False)"""
        )
    )

    nb["cells"] = cells
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Built notebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
