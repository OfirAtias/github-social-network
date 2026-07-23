# -*- coding: utf-8 -*-
"""Generate README.md from the current result tables."""

from pathlib import Path
import sys

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from final_validation import collect_key_results


def _f(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def main() -> None:
    key = collect_key_results(PROJECT_DIR)
    tables = PROJECT_DIR / "outputs" / "tables"
    ablation = pd.read_csv(
        tables / "ablation_model_results_summary.csv"
    ).set_index("model")
    groups = pd.read_csv(
        tables / "edge_reconstruction_same_vs_cross_summary.csv"
    )
    best_groups = groups[
        groups["heuristic"] == key["best_reconstruction_heuristic"]
    ].set_index("edge_group")
    modularity = pd.read_csv(
        tables / "null_model_modularity.csv"
    ).iloc[0]
    content_auc = ablation.loc[
        "Logistic Regression - Content Features Only", "roc_auc_mean"
    ]
    network_auc = ablation.loc[
        "Logistic Regression - Network Only (Graph + Community)",
        "roc_auc_mean",
    ]
    hybrid_auc = ablation.loc[
        "Logistic Regression - Hybrid (Network + Content)",
        "roc_auc_mean",
    ]

    readme = f"""# תחום התמחות של מפתחי GitHub מתוך רשת מעקב הדדית

פרויקט גמר בקורס **אלגוריתמים ברשתות**. הפרויקט מנתח רשת לא־מכוונת של
{key['network_nodes']:,} מפתחי GitHub ו־{key['network_edges']:,} קשרי מעקב
הדדיים, ובוחן כמה מידע על התחום המקצועי (Web מול Machine Learning) נמצא
בשכנות, במרכזיות, בקהילות ובהטמעות גרף.

הגרסה המרכזית משתמשת בשיטות הקורס בלבד. תיקיית `extensions/` שומרת ניסויים
אופציונליים שאינם חלק מהצנרת, מהתוצאות או מטענות ההגשה.

## שאלות המחקר

1. **RQ1 — סיווג צמתים:** באיזו מידה שכנות, מרכזיות, קהילות וייצוגים נלמדים
   מאפשרים לסווג את תחום המפתח ללא שימוש בתוויות הבדיקה?
2. **RQ2 — רשת מול תוכן:** מהו הערך השולי של מבנה הרשת מול מאפייני התוכן,
   ומה מוסיף שילובם?
3. **RQ3 — שחזור קשתות:** האם ניתן לשחזר קשתות מעקב שהוסתרו אקראית באמצעות
   היוריסטיקות מבניות וייצוגים נלמדים, והאם קשתות תוך־תחומיות קלות יותר
   לשחזור מקשתות בין־תחומיות?
4. **RQ4 — מודלי אפס ורשת:** האם ההומופיליה והמודולריות הנצפות חריגות ביחס
   לערבוב תוויות ולגרפים משמרי־דרגות, ואילו מאפיינים משחזרים ER, BA,
   Small-World ו־Configuration?

RQ3 עוסק **בשחזור קשתות שהוסתרו מתמונת מצב סטטית**, לא בתחזית זמן ולא
בהסקת מנגנון סיבתי.

## הדאטה

הנתונים הם MUSAE GitHub Social Network (יוני 2019):

- `data/musae_git_edges.csv` — {key['network_edges']:,} קשתות ללא לולאות או
  כפילויות.
- `data/musae_git_target.csv` — תווית בינארית מלאה לכל {key['network_nodes']:,}
  הצמתים.
- `data/musae_git_features.json` — מאפייני תוכן דלילים המשמשים רק בניסוי
  ההשוואה רשת/תוכן.
- הרשת היא רכיב קשיר יחיד; שיעור הקשתות התוך־תחומיות הוא
  {_f(100 * key['homophily_rate'], 1)}%.

## אלגוריתמים בגרסה המרכזית

- ניתוח מבני: Degree Centrality, Clustering Coefficient, PageRank,
  Betweenness Centrality (בקירוב), HITS Hub יחיד, Louvain ומודולריות.
- מודלי רשת: Erdős–Rényi, Barabási–Albert, Watts–Strogatz Small-World,
  Configuration/degree-preserving null ו־Power Law.
- סיווג: Majority, Neighbor Majority ורגרסיה לוגיסטית על פיצ'רים ידניים,
  קהילה, תוכן ושילובים.
- ייצוגים נלמדים: DeepWalk ו־Node2Vec, לבדם ובשילוב פיצ'רים ידניים.
- שחזור קשתות: Common Neighbors, Jaccard, Adamic-Adar, Preferential
  Attachment, Same Community, DeepWalk ו־Node2Vec.
- מודלי אפס: 1,000 ערבובי תוויות ו־20 גרפים משמרי־דרגות, עם
  `(extreme_count + 1) / (num_simulations + 1)`.

בסך הכול מדוּוחים **10 מודלי סיווג לוגיים**: ארבעה קווי בסיס/מודלים ידניים,
תוכן והיברידי, וארבעה מודלי DeepWalk/Node2Vec לבדם ובשילוב ידני.

## תוצאות מרכזיות מעודכנות

- הסיווג הידני החזק ביותר הוא **{key['best_classification_model']}** עם
  ROC-AUC ממוצע {_f(key['best_classification_roc_auc'])}.
- רשת ידנית מול תוכן מול היברידי: {_f(network_auc)} מול {_f(content_auc)}
  מול {_f(hybrid_auc)} ROC-AUC.
- DeepWalk לבדו: {_f(key['deepwalk_roc_auc'])}; Node2Vec לבדו:
  {_f(key['node2vec_roc_auc'])} ROC-AUC. דמיון בביצועים אינו הוכחת שקילות
  בין מקורות מידע.
- **{key['best_reconstruction_heuristic']}** מוביל בשחזור הקשתות עם
  ROC-AUC {_f(key['best_reconstruction_roc_auc'])}.
- עבור שיטה זו, same-field מול cross-field:
  {_f(best_groups.loc['same-field', 'roc_auc_mean'])} מול
  {_f(best_groups.loc['cross-field', 'roc_auc_mean'])} ROC-AUC,
  ו־{_f(best_groups.loc['same-field', 'average_precision_mean'])} מול
  {_f(best_groups.loc['cross-field', 'average_precision_mean'])} Average
  Precision.
- ערכי p האמפיריים הם
  {_f(key['label_shuffle_empirical_p'], 6)} (ערבוב תוויות),
  {_f(key['degree_rewire_empirical_p'], 6)} (גרף משמר־דרגות) ו־
  {_f(key['modularity_empirical_p'], 6)} (מודולריות); אף אחד מהם אינו אפס.
- המודולריות הנצפית היא {_f(modularity['observed_modularity'])}, מול
  {_f(modularity['rewired_null_mean'])} בממוצע בגרפים משמרי־דרגות.

## תכנון הניסויים

- כל תוצאות הסיווג מדווחות כממוצע וסטיית תקן על seeds
  `42, 7, 21, 84, 123`.
- מאפיינים המבוססים על תוויות משתמשים בתוויות train בלבד.
- בחירת סף נעשית על validation פנימי, לא על test.
- DeepWalk ו־Node2Vec נלמדים ללא תוויות.
- בשחזור קשתות מוגן עץ פורש; כל קשתות test מגיעות מן הקשתות שאינן בעץ.
  כל גרף אימון נבדק עם `assert nx.is_connected(G_train)` וללא צמתים
  מבודדים.
- non-edges נדגמים באותו מספר בתוך Web-Web, ML-ML ו־Web-ML, כך שהשוואת
  same-field/cross-field הוגנת.

## מבנה הריפוזיטורי

```text
data/                 נתוני MUSAE המקוריים
src/                  קוד הצנרת, builders ו-validation
extensions/           הרחבות אופציונליות מחוץ להגשה המרכזית
notebooks/            המחברת הסופית והמורצת
outputs/tables/       כל טבלאות ה-CSV העדכניות
outputs/figures/      כל הגרפים העדכניים ברזולוציה גבוהה
outputs/embeddings/   קובצי DeepWalk ו-Node2Vec
report/report_source.docx   מקור העריכה של הדוח
report/final_report.pdf     הדוח הסופי (עד 8 עמודים)
tests/                בדיקות יחידה לפרוטוקולים הקריטיים
```

## התקנה והרצה — Windows PowerShell

```powershell
cd C:\\path\\to\\github-social-network
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONHASHSEED="0"

# מוחק רק פלטים ישנים תחת outputs/ ובונה הכול מחדש
python src/run_all.py

# ה-PDF וה-DOCX נבנו כבר ב-run_all; לבנייה ממוקדת של ה-PDF:
python src/build_report_pdf.py

# הרצת המחברת כולה מתוך הריפוזיטורי
python -m nbconvert --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=-1 `
  notebooks/GitHub_Developer_Field_Prediction_Network_Only.ipynb

python src/final_validation.py
```

## התקנה והרצה — Linux/macOS

```bash
cd /path/to/github-social-network
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONHASHSEED=0

python src/run_all.py

# ה-PDF וה-DOCX נבנו כבר ב-run_all; לבנייה ממוקדת של ה-PDF:
python src/build_report_pdf.py

python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  notebooks/GitHub_Developer_Field_Prediction_Network_Only.ipynb

python src/final_validation.py
```

## הרצה ממוקדת

```bash
python src/run_core_analysis.py
python src/run_classification.py
python src/run_ablation.py
python src/run_embeddings.py
python src/run_link_prediction.py
python src/run_null_models.py
python src/build_notebook.py
python src/build_report.py
python src/build_report_pdf.py
```

`python src/run_all.py` הוא המסלול הסמכותי: הוא מנקה את `outputs/`, מריץ את
השלבים בסדר הנכון ובונה מחדש את המחברת, ה־README, מקור הדוח וה־PDF הסופי.

## Validation סופי

`python src/final_validation.py` נכשל אם:

- חסר פלט או CSV ריק/לא־קריא;
- יש NaN/Inf או מדד מחוץ לטווח;
- מספרי הצמתים/קשתות אינם תואמים;
- יש לולאות, כפילויות או תוויות חסרות;
- אחד מחמשת גרפי האימון מנותק או מכיל מבודדים;
- positive/negative אינם מותאמים לפי סוג קשת;
- ערך p אמפירי הוא אפס או גדול מאחד;
- מודל חסר/עודף או פלט ישן נשאר;
- ערכי המחברת אינם תואמים ל־CSV;
- הדוח חסר מקור עריכה או חורג משמונה עמודים.

רק לאחר כל הבדיקות מודפסת ההודעה:

```text
ALL FINAL PROJECT VALIDATIONS PASSED
```

## מגבלות

- הנתונים נאספו ביוני 2019 ותווית היעד בינארית.
- הסיווג transductive ומניח שמבנה הגרף ידוע.
- Betweenness מקורב; תקציב ההטמעות קבוע ולא בוצע חיפוש פרמטרים נרחב.
- שחזור קשתות שהוסתרו אינו תחזית זמן ואינו הוכחה סיבתית.
- מודלי האפס והמודלים הגנרטיביים מבודדים מאפיינים מסוימים ואינם תיאור מלא
  של כל תהליכי הרשת.

## מקורות עיקריים

MUSAE; Louvain; PageRank; DeepWalk; Node2Vec; Link Prediction; Power-law
distributions; ומצגות/תרגולי הקורס בנושאי מרכזיות, קהילות ומודלי רשת.
"""
    (PROJECT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(f"Built README: {PROJECT_DIR / 'README.md'}")


if __name__ == "__main__":
    main()
