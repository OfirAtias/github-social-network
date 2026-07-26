# סדרת מחברות המחקר

סדרה זו היא הנתיב המתועד והמורץ של הפרויקט של נועה ירין לוי, ג'ימי חלאוי ואופיר אטיאס. היא נבנתה מחדש מתוך **כל 20 קובצי `src/*.py`** ובהתאם ל-`Graph_Algo_project_guidelines_2026 (3).pdf`. קובצי המקור אינם משתנים; קוד המקור מופיע בתאי קוד עם נתיב וטווח שורות, ומחברת 09 מוכיחה התאמה byte-for-byte.

## סדר ההרצה

| סדר | מחברת | מטרה | תפקיד | קשר למחקר |
|---:|---|---|---|---|
| 01 | [`01_environment_and_reproducibility.ipynb`](01_environment_and_reproducibility.ipynb) | סביבת עבודה, התקנות ושחזוריות | מוודאת שהריפוזיטורי, הנתונים והתלויות זמינים לפני כל ניתוח. | כל שאלות המחקר: זהו בסיס השחזוריות המשותף ל-RQ1-RQ4. |
| 02 | [`02_data_loading_and_preparation.ipynb`](02_data_loading_and_preparation.ipynb) | טעינת הנתונים, אימותם והכנת אובייקטי הבסיס | טוענת את קשתות MUSAE ואת תוויות המפתחים, מאמתת אותן ובונה גרף לא-מכוון. | מגדירה את יחידת הניתוח המשותפת ל-RQ1-RQ4 ומונעת כשלים סמויים בנתונים. |
| 03 | [`03_cleaning_eda_and_graph_statistics.ipynb`](03_cleaning_eda_and_graph_statistics.ipynb) | ניקוי, EDA וסטטיסטיקה תיאורית של הרשת | מחשבת איכות נתונים, התפלגות תוויות, הומופיליה, מרכזיות ותרשימי בסיס. | מספקת את הראיות המבניות שמניעות את RQ1 ואת השערת ההומופיליה של RQ4. |
| 04 | [`04_communities_and_problem_definition.ipynb`](04_communities_and_problem_definition.ipynb) | קהילות Louvain והגדרה תפעולית של בעיות המחקר | מאתרת קהילות, מודדת מודולריות וטוהר ומגדירה תכונות קהילה ללא דליפת תוויות. | מבססת את רכיב הקהילה ב-RQ1 ואת השוואת המודולריות ב-RQ4. |
| 05 | [`05_node_classification_algorithms.ipynb`](05_node_classification_algorithms.ipynb) | מימוש אלגוריתמי סיווג צמתים וניסוי חוזר | מגדירה baselines, מאפייני גרף/שכנים/קהילה, רגרסיה לוגיסטית וכיוון סף ללא leakage. | עונה ישירות על RQ1 ומכינה baseline הוגן להשוואות RQ2. |
| 06 | [`06_embeddings_content_and_ablation.ipynb`](06_embeddings_content_and_ablation.ipynb) | DeepWalk, Node2Vec, תוכן וניסויי אבלציה | לומדת ייצוגי צמתים מהגרף ומשווה רשת, תוכן ושילוב תחת אותו פרוטוקול. | עונה על RQ2 ומרחיבה את RQ1 בייצוגים נלמדים בלתי-מפוקחים. |
| 07 | [`07_link_reconstruction_experiments.ipynb`](07_link_reconstruction_experiments.ipynb) | שחזור קשתות: פרוטוקול, אלגוריתמים וניסויים | מסתירה קשתות תוך שמירת קישוריות ומשווה heuristics מבניים ל-embeddings. | עונה על RQ3; מדובר בשחזור קשתות סטטי ולא בתחזית זמן. |
| 08 | [`08_null_models_results_and_research_answers.ipynb`](08_null_models_results_and_research_answers.ipynb) | מודלי אפס, מובהקות ותשובות לשאלות המחקר | משווה את הרשת לערבובי תוויות, rewiring משמר-דרגה ומודלים גנרטיביים ומרכזת מסקנות. | עונה ישירות על RQ4 ומסכמת באופן כמותי את RQ1-RQ3. |
| 09 | [`09_EXTENTIONS_and_validation.ipynb`](09_EXTENTIONS_and_validation.ipynb) | EXTENTIONS — הרחבות, הפקת תוצרים ואימות סופי | מפרידה הרחבות ניסיוניות מן הליבה, מציגה builders ומריצה בדיקות שחזוריות וכיסוי. | מבהירה מהו המימוש המרכזי ומהן הרחבות עתידיות, כנדרש בהנחיות הפרויקט. |

המחברת הישנה `GitHub_Developer_Field_Prediction_Network_Only.ipynb` נשמרת לצורך עקיבות, אך הסדרה הממוספרת היא התיעוד המחקרי המפורט.

## הרצה מלאה ב-Windows PowerShell

```powershell
Set-Location (git rev-parse --show-toplevel)
$env:PYTHONHASHSEED="0"
if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    py -3.11 -m venv .venv
}
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
New-Item -ItemType Directory -Force .\tmp\jupyter-runtime | Out-Null
.\.venv\Scripts\python.exe -m ipykernel install --prefix .\tmp\jupyter-runtime --name github-social-network --display-name "GitHub Social Network (.venv)"
$env:JUPYTER_PATH = (Resolve-Path .\tmp\jupyter-runtime\share\jupyter).Path
$notebooks = Get-ChildItem .\notebooks -Filter "[0-9][0-9]_*.ipynb" | Sort-Object Name
foreach ($notebook in $notebooks) {
    .\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=github-social-network --ExecutePreprocessor.timeout=-1 $notebook.FullName
}
```

כל מחברת מאתרת את שורש הפרויקט גם אם Jupyter פתח אותה מתוך `notebooks/`. מחברת 01 מתקינה תלויות חסרות באופן מותנה. להרצה מלאה מומלץ להשתמש ב-Python 3.11 ובסביבה הווירטואלית של הפרויקט.

## מפת תלות

`01 סביבה` → `02 נתונים` → `03 EDA` → `04 קהילות` → `05 סיווג` → `06 embeddings/ablation` → `07 שחזור קשתות` → `08 nulls ותשובות` → `09 EXTENTIONS ואימות`.

הפלטים נשמרים ב-`outputs/tables`, `outputs/figures` ו-`outputs/embeddings`. בכל מחברת מופיעים הסברים לפני קוד, פרשנות אחרי פלטים וסיכום המקשר לשלב הבא.
