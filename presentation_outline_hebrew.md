# חיזוי תחום התמחות של מפתחי GitHub לפי קשרי מעקב הדדיים ברשת

## שקף 1: כותרת ושמות
- חיזוי תחום התמחות של מפתחי GitHub לפי קשרי מעקב הדדיים ברשת
- פרויקט גמר בקורס אלגוריתמים בגרפים 2026
- חברי צוות: שם סטודנט 1, שם סטודנט 2, שם סטודנט 3
- שאלת המחקר: באיזו מידה מאפיינים מבניים של רשת קשרי המעקב ההדדיים מאפשרים לחזות האם מפתח הוא Web או Machine Learning?

גרף מומלץ: אין

## שקף 2: מבוא ומוטיבציה
- GitHub היא גם פלטפורמת קוד וגם רשת חברתית-מקצועית.
- קשרי מעקב הדדיים עשויים לשקף קרבה מקצועית, תחומי עניין או השתייכות קהילתית.
- הפרויקט בודק חיזוי **network-only**: ללא טקסט, ללא פרופילים, ללא מאפייני repositories.
- המטרה היא לבדוק האם מבנה הרשת עצמו מכיל סיגנל לגבי תחום התמחות.

גרף מומלץ: `outputs/figures/sampled_network_by_label.png`

## שקף 3: תיאור הדאטה והגרף
- צמתים: 37,700 מפתחי GitHub
- קשתות: 289,003 קשרי מעקב הדדיים
- הגרף לא מכוון ומחובר
- Web: 27,961 (74.17%)
- Machine Learning: 9,739 (25.83%)
- הדאטה לא מאוזן, לכן Accuracy אינו מספיק.

גרפים מומלצים:
- `outputs/figures/label_distribution.png`
- `outputs/figures/degree_distribution.png`

## שקף 4: ניסוח הבעיה והגדרת הניסוי
- משימה: Binary Node Classification
- קלט למודלים: מבנה הגרף ומאפיינים שנגזרים ממנו בלבד.
- פלט: תווית Web או Machine Learning לכל צומת test.
- הגדרה מתודולוגית: Transductive Node Classification.
- מבנה הגרף ידוע לכל הצמתים, אבל תוויות test אינן משמשות באימון.

גרף מומלץ: אין

## שקף 5: הומופיליה ואסורטטיביות
- בדקנו האם קשתות נוטות לחבר מפתחים מאותו תחום.
- same-label edge rate: כ-84.53%.
- המשמעות: השכנות בגרף אינה אקראית ביחס לתחום ההתמחות.
- ממצא זה מסביר מדוע מודלים מבוססי שכנות וקהילה יכולים לעבוד.

גרף מומלץ: `outputs/figures/edge_type_distribution.png`

## שקף 6: האלגוריתמים
- Majority Baseline: קו בסיס לא גרפי.
- Neighbor Majority: חיזוי לפי תוויות שכני train.
- Label Propagation: אלגוריתם גרפי ישיר להפצת תוויות דרך הקשתות.
- Graph Features: degree, PageRank, clustering coefficient, core number, יחסי שכנים.
- Louvain: זיהוי קהילות ומאפייני קהילה.
- Graph + Community Model: שילוב מאפיינים מקומיים וגלובליים.

גרפים מומלצים:
- `outputs/figures/community_size_distribution.png`
- `outputs/figures/top_communities_label_composition.png`

## שקף 7: תוצאות המודלים
| מודל | F1 | Macro F1 | ROC-AUC |
|---|---:|---:|---:|
| Majority Baseline | 0.0000 | 0.4258 | 0.5000 |
| Neighbor Majority | 0.5165 | 0.7025 | 0.7851 |
| Label Propagation | 0.5517 | 0.7228 | 0.8021 |
| Graph Features Only | 0.5909 | 0.7416 | 0.8415 |
| Graph + Community Features | 0.6494 | 0.7736 | 0.8683 |

גרפים מומלצים:
- `outputs/figures/repeated_f1_comparison_with_communities.png`
- `outputs/figures/repeated_macro_weighted_f1_comparison_with_communities.png`

## שקף 8: כיול סף וניתוח שגיאות
- סף 0.5: F1 = 0.6379, Recall = 0.5582
- סף 0.4 לפי validation F1: F1 = 0.6581, Recall = 0.6294
- קהילות מעורבות קשות יותר לחיזוי.
- עדיין יש יותר טעויות ML -> Web מאשר Web -> ML.

גרפים מומלצים:
- `outputs/figures/threshold_f1_recall_precision.png`
- `outputs/figures/community_purity_error_analysis.png`
- `outputs/figures/prediction_error_types.png`

## שקף 9: קשיים ומגבלות
- הדאטה מ-2019 ולכן אינו בהכרח מייצג את GitHub כיום.
- המשימה בינארית בלבד, אף שבפועל קיימים תחומי התמחות רבים.
- התוויות נגזרו מתיאור תפקיד ועלולות לכלול רעש.
- לא השתמשנו במאפייני תוכן כדי לשמור על network-only, ולכן תקרת הביצועים מוגבלת.
- חישובי Betweenness/Closeness על כל הגרף כבדים ולכן לא הוכנסו למודל.

גרף מומלץ: `outputs/figures/error_analysis_by_degree_group_tuned_model.png`

## שקף 10: מסקנות
- מבנה קשרי המעקב ההדדיים מכיל סיגנל משמעותי לגבי תחום התמחות.
- מידע שכנותי עוזר, אבל אינו מספיק לבדו.
- Label Propagation מחזק את ההיבט האלגוריתמי-גרפי של העבודה.
- מאפייני גרף וקהילות Louvain משפרים את החיזוי מעבר לשכנות מקומית.
- הפרויקט מתאים לקורס אלגוריתמים בגרפים כי הפתרון נשען על מבנה, קהילות, מרכזיות וסיווג מבוסס גרף.

גרף מומלץ: `outputs/figures/repeated_roc_auc_comparison_with_communities.png`
