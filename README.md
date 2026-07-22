# חיזוי תחום התמחות של מפתחי GitHub לפי קשרי מעקב הדדיים ברשת

פרויקט גמר בקורס אלגוריתמים בגרפים. הפרויקט חוזה את תחום ההתמחות (Web מול Machine Learning) של 37,700 מפתחי GitHub מתוך מבנה רשת קשרי המעקב ההדדיים בלבד, ובוחן את הרשת מארבע זוויות מחקריות.

## שאלות המחקר

1. **חיזוי צמתים:** באיזו מידה מאפיינים מבניים של הרשת (שכנות, מרכזיות, קהילות וייצוגים נלמדים כגון DeepWalk ו-GCN) מאפשרים לחזות את תחום המפתח?
2. **רשת מול תוכן:** מהו הערך השולי של מבנה הרשת ביחס למאפייני תוכן (repositories מסומני כוכב, מעסיק, מיקום)?
3. **שחזור קשתות:** האם ניתן לשחזר קשתות מעקב שהוסתרו באמצעות היוריסטיקות מבניות, והאם קשתות תוך-תחומיות קלות יותר לשחזור?
4. **תוקף מבני:** האם ההומופיליה, המבנה הקהילתי והתפלגות הדרגות מובהקים סטטיסטית מול מודלי רשת אקראיים?

## הדאטה

MUSAE GitHub Social Network (נאסף מה-API הציבורי, יוני 2019):
- 37,700 צמתים (מפתחים), 289,003 קשתות (מעקב הדדי), רכיב קשיר אחד.
- תוויות: Web ‏(74.17%) מול Machine Learning ‏(25.83%).
- `musae_git_features.json`: ‏4,005 מאפייני תוכן בינאריים לצומת. אינם משמשים כקלט בצנרת המרכזית, רק כנקודת ייחוס בניסוי ה-Ablation וכבסיס לגילוי תת-התמחויות.

קבצי הדאטה בתיקיית `data/`.

## מסגרת מתודולוגית

- **Transductive node classification:** מבנה הגרף ידוע לכל הצמתים; תוויות test אינן משמשות באימון, בחישוב מאפיינים או בבחירת ספים.
- **פרוטוקול אחיד:** כל הניסויים רצים על אותן 5 חלוקות Train/Test מרובדות (70/30, seeds ‏42, 7, 21, 84, 123). הבדל בין מודלים מדווח כעקבי רק אם הוא מופיע בכל חמש החלוקות.
- **מניעת דליפה בכל משימה:** מאפייני תוויות מ-train בלבד; DeepWalk נלמד ללא תוויות; ב-GCN התוויות רק דרך פונקציית ההפסד ו-epoch נבחר על validation פנימי; בחיזוי קשרים כל ההיוריסטיקות (כולל Louvain) מחושבות על גרף האימון בלבד.

## תוצאות מרכזיות: חיזוי צמתים (ממוצע על 5 חלוקות)

| מודל | F1 | Macro F1 | ROC-AUC |
|---|---:|---:|---:|
| Majority Baseline | 0.000 | 0.426 | 0.500 |
| Neighbor Majority | 0.517 | 0.703 | 0.785 |
| Label Propagation | 0.552 | 0.723 | 0.802 |
| LogReg - Graph Features | 0.637 | 0.751 | 0.841 |
| LogReg - Graph + Community | 0.672 | 0.772 | 0.869 |
| GCN - Structural Input | 0.580 | 0.679 | 0.816 |
| GraphSAGE - Structural Input | 0.585 | 0.697 | 0.808 |
| LogReg - DeepWalk | 0.694 | 0.786 | 0.887 |
| LogReg - Content Only | 0.699 | 0.790 | 0.889 |
| GraphSAGE - DeepWalk Input | 0.703 | 0.791 | 0.893 |
| GCN - DeepWalk Input | 0.707 | 0.794 | 0.896 |
| LogReg - DeepWalk + Handcrafted | 0.703 | 0.791 | **0.898** |
| LogReg - Hybrid (Network + Content) | **0.709** | **0.797** | 0.898 |

כל מעבר בסולם המרכזי משפר את קודמו בכל אחת מחמש החלוקות. ממצאים בולטים:
- **הרשת מגיעה לרמת התוכן:** DeepWalk ‏(0.887) השיג ביצועים דומים מאוד ל-Content ‏(0.889) בכל חמש החלוקות.
- **ייצוג חשוב מארכיטקטורה:** בשתי ארכיטקטורות GNN ‏(GCN ו-GraphSAGE), החלפת הקלט שווה ~+0.08 AUC והחלפת הארכיטקטורה פחות מ-0.01.
- **תקרת ביצועים משותפת:** ארבעת המודלים המובילים מתכנסים ל-AUC≈0.90, עקבי עם רעש בתוויות.

## ממצאים נוספים

- **שחזור קשתות:** Preferential Attachment מוביל (AUC ‏0.910). בפירוק לפי סוגי קשתות, צפיפות שכונתית ולא דמיון תוויות היא שמסבירה את השחזור (Web-Web ‏0.858 > ‏ML-ML ‏0.767 ב-Adamic-Adar). ציון DeepWalk מבוסס-דמיון (0.694) מפסיד להיוריסטיקת הדרגות, כלומר ייצוגים נלמדים אינם שולטים בכל משימה.
- **הומופיליה מפורקת לגורמים:** 84.5% קשתות same-label, רחוק מכל 1,000 ערבובי התוויות (צפי 61.7%) ומכל 20 הערבולים משמרי-הדרגות (צפי 75.1%); כ-60% מוסברים בהבדלי דרגות, והעודף (~9.4 נק') הוא העדפה תוך-תחומית אמיתית.
- **מודולריות:** 0.453 מול 0.189±0.001 ברשתות מעורבלות משמרות-דרגות.
- **תת-התמחויות:** קהילות הרשת מיושרות עם אשכולות תוכן (NMI ‏0.257) פי ~2.4 יותר מאשר עם התווית הבינארית (0.106).
- **אפיון הרשת:** Small-World (התקבצות פי ~420 מ-ER, מרחק ממוצע 3.22), כבדת-זנב (α=2.54, לא ניתן להבחנה מלוג-נורמלית).

## מבנה הפרויקט

```
data/                  קבצי הדאטה (MUSAE)
notebooks/             המחברת הראשית (23 פרקים, מקצה לקצה)
src/
  data_loader.py       טעינת דאטה ובניית הגרף
  graph_analysis.py    סטטיסטיקה מבנית והומופיליה
  community_analysis.py  Louvain ומאפייני קהילה
  modeling.py          חמשת מודלי הבסיס, הערכה, כיול סף
  visualization.py     גרפים של הצנרת הבסיסית
  content_features.py  מאפייני תוכן ומודלי Ablation
  embeddings.py        DeepWalk (הילוכים + Word2Vec)
  gnn.py               GCN ו-GraphSAGE ב-PyTorch (מטריצות דלילות)
  link_prediction.py   פרוטוקול והיוריסטיקות לחיזוי קשרים
  run_ablation.py      ניסוי רשת/תוכן/היברידי
  run_subfields.py     גילוי תת-התמחויות
  run_deepwalk.py      ניסוי DeepWalk + t-SNE
  run_gnn.py           ניסוי GNN ‏2x2: שתי ארכיטקטורות, שתי גרסאות קלט
  run_link_prediction.py  חיזוי קשרים
  run_null_models.py   מודלי אפס ומובהקות מבנית
outputs/
  tables/              כל טבלאות התוצאות (CSV)
  figures/             כל האיורים (PNG)
  embeddings/          ייצוגי DeepWalk שמורים (NPZ)
report/                הדוח המסכם
```

## הרצה

```bash
pip install -r requirements.txt
PYTHONHASHSEED=0 python -m nbconvert --to notebook --execute --inplace notebooks/GitHub_Developer_Field_Prediction_Network_Only.ipynb
```

המחברת מריצה את הכל מקצה לקצה (מספר שעות). לחלופין, כל ניסוי ניתן להרצה עצמאית: `python src/run_ablation.py` וכו'. הערה: `run_gnn.py` דורש ש-`run_deepwalk.py` ירוץ קודם (משתמש ב-embeddings השמורים).

**שחזוריות:** כל התהליכים האקראיים קבועי seed, ואימון ה-DeepWalk רץ על thread יחיד כדי שהתוצאות יהיו זהות בין הרצות. לשחזור מלא מומלץ להריץ עם משתנה הסביבה `PYTHONHASHSEED=0`.

## מקורות

1. Rozemberczki, B., Allen, C., & Sarkar, R. (2019). Multi-scale Attributed Node Embedding (MUSAE).
2. McPherson, M., Smith-Lovin, L., & Cook, J. M. (2001). Birds of a Feather: Homophily in Social Networks. Annual Review of Sociology.
3. Blondel, V. D., et al. (2008). Fast unfolding of communities in large networks.
4. Perozzi, B., Al-Rfou, R., & Skiena, S. (2014). DeepWalk: Online Learning of Social Representations. KDD.
5. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR.
6. Liben-Nowell, D., & Kleinberg, J. (2003). The Link Prediction Problem for Social Networks. CIKM.
7. Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). Power-law distributions in empirical data. SIAM Review.
8. Broido, A. D., & Clauset, A. (2019). Scale-free networks are rare. Nature Communications.
9. Page, L., et al. (1999). The PageRank Citation Ranking.
10. Hamilton, W., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. NeurIPS.
11. מצגות ותרגולי הקורס.
