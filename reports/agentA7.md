# A·7: diagnóstico reproducible de pares perdidos

## Resultado H1 2025

Snapshot completo existente, sin regeneración ni cambios de resolvers: 735.474 pares golden, 584.872 propuestos, 580.205 verdaderos positivos y **155.269 pares perdidos**. Precisión 0,992020; recall 0,788886. Score local 85,14, F1 0,8789, TxnAcc 0,9132, ExcF1 0,6902. El score corresponde al modelo `categories_to2024.joblib`; H2 necesita modelo hasta junio de 2024 para una validación totalmente temporal.

El problema dominante de A es **AP FX agregado**: 43.151 pares perdidos de 45.117 (recall 4,36%, precisión 100%). Confirming pierde 5.155 de 8.133. Ambos acaban como categoría, no como factura. Los 30 movimientos con más pérdidas son todos AP FX agregados: **33.135 pares**, 21,34% del total perdido. El texto contiene `BULK PAYMENT PR… N TRANSFERS`, una referencia estructurada al payment run. A1 recibió estos ejemplos y es el propietario de su corrección; no duplico cambios en `deterministic.py`.

No debe confundirse precisión por movimiento de un segmento LIVE con recall ponderado por pares: unos pocos adeudos grandes pueden perder miles de pares aunque casi todas las líneas pequeñas se resuelvan.

## Causas exhaustivas y dueño propuesto

| owner | cause | missing_pairs |
| --- | --- | --- |
| B/C | category_instead_of_items | 101707 |
| A | category_instead_of_items | 48980 |
| B/C | wrong_items | 1450 |
| C | category_instead_of_items | 1440 |
| B/C | too_few_or_wrong_items | 951 |
| A | wrong_items | 504 |
| B/C | amount_outside_tolerance | 237 |

La suma es exactamente 155.269. `B/C` es asignación de investigación AR pendiente de concretar; el golden no demuestra qué módulo alternativo habría acertado. Sin propuestas standalone no es defendible llamar «módulo equivocado» a todas estas líneas ni prometer que los pares son recuperables. `category_instead_of_items` describe el resultado observado; la causa raíz exige examinar los identificadores y propuestas.

## Mayor pérdida por segmento y módulo

| owner | segment | cause | module | missing_pairs |
| --- | --- | --- | --- | --- |
| A | ap_fx | category_instead_of_items | cat_rest | 43151 |
| B/C | ar_ref_noise | category_instead_of_items | cat_rest | 36363 |
| B/C | ar_1toN_credit_note | category_instead_of_items | cat_rest | 22211 |
| B/C | ar_short_dispute | category_instead_of_items | cat_rest | 17677 |
| B/C | ar_1toN_text | category_instead_of_items | cat_rest | 11207 |
| B/C | ar_n_to_m | category_instead_of_items | cat_rest | 5273 |
| A | ap_confirming | category_instead_of_items | cat_rest | 5155 |
| B/C | ar_fx | category_instead_of_items | cat_rest | 2695 |
| B/C | ar_discount_late | category_instead_of_items | cat_rest | 1796 |
| C | ar_1toN_remittance | category_instead_of_items | cat_rest | 1440 |
| B/C | ar_partial | category_instead_of_items | cat_rest | 1063 |
| B/C | ar_short_discount | category_instead_of_items | cat_rest | 895 |
| B/C | ar_short_withholding | category_instead_of_items | cat_rest | 690 |
| A | ap_direct_debit | category_instead_of_items | cat_rest | 390 |
| B/C | ar_ref_noise | too_few_or_wrong_items | ar_multi | 381 |
| B/C | ar_retention | category_instead_of_items | cat_rest | 295 |
| B/C | ar_third_party_agency | category_instead_of_items | cat_rest | 295 |
| B/C | ar_same_amount_fifo | wrong_items | ar_exact | 274 |
| B/C | ar_replaced_invoice | category_instead_of_items | cat_rest | 273 |
| B/C | ar_overpayment | category_instead_of_items | cat_rest | 251 |

## Pares por tamaño de movimiento

| size_bucket | gold_pairs | true_positive | submitted_pairs | recall | precision |
| --- | --- | --- | --- | --- | --- |
| >100 | 203275 | 143953 | 143954 | 0.7081687369327266 | 0.9999930533364825 |
| 21-100 | 143950 | 115119 | 115183 | 0.7997151788815561 | 0.9994443624493198 |
| 6-20 | 138689 | 92651 | 93327 | 0.6680486556251758 | 0.9927566513441983 |
| 2-5 | 133151 | 116059 | 117786 | 0.8716344601242199 | 0.9853378160392576 |
| 1 | 116409 | 112423 | 114469 | 0.9657586612718948 | 0.9821261651626204 |
|  | 0 | 0 | 153 | NaN | 0.0 |

La fila sin tamaño corresponde a pares propuestos en movimientos que no tienen ningún ítem golden (falsos positivos). NaN significa denominador cero.

## Para B/C

Prioridades por cantidad: referencias ruidosas (36.363 pares categorizados), abonos neteados 1:N (22.211), disputas (17.677), pagos 1:N por texto (11.207) y N:M (5.273). C: 1.440 pares de remesas 1:N sin reclamar. El módulo `ar_group_subset` tiene 65 verdaderos positivos de 241 propuestos (precisión 26,97%); revisar su abstención antes de ampliar alcance.

El diagnóstico exporta todas las filas perdidas, no solo una muestra, a `out/recall_h1/missing_pairs.parquet`. También produce precisión/recall por módulo y segmento en `by_module.csv` y `by_segment.csv`.

## Los 30 movimientos prioritarios

Todos contienen en el texto un payment run PR y número de transferencias; el diagnóstico une texto/ref1/ref2/e2e/beneficiario en `top30.csv` para comprobarlo sin cargar ctx. Todos fueron reclamados por `cat_rest`, omitiendo la totalidad de sus facturas.

| bank_txn_id | segment | missing_pairs | gold_items |
| --- | --- | --- | --- |
| BT-2025-0178444 | ap_fx | 1540 | 1540 |
| BT-2025-0178997 | ap_fx | 1515 | 1515 |
| BT-2025-0043759 | ap_fx | 1333 | 1333 |
| BT-2025-0043625 | ap_fx | 1240 | 1240 |
| BT-2025-0196150 | ap_fx | 1226 | 1226 |
| BT-2025-0196292 | ap_fx | 1199 | 1199 |
| BT-2025-0153299 | ap_fx | 1170 | 1170 |
| BT-2025-0196272 | ap_fx | 1169 | 1169 |
| BT-2025-0196118 | ap_fx | 1139 | 1139 |
| BT-2025-0153151 | ap_fx | 1130 | 1130 |
| BT-2025-0120518 | ap_fx | 1129 | 1129 |
| BT-2025-0105707 | ap_fx | 1122 | 1122 |
| BT-2025-0138418 | ap_fx | 1115 | 1115 |
| BT-2025-0138526 | ap_fx | 1101 | 1101 |
| BT-2025-0153381 | ap_fx | 1076 | 1076 |
| BT-2025-0073546 | ap_fx | 1072 | 1072 |
| BT-2025-0138525 | ap_fx | 1069 | 1069 |
| BT-2025-0088463 | ap_fx | 1039 | 1039 |
| BT-2025-0153262 | ap_fx | 1037 | 1037 |
| BT-2025-0073536 | ap_fx | 1030 | 1030 |
| BT-2025-0138568 | ap_fx | 1018 | 1018 |
| BT-2025-0073528 | ap_fx | 1008 | 1008 |
| BT-2025-0009847 | ap_fx | 989 | 989 |
| BT-2025-0009406 | ap_fx | 984 | 984 |
| BT-2025-0025665 | ap_fx | 967 | 967 |
| BT-2025-0009859 | ap_fx | 962 | 962 |
| BT-2025-0088444 | ap_fx | 961 | 961 |
| BT-2025-0073549 | ap_fx | 948 | 948 |
| BT-2025-0023197 | ap_fx | 924 | 924 |
| BT-2025-0023212 | ap_fx | 923 | 923 |

## Reproducción y límites

```bash
POLARS_MAX_THREADS=4 OPENBLAS_NUM_THREADS=1 /home/liame/claude_hackathon/.venv/bin/python eda/11_pair_recall.py \
  --sub /home/liame/claude_hackathon/out/snap_h1_sub.parquet \
  --gold /home/liame/claude_hackathon/out/snap_h1_gold.parquet --out-dir out/recall_h1
```

El script usa exactamente la tolerancia de importes publicada, `max(0,01; 0,5%)+1e-9`, y atribuye precisión al segmento del movimiento golden, incluyendo falsos positivos. No entrena ni consulta etiquetas fuera del análisis. No cambia la puntuación: antes/después H1 85,14/85,14. H2 se analizó con la foto generada por el integrador: no cambia el resultado del pipeline (84,83 antes/después).

Riesgos FINAL: priorizar reglas estructurales de payment runs, no prefijos de una filial concreta; importes en moneda del documento; nueva filial, monedas y maestros; procesar agregados de más de mil facturas sin recortes arbitrarios. Ningún envío realizado.

## Validación H2 2024 del diagnóstico

716.500 pares golden, 570.239 propuestos, 565.313 verdaderos positivos y **151.187 pares perdidos**. Precisión 0,991362; recall 0,788992. La misma causa estructural se reproduce: AP FX 40.086 pares y confirming 4.970 terminan como categoría. El diagnóstico contabiliza todos los pares perdidos en ambas ventanas.

| Ventana | Score antes/después | F1 | P | R | TxnAcc | ExcF1 |
|---|---|---|---|---|---|---|
| 2025 H1 | 85,14 / 85,14 | 0,8789 | 0,9920 | 0,7889 | 0,9132 | 0,6902 |
| 2024 H2 | 84,83 / 84,83 | 0,8787 | 0,9914 | 0,7890 | 0,9053 | 0,6868 |

**Límite temporal:** ambas fotos usan el modelo de categorías entrenado hasta diciembre de 2024. La segunda ventana coincide parcialmente con su entrenamiento; los números sirven para reproducir y comparar la configuración original, no como estimación temporal independiente del clasificador. A4 prepara un modelo hasta junio de 2024 para la validación limpia.

Causas H2:

| Dueño | Causa | Pares perdidos |
|---|---|---|
| B/C | category_instead_of_items | 101578 |
| A | category_instead_of_items | 45733 |
| B/C | wrong_items | 1548 |
| B/C | too_few_or_wrong_items | 1163 |
| A | wrong_items | 590 |
| C | category_instead_of_items | 318 |
| B/C | amount_outside_tolerance | 252 |
| A | too_few_or_wrong_items | 4 |
| A | amount_outside_tolerance | 1 |

Reproducción H2: mismo comando cambiando `snap_h1_` por `snap_h2_` y `out/recall_h1` por `out/recall_h2`.
