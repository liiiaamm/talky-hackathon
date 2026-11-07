# A·4 — Categorías y validación temporal

Investigación recuperada de `agentA/4-categories`, base `bf9a167`. Se han medido las dos ventanas y se conserva `STRONG = 0.97`: bajar el umbral no recupera ningún movimiento del pipeline histórico.

## Corrección metodológica y cambio implementado

`categories_to2024.joblib` se entrenó hasta 2024-12-31. Es adecuado como baseline histórico de 2025-H1, pero **2024-H2 forma parte de su entrenamiento**. No se puede presentar ese segundo resultado como validación temporal fuera de muestra de categorías.

Se añadió `python -m bankrec.categories --train --until 2024-06-30`. `--until` y `--all` son mutuamente excluyentes; el comportamiento de entrenamiento existente no cambia si no se indica el nuevo flag. El entrenamiento guarda junto al modelo un JSON con fecha de corte, tamaño y semilla de muestreo. No se modificó la inferencia ni se sobrescribieron modelos compartidos.

Modelo privado validado y congelado:

- `out/categories_to2024h1_frozen.joblib`.
- SHA-256: `bf8c0c172774684477b29fb3dedc4dad951e4cd8332758f127cbdf66c6d47da2`.
- 504.807 líneas de entrenamiento, última fecha 2024-06-30.
- Primera ejecución completada: 248,02 s de proceso, pico RSS 7.036.484 KiB, bajo candado global.
- Hay validación completa baseline/candidato con este mismo modelo coordinada por el integrador; sus métricas son independientes de las de clasificación que siguen.

## Matrices de confusión y umbrales

Predicción de todas las líneas bancarias, en bloques de 4.096, usando exactamente las features de producción. `ITEM` significa que la línea liquida documentos; las clases restantes son categorías. Cada matriz completa por `(segment, truth, pred)` se guarda como CSV en `out/confusion_<modelo>_<ventana>.csv`.

| Modelo / ventana | Líneas | Exactitud clasificador | Claims ≥0,97 | Claims ≥0,95 | Claims ≥0,90 | Claims erróneos en esos tres umbrales |
|---|---:|---:|---:|---:|---:|---:|
| Hasta dic-2024 / 2025-H1 | 199.987 | 99,3535 % | 14.237 | 14.284 | 14.344 | 0 / 0 / 0 |
| Hasta dic-2024 / 2024-H2 (incluida en train) | 196.970 | 99,3654 % | 13.950 | 14.005 | 14.065 | 0 / 0 / 0 |
| Hasta jun-2024 / 2025-H1 | 199.987 | 99,4185 % | 14.215 | 14.321 | 14.374 | 0 / 0 / 0 |
| Hasta jun-2024 / 2024-H2 | 196.970 | 99,4319 % | 13.938 | 14.019 | 14.091 | 0 / 0 / 0 |

Son claims sobre todas las líneas; el pipeline solo aplica el clasificador a las que siguen pendientes. La exactitud del clasificador no es el Score de conciliación.

Errores dominantes del modelo limpio:

| Confusión / segmento | 2025-H1 | 2024-H2 |
|---|---:|---:|
| UNIDENTIFIED → ITEM, `unidentified` | 609 | 584 |
| UNIDENTIFIED → ITEM, `unidentified_lookalike` | 278 | 290 |
| ITEM → UNIDENTIFIED, `ar_ref_noise` | 184 | 151 |
| ITEM → UNIDENTIFIED, `ar_same_amount_fifo` | 22 | 26 |
| ITEM → UNIDENTIFIED, `ar_partial` | 21 | 17 |
| SUPPLIER_REFUND → ITEM, `ap_duplicate_refund` | 13 | 12 |

La principal pérdida es decidir ITEM frente a UNIDENTIFIED, no confundir BANK_FEES/CASH_POOL/FX_DEAL/INTERNAL_TRANSFER. Un segmento internacional de comisiones sigue teniendo categoría BANK_FEES: no es una clase adicional. El modelo limpio presenta dos confusiones INTEREST_EXPENSE → INTEREST_INCOME en H1 y cuatro en H2; no afectan a los claims tempranos de alta confianza.

## ¿Mejora bajar el umbral temprano?

Se reprodujo la sustitución de los nuevos claims en las salidas completas del pipeline histórico, respetando los ocho módulos deterministas anteriores a categorías. Con el modelo hasta diciembre:

| Ventana / umbral | Nuevos claims tempranos | Módulo que los resolvía antes | Líneas antes erróneas | Score antes → shadow después |
|---|---:|---|---:|---:|
| H1 / 0,95 | 47 | cat_rest en las 47 | 0 | 85,14 → 85,14 |
| H1 / 0,90 | 107 | cat_rest en las 107 | 0 | 85,14 → 85,14 |
| H2 / 0,95 | 55 | cat_rest en las 55 | 0 | 84,83 → 84,83 |
| H2 / 0,90 | 115 | cat_rest en las 115 | 0 | 84,83 → 84,83 |

También permanecen idénticos F1, precisión, recall, exactitud por movimiento y macro-F1 de excepciones:

| Ventana | F1 | P | R | TxnAcc | ExcF1 |
|---|---:|---:|---:|---:|---:|
| H1, cualquiera de los tres umbrales en shadow | 0,8789 | 0,9920 | 0,7889 | 0,9132 | 0,6902 |
| H2, cualquiera de los tres umbrales en shadow | 0,8787 | 0,9914 | 0,7890 | 0,9053 | 0,6868 |

Este es un replay de las salidas completas, no cuatro nuevas ejecuciones completas. Como todos los nuevos claims corresponden a líneas que ningún resolver intermedio reclamaba, no se ha identificado beneficio que justifique cambiar el umbral y gastar cuatro ejecuciones. Se conserva 0,97.

## UNIDENTIFIED, riesgo FINAL y Para B/C

- Poner la excepción UNIDENTIFIED a todos los cobros sin resolver introduciría falsos positivos sobre verdaderos ITEM. El golden distingue «no existe partida» de «el resolver no la encontró». No se ha introducido esa regla.
- Una recepción sin matching no es evidencia suficiente para saber que el pagador carece de factura. La coincidencia de importe por sí sola tampoco permite forzar el match, especialmente en lookalikes.
- El diagnóstico identifica transferencias `ar_ref_noise` como principal zona de falsos UNIDENTIFIED; corresponde mejorar evidencia de documento/pagador en B, no reducir precisión del clasificador temprano.
- En FINAL se debe entrenar `--all` con todo TRAIN y usar maestros FINAL en los resolvers. El modelo privado hasta junio existe para validar sin fuga, no sustituye automáticamente al modelo de producción entrenado con más historia.
- La simulación de tokens de entidad nueva y renombres la realiza A9. No se presenta una mera sustitución de `entity=ZA01` como experimento leave-one-entity-out.
- Se comprobó CLI `--help`, compilación de código, entrenamiento con el corte solicitado y predicciones de ambos modelos sobre ambas ventanas. La inferencia por bloques tuvo pico RSS 918.844 KiB con el modelo histórico.

Artefactos locales: `out/evaluate_categories.py`, `out/shadow_threshold.py`, `out/categories_metrics_*.json`, `out/threshold_shadow.json`, modelos y matrices. Estos datos/modelos permanecen fuera del repositorio.
