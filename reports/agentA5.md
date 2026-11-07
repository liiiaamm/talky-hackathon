# A·5 — Ledger y reconstrucción de saldos

Se recuperó la investigación original detenida por el límite de Claude. Rama existente `agentA/5-ledger`, base `bf9a167`. No se ha enviado nada a la plataforma ni publicado cambios.

## Resultado y decisión

Corregido el ledger opcional en `src/bankrec/pipeline.py`: conserva documentos cerrados, reabre devoluciones, ignora duplicados y aplica los importes contractuales de las excepciones que cancelan totalmente la partida. Reproduce todos los eventos aceptados en orden `(booking_date, bank_txn_id)` cada vez que llega una propuesta. El estado privado es `ctx._ledger = {'opening': DataFrame, 'events': [DataFrame]}`. `BANKREC_LEDGER` sigue desactivado por defecto.

Esto corrige el estado de los eventos **aceptados**, pero no convierte el pipeline en un árbitro cronológico: un resolver tardío sigue viendo el saldo final tras las propuestas de resolvers anteriores. No se recomienda activar el ledger en FINAL sin comprobar el pipeline completo en ambas ventanas.

## Evidencia de reutilización

Se analizaron las salidas completas existentes, con comparación de pares e importes contra golden (tolerancia oficial). No basta con prohibir que un item aparezca dos veces.

| Medida | 2025-H1 | 2024-H2 |
|---|---:|---:|
| Items repetidos sin PARTIAL/SHORT_DISPUTE/RETURNED/DUPLICATE_PAYMENT en sus filas | 1.540 | 1.453 |
| De ellos, todas las asignaciones son correctas según golden | 615 | 566 |
| De ellos, al menos una asignación es incorrecta | 925 | 887 |
| Filas incorrectas en ese último grupo | 950 | 914 |
| Items con inversión entre prioridad de módulo y fecha bancaria | 2.350 | 2.106 |
| Pares de eventos con esa inversión | 2.535 | 2.198 |

En H1 hay 9.739 items presentes en más de un movimiento, 7.034 de ellos cruzan módulos y 8.199 presentan alguna excepción explicativa. Estas cantidades incluyen casos legítimos y no son una estimación de puntos recuperables.

Ejemplos H1:
- `ARI-2025-8679133`: `BT-2025-0032571` (`dd_batch`) y `BT-2025-0199508` (`ar_party_subset`), sin excepción explicativa.
- `ARI-2025-9851638`: `BT-2025-0184257` (`ar_adjusted`) y `BT-2025-0199707` (`ar_multi`).
- `ARI-2025-6218296`: `dd_batch` reclama `BT-2025-0159425` del 2025-05-27 antes de que `dd_return` procese `BT-2025-0134245` del 2025-05-07. Es una inversión de prioridad, no necesariamente un error del golden.

## Paridad de `_open_at` con LIVE

La reconstrucción actual al 2025-07-01 **no reproduce** `live/open_items_at_start`:

- 238.388 discrepancias mayores de 0,01 entre documentos emitidos antes de LIVE.
- 237.421 tienen saldo reconstruido positivo pero no aparecen en el snapshot oficial.
- 967 sí aparecen en el snapshot y tienen otro saldo.
- 212.732 discrepancias corresponden a documentos anteriores al primer movimiento disponible (2021-07-01); 890 de estas figuran en el snapshot.
- Todos los items oficiales existen en las tablas históricas; no faltan documentos en esta comparación.

Ejemplo inequívoco: `ARI-2021-2035108`, emitida 2021-03-01, total 18.992,51, sin asignación histórica disponible: `_open_at` supone 18.992,51 abiertos, pero LIVE declara 189,93. Por tanto no se puede inferir un saldo inicial exacto partiendo solo del total de cada factura y del golden disponible. Un replay de importes que arregle duplicados/devoluciones tampoco recupera esa información ausente. No se ha sustituido la reconstrucción por una heurística que podría sesgar las validaciones.

## Medición y comprobaciones

| Salida completa de referencia (ledger OFF) | Score | F1 | P | R | TxnAcc | ExcF1 |
|---|---:|---:|---:|---:|---:|---:|
| 2025-H1 | 85,14 | 0,8789 | 0,9920 | 0,7889 | 0,9132 | 0,6902 |
| 2024-H2 | 84,83 | 0,8787 | 0,9914 | 0,7890 | 0,9053 | 0,6868 |

El modelo `categories_to2024` usado en estas referencias incluye 2024-H2 en su entrenamiento: sirve para reproducir el baseline histórico, no como validación temporal limpia de categorías. El agente A4 prepara un modelo cortado al 2024-06-30.

Seis tests de invariantes pasan en `tests/test_ledger.py`: cierre seguido de devolución, devolución de item cerrado antes de la fase, cada tipo de diferencia que cancela completamente seguido de devolución, duplicado sin consumo, propuestas recibidas en orden inverso de fechas y SHORT_DISPUTE con saldo residual. Los cambios no activan el ledger ni cambian el flujo por defecto. No se atribuye una ganancia de Score al ledger corregido sin una ejecución completa que la mida.

## Riesgos para FINAL y para B/C

- Usar `open_items_at_start` oficial en LIVE/FINAL es esencial; TRAIN no dispone de una historia inicial suficiente para reconstruirlo exactamente.
- Ordenar eventos ya aceptados no basta para adjudicar una misma factura entre resolvers de fechas distintas. La alternativa correcta exige propuestas con evidencia, arbitraje por fecha y reintentos de resolvers que perdieron su mejor candidato; no un veto indiscriminado de reutilización.
- Los caches de B/C que dependan de `ctx.items.open_amount` deben invalidarse si se habilita un ledger mutable.
- El replay acumulado consume memoria y tiempo extra cuando el flag se activa; mantenerlo OFF mientras no se valide esa combinación completa.
- Los desempates dentro del mismo día usan `bank_txn_id`, ya que no hay hora bancaria en esta interfaz.

Artefactos reproducibles locales: `out/diagnose_ledger.py`, `out/analyse_reuse.py`, `out/ledger_diagnosis.json`, `out/reuse_diagnosis.json`, `out/reuse_diagnosis_h2.json`, `out/open_at_discrepancies.parquet`. No se suben datos ni salidas al repositorio.
