# A·3 — Devoluciones AR y pago de notas de abono

Estado: cambios locales revisables; no envío, commit, push ni PR. Se continuó desde los worktrees de Claude en bf9a167.

## Hallazgos y cambios

1. **La remesa original explícita vence al recibo más reciente.** `dd_returns` escogía el último recibo con igual mandato/importe. `BT-2025-0112253` cita REM 04250411803, pero había otro recibo del mismo importe el propio día de la devolución (04250418805). `BT-2025-0148317` cita DEV01250318103, mientras un recibo regular de mayo tenía el mismo importe. Se prioriza la coincidencia literal de batch_id en el texto, y después la fecha de cobro, manteniendo el fallback previo cuando falta referencia.
2. **ARC pagado al cliente.** Se incorpora `credit_note_payouts` a los deterministas tempranos. Exige adeudo, misma entidad, importe exacto del saldo abierto, fecha de emisión no futura, número completo de ARC explícito y un único candidato. No se decide por nombre/importe solos ni por la palabra RAPPEL. Ejemplos: `BT-2025-0026622` (CM142500033, 19.103,25), `BT-2025-0132952` (CRN25-0100270, 5.660,25).
3. **Colisiones comprobadas.** En los ficheros completos de TRAIN no hay batch_id, settlement_id o deposit_id reutilizados entre cuentas; tampoco mandatos SEPA compartidos entre entidades del maestro. Aun así, la implementación tiene joins finales por id sin cuenta en tarjetas/cheques y se debe conservar la precaución ante nuevos formatos.
4. **Fuente histórica incompleta por fase.** ctx.all_items contiene documentos anteriores, pero ctx.dd/cards/cheques solo carga el fichero estructurado de la fase actual. Una devolución/chargeback de una operación anterior a LIVE/FINAL puede quedar sin vínculo. TRAIN H1/H2 contiene todo el histórico TRAIN y por eso no reproduce ese hueco temporal. No se amplió la carga global sin medir su impacto en RAM.

## Medición

Evaluación dirigida de dos ventanas y overlay sobre snapshots completos originales, sin filas anteriormente reclamadas que desaparezcan. H1 85,14→85,17: 75 líneas corregidas, 32 DD returned y las 43/43 ARC paid out, sin pérdida de líneas antes correctas. H2 84,83→84,86: 70 líneas, 33 DD returned y 37/37 ARC paid out, sin pérdidas. Son resultados de **overlay**, pendientes del pipeline completo de integración coordinada; no se suman a los incrementos de otros agentes.

En el snapshot H1 card_chargeback y cheque_returned ya tienen todos sus pares correctos; en H2 hay un cheque devuelto incorrecto (`BT-2024-0236241`, número 7243834, 1.196,64), que no se arregló con una heurística de último candidato. Re-presentaciones por transferencia: H1 conserva 175 errores; son cobros que dependen de identidad/texto y saldo reabierto, no del nuevo resolver de adeudos.

Cuatro regresiones pasan: batch original frente a recibo posterior del mismo importe, ARC con referencia exacta, rechazo de pago sin referencia y rechazo de ARC ambiguo. Comando: `POLARS_MAX_THREADS=4 /home/liame/claude_hackathon/.venv/bin/python tests/test_ar_deterministic.py`.

Los snapshots originales usan `categories_to2024.joblib` (entrenado hasta diciembre de 2024); por tanto la cifra H2 anterior no es una evaluación temporal limpia del clasificador. La aceptación final se medirá por el coordinador con un modelo entrenado como máximo hasta 2024-06-30 y su baseline correspondiente. Los cambios de este informe son deterministas y no entrenan con golden.

| Ventana / propuesta | Score | F1 pares | P | R | TxnAcc | ExcF1 |
|---|---:|---:|---:|---:|---:|---:|
| H1 baseline snapshot | 85.14 | 0.8789 | 0.992 | 0.7889 | 0.9132 | 0.6902 |
| H1 overlay | 85.17 | 0.879 | 0.9921 | 0.789 | 0.9135 | 0.691 |
| H2 baseline snapshot | 84.83 | 0.8787 | 0.9914 | 0.789 | 0.9053 | 0.6868 |
| H2 overlay | 84.86 | 0.8788 | 0.9914 | 0.7891 | 0.9057 | 0.6877 |

## Riesgos para FINAL

La regla ARC usa ctx.items y depende de que el saldo inicial sea correcto; A5 encontró discrepancias al reconstruirlo. No se amplía a documentos cerrados sin evidencia adicional. Los cobros históricos de otra fase requieren una carga selectiva de ficheros estructurados antiguos para medir con frontera temporal. Los rechazos sin referencia preservan precisión frente a recibos de cuota fija.

## Para B/C

B: los recibos devueltos pagados después por transferencia todavía necesitan identidad del cliente y reapertura del saldo. Ejemplos sin resolver: `BT-2025-0016203`, `BT-2025-0069514`, `BT-2025-0108256`. No se cambió ar_text, matching, ar_exact ni ar_multi. C: sin cambios en remittance. El coordinador debe verificar el pool después de incorporar el nuevo resolver ARC temprano.
