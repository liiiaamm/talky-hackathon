# Exploración de datos: Close the Books

Scripts en [eda/](eda/) (salidas en `eda/*.out`). Datos en `data/` y caché parquet en `data/parquet/` (se genera con `eda/00_to_parquet.py`).
Entorno: `.venv/` (polars, duckdb, pyarrow, pandas, rapidfuzz, pypdf, openpyxl, scikit-learn, lightgbm).

## 0. Avisos prácticos

- Los tarballs traían **75.182 ficheros basura de macOS (`._*`)**; ya los he borrado. Si alguien descomprime otra vez, tiene que ejecutar `find data -name '._*' -delete`. Con FINAL pasará lo mismo.
- `masters/customers.csv` tiene nombres con comas entre comillas, y DuckDB no autodetecta bien el entrecomillado. Hay que leerlo con pandas o con `quote='"'`.
- `document_number` **no es único entre sociedades**: los cruces deben hacerse siempre por `(entity, document_number)`.
- **TRAIN-lite** trae todas las tablas y el golden completo, pero solo el 26 % de los avisos de remesa (13,4 k de 50,9 k) y el 25 % de los PDFs (33 k de 132 k), los de 2024-07 en adelante. Para estudiar avisos y PDFs hay que usar ese periodo. LIVE los trae completos (6,9 k avisos y 16,7 k PDFs).
- Tamaños: TRAIN tiene 1,49 M movimientos, 3,96 M facturas emitidas, 1,53 M recibidas, 150 k abonos y 5,4 M filas de golden. LIVE tiene 203 k movimientos y 325 k partidas abiertas al inicio.

## 1. Dónde está la puntuación (golden de TRAIN completo, 72 segmentos)

El golden cubre el 100 % de los movimientos. El 61 % de los movimientos son cobros. La mezcla de segmentos es **muy estable año a año**, así que TRAIN es representativo.

| Segmento | % movimientos | % pares | Pares por movimiento (media, máximo) | Cómo se resuelve |
|---|---|---|---|---|
| ar_ref_noise | 21,0 | 12,9 | 2,2 (49) | Cliente + importe (subset-sum) + referencia difusa |
| ap_payment_multi | 11,7 | 11,4 | 3,4 (53) | **payment_runs** |
| ar_1to1 | 9,5 | 2,7 | 1 | Número de factura exacto en el texto (99 %) |
| ap_payment_single | 9,4 | 2,7 | 1 | **payment_runs** |
| ar_same_amount_fifo | 3,3 | 0,9 | 1 | Cliente + importe, la más antigua (FIFO) |
| nonitem_bank_fees | 3,2 | 0 | n/a | Clasificador de texto |
| ar_1toN_text | 3,1 | 4,9 | 5,6 (189) | Varios números de factura en el texto (96 %) |
| ar_short_discount | 2,9 | 0,9 | 1,1 | Fórmula de descuento (POLICIES) |
| ap_direct_debit | 2,8 | 0,8 | 1 | "RECIBO <doc>" o mandato |
| ar_same_amount_month | 2,2 | 0,6 | 1 | Mes de servicio en el texto frente al PDF o la fecha de factura |
| ap_paid_by_holding | 2,2 | 1,6 | 2,5 | payment_runs de ES00 con partidas de la filial |
| ar_instalment | 2,2 | 0,6 | 1 | PDF "N instalments" |
| ar_partial | 1,9 | 0,5 | 1 | Número de factura con importe menor → PARTIAL |
| ar_dd_returned | 1,8 | 0,5 | 1 | Mandato (RUM/MREF) → recibo de la remesa |
| unidentified + lookalike | 2,3 | 0 | n/a | No forzar |
| ar_1toN_credit_note | 1,6 | 4,4 | 9,8 (459) | Subset-sum con abonos (y avisos de remesa) |
| ar_dd_batch | 0,8 | **13,6** | 57 (346) | **direct_debit_batches** |
| ar_card_settlement | 0,7 | 2,8 | 15 | **card_settlements** |
| ar_cheque_deposit | 1,0 | 0,8 | 2,8 | **cheque_deposits** |
| ar_1toN_remittance | 0,6 | **7,4** | 41 (561) | **Avisos de remesa** |
| ar_short_dispute | 0,5 | **13,1** | 91 (912) | **Avisos de remesa** con deducción y motivo |
| ap_fx | 0,6 | **6,3** | 39 (1540) | payment_runs + FX_DIFFERENCE |
| ar_fx | 0,4 | 1,9 | 17 | Divisa distinta + FX_DIFFERENCE |
| … ~50 segmentos más, cada uno < 1 % | | | | |

**Bloques por mecanismo** (porcentaje aproximado de los pares, que es el 50 % de la nota):
- **Cruces deterministas con ficheros: ~45 %.** Remesas de recibos 13,6 + tarjetas 2,8 + cheques 0,8 + todo el lado AP en payment_runs (~28).
- **Avisos de remesa: ~25 %.** short_dispute 13,1 + 1toN_remittance 7,4 + parte de credit_note y n_to_m.
- **Texto + importe en cobros: ~25 %.** ref_noise 12,9 + 1toN_text 4,9 + resto.

## 2. Excepciones y categorías

Filas con excepción: FX_DIFFERENCE 113 k · SHORT_DISCOUNT 100 k · PARTIAL 92 k · UNIDENTIFIED 35 k · SHORT_WITHHOLDING 34 k · RETURNED 30 k · SHORT_DISPUTE 20 k · DUPLICATE_PAYMENT 4,9 k · OVERPAYMENT 2,8 k · SHORT_BANK_CHARGES 1,1 k.
**Las 10 clases pesan igual en el macro-F1** (20 % de la nota), así que las clases raras valen lo mismo que FX.

Categorías: BANK_FEES 70 k · UNIDENTIFIED 35 k · CASH_POOL 12,6 k · PAYROLL 3,9 k · INTERCOMPANY 3,2 k · LEASING 2,8 k · BANK_REVERSAL 2,7 k · CUSTOMER_REFUND 2,6 k · CUSTOMER_ADVANCE 2,0 k · INTERNAL_TRANSFER 1,9 k · FX_DEAL 1,6 k · resto < 1 k.
Los textos de las categorías son muy plantillados ("NOMINA 12 24", "LIQUIDACION INTERESES 1T24", "TRASPASO A CTA", "CASH CONCENTRATION", "ANULACION OPERACION INDEBIDA", "VAT RETURN", "FX DEAL", "CUOTA LEASING"…). Con reglas o un clasificador ligero sobre `bank_code` + texto + signo deberíamos acercarnos al 100 %.

40,9 k movimientos combinan excepciones distintas en sus filas, y 75,9 k incluyen abonos.

## 3. Enlaces deterministas (verificados contra el golden)

| Fichero | Clave en el movimiento | Resultado |
|---|---|---|
| `direct_debit_batches` (713.554 filas, igual que los pares de ar_dd_batch) | `batch_id` en el texto ("EINZUG 02241015509 26 POSTEN", "ACH ORIGINATION BATCH …") e importe igual a `batch_total` en la misma cuenta | **12.505 de 12.505 remesas.** El `batch_id` también aparece en la comisión de la remesa ("COMISION REMESA 0124… 129 RECIBOS" → BANK_FEES). Cada fila lleva `remittance_info` con el número de factura. |
| `card_settlements` (147.622 filas, igual que los pares de card) | `MID` en el texto + fecha de liquidación (hasta 3 días), con importe del banco igual a la **suma neta**; las partidas se liquidan por el bruto | **10.051 de 10.051** |
| `cheque_deposits` (42.284 filas, igual que los pares de cheque) | `deposit_id` en el texto (LOCKBOX LBX…, remesas de cheques "RCH…") | **15.231 de 15.231** |
| `payment_runs` (1,42 M filas) | El id de remesa `PR\d{12}` aparece en el 100 % de los movimientos AP (en N43 llega truncado: "PR010024051008-0"). Los `transfers` tienen un adeudo por beneficiario (e2e = `run_id-NNNN`); los `aggregated` y `confirming` (`CF\d{12}`, "REMESA CF… N PROVEEDORES") tienen un único adeudo. | **Todos los pares AP de golden están en payment_runs**, salvo ap_direct_debit (99 %) y ap_supplier_refund |

En el lado AP solo queda decidir las excepciones: `document_amount` < total → PARTIAL; divisa del movimiento ≠ divisa de la factura → FX_DIFFERENCE; "RETURN MD07 … PR…" → RETURNED; pago repetido → DUPLICATE_PAYMENT; abono AP en la remesa → fila del APC en positivo.

## 4. Cobros de clientes: qué señales funcionan (2024–2025-06)

| Segmento | Número exacto | Número parcial | Nombre en texto | IBAN = maestro (camt) | Suma = importe | 1 factura = importe |
|---|---|---|---|---|---|---|
| ar_ref_noise | 2 % | 23 % | 96 % | 99,8 % | 100 % | 74 % |
| ar_1to1 | 99 % | 99 % | 96 % | 99,9 % | 100 % | 100 % |
| ar_same_amount_fifo / month | 0 % | 0 % | 96 % | 99,8 % | 99,5 % | 99,5 % |
| ar_1toN_text | 96 % | 98 % | 97 % | 100 % | 100 % | n/a |
| ar_short_discount | 34 % | 47 % | 99 % | 99 % | 0 % (fórmula) | n/a |
| ar_1toN_remittance | 0 % | 1 % | 96 % | 100 % | 40 % | n/a |
| ar_short_dispute | 6 % | 17 % | 94 % | 99 % | 26 % | n/a |
| ar_third_party_parent | 56 % | 58 % | **37 %** | **46 %** | 85 % | n/a |
| ar_third_party_confirming | 41 % | 62 % | 75 % | n/a | 100 % | 100 % |
| ar_new_iban | 0 % | 0 % | 86 % | 100 % | 100 % | 79 % |

**Conclusiones**
1. **Identificar al pagador es fácil y es la primera clave**: el nombre (los 10 primeros caracteres normalizados del nombre o el `trade_name`) aparece en ~95 % de los textos, y en camt el `counterparty_account` coincide con el IBAN del maestro en el 99,8 %. Los fallos son terceros (matriz, confirming, agencias), que hay que resolver por grupo, `parent_customer_id`, "ORD. <cliente>" o "EXPTE …".
2. **El número de factura no es la clave principal**: en ref_noise solo aparece literal en el 2 %. La clave es **cliente + importe**, que cuadra al céntimo en el 100 % de los casos sin excepción. En el 74 % de ref_noise hay una sola factura con el importe exacto; el resto necesita subset-sum sobre las abiertas del cliente, desempatando con dígitos parciales de la referencia y con FIFO.
3. Cuando el importe no cuadra, son las excepciones de POLICIES: el descuento se calcula con la fórmula, la retención con un 1,25/1,5/10 %, el FX con `orig_amount`/`orig_currency` o `/OCMT/…/EXCH/`, y los parciales con el PDF de plazos, la retención de garantía o el anticipo.
4. **Unidentified + lookalike suman 2,3 % de los movimientos**: si no hay cliente o el texto no cuadra, se marca UNIDENTIFIED. Los lookalike tienen el importe exacto de una factura ajena, así que exigir también la coincidencia de cliente lo evita.

## 5. Avisos de remesa (el ~25 % de pares que falta)

- Índice: `file, entity, payer_name, received_date, format` (csv 19 k, pdf 17 k, txt 13 k, xlsx 2 k en TRAIN; pueden llegar días después del cobro).
- CSV: cabecera `payment_ref;<ref>;payment_date;…;currency` + filas `document;doc_date;gross;discount_or_deduction;reason;paid;your_order` + TOTAL.
- TXT: tipo correo, "Subject: Payment advice <ref>", "payment of 73,170.54 USD on 07/01/2024 covering:" + líneas `doc fecha importe [-deducción MOTIVO]`.
- PDF: "REMITTANCE ADVICE / Payment ref. / Amount / Document Date Gross Deduction Paid…". `pypdf` extrae el texto limpio.
- **Enlace con el banco**: la `payment_ref` del aviso aparece en el texto del movimiento (p. ej. "LOMASECA A 240430341864", "PAY-2024-…", "REMITTANCE 2533101263"), más pagador, importe total y fecha.
- Los documentos del aviso pueden venir **sin prefijo** y con el número de pedido del cliente (`your_order`). Un gran cliente cambia de layout en 2025-09.

## 6. PDFs de documentos (cláusulas que no están en ningún CSV)

Índice: invoice 76 k · supplier_invoice 28 k · credit_note 15 k · supplier_credit_note 10 k · proforma 2,2 k · bank_account_notice 0,6 k.
Son frases de plantilla (ES/EN/DE/FR/PT), así que basta con expresiones regulares:
- "This invoice cancels and replaces invoice X dated …" / "Esta factura anula y sustituye a la factura X …" → ar_replaced_invoice
- "Payment terms: 3 instalments at 30/60/90 days (a / b / c)" / "Forma de pago: 3 vencimientos …" → plazos (PARTIAL salvo el último)
- "Retention 5% (…) withheld until final acceptance (fecha)" / "Retención en garantía del 5 % …" → retención y su liberación
- "Less advance received against proforma PF…: X. Net amount due Y." → ar_advance_deduction (PARTIAL)
- "Maintenance and service fee for the month of June 2024." / "Cuota de mantenimiento … correspondiente a …" → same_amount_month
- "The receivable … has been assigned to <factor/confirming>" → pagos de terceros
- Abonos: "to be deducted from your next payment" / "will be refunded by bank transfer"; rappel "to be offset" o "will be paid to your bank account"
- "FACTURA RECTIFICATIVA / ABONO", "CANCELLING OF INVOICE Nº …"
- Condiciones de pago con descuento: "2 % 10 días, neto 30", "1 % 15 días, neto 60"

## 7. Drift: LIVE frente a TRAIN

- El volumen y el signo son idénticos (203 k frente a 200 k en el semestre anterior; 61 % cobros).
- **Banco del Moncayo**: N43 hasta 2025-09 y camt053 desde 2025-10 (confirmado). Ya podemos probarlo en LIVE.
- **ZA01-UMVC-ZAR ya aparece en LIVE** con 2 movimientos en 2025-12: anticipos de proformas (PF25-0100001/2) cobrados antes de la adquisición → `CUSTOMER_ADVANCE`. En FINAL, las facturas de ZA01 ("KWI…") dirán "a deducir anticipo" y su cobro será `PARTIAL`.
- No hay `bank_code` nuevos en LIVE. Hay 1.515 clientes nuevos (con facturas en LIVE que no aparecen en TRAIN), así que **no hay que depender solo de un histórico por cliente**.
- Partidas abiertas al inicio de LIVE: 229,5 k ARI (13 % parcialmente abiertas), 68 k API, 25 k ARC y 2 k APC. Las hay desde 2021, así que hay que usar `open_amount`, no `total_amount`.

## 8. Implicaciones para la arquitectura (orden de ejecución)

1. **Deterministas primero** (alta precisión, ~45 % de pares): DD batches → tarjetas → cheques → payment_runs (incluyendo confirming, aggregated, holding, factored, FX y devoluciones).
2. **Avisos de remesa** (~25 % de pares): parser de 4 formatos → enlace por payment_ref + pagador + importe → filas con deducciones (SHORT_DISPUTE con importe pagado, abonos en positivo).
3. **Cobros por texto**: identificar el cliente (IBAN > nombre > terceros) → candidatas abiertas → número exacto/parcial → importe exacto → subset-sum → fórmulas de excepción → FIFO/mes. Si no hay nada, UNIDENTIFIED.
4. **No-partida**: reglas o clasificador para las 20 categorías (y BANK_FEES de remesas y transferencias internacionales).
5. **Ledger cronológico**: descontar lo liquidado para no reutilizar partidas (DUPLICATE_PAYMENT es justo una partida ya liquidada que vuelve a pagarse).
