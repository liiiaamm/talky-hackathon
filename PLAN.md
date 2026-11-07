# Plan de ataque: Claude Code Hackathon "Close the Books"

**Sábado 7/11/2026 · La Nave, Madrid · 09:00–21:00 · Talky × Claude Community × UCppM (con apoyo de Anthropic)**
Web: https://hackathon.usetalky.com · ~70 personas · equipos de 1 a 8 (somos 5)

---

## 1. El reto en 30 segundos

Conciliación bancaria de **Grupo Valtermia** (ficticio): 17 sociedades, 12 divisas, 13 bancos, ~50 cuentas.
Para **cada movimiento bancario** hay que decir:
- qué partidas abiertas liquida (`ARI-`/`ARC-`/`API-`/`APC-`) y **por qué importe**, en la divisa de la partida, o
- si no liquida ninguna, qué es (`category`: BANK_FEES, CASH_POOL, PAYROLL…),
- y marcar la **excepción** (lista cerrada) que explica la diferencia.

```
bank_txn_id,item_id,amount,category,exception
BT-2025-0001234,ARI-2025-0456789,1210.00,,
BT-2025-0001240,,,BANK_FEES,
BT-2025-0001300,ARI-2025-0400001,980.00,,SHORT_DISCOUNT
BT-2025-0001301,,,UNIDENTIFIED,UNIDENTIFIED
```

**Score = 100 × (0,5 · Allocation F1 + 0,3 · Transaction accuracy + 0,2 · Exception macro-F1)**
- Tolerancia de importe: |enviado − golden| ≤ max(0,01; 0,5 % del golden).
- Referencias oficiales: Naive = **19**, heurística bien trabajada = **61**, perfecta = 100. **Objetivo: >80.**

## 2. Fases y calendario

| Fase | Periodo | Movimientos | Etiquetas | Uso |
|---|---|---|---|---|
| TRAIN | 2021-07 → 2025-06 | 1,49 M | **Sí** (golden + `segment` + `difficulty`) | Entrenar y autoevaluar en local |
| LIVE | 2025-07 → 2025-12 | 203 k | No | Clasificación pública. Cuenta el mejor envío |
| FINAL | 2026-01 → 2026-06 | 220 k | No | **Decide el ganador.** UN solo envío, oculto hasta las 20:30 |

| Hora | Hito |
|---|---|
| 09:00 | Acreditación y café |
| 09:30 | Presentación del reto |
| 10:00 | Arranca el hackathon y abre LIVE |
| 14:00 | Comida |
| **18:00** | **Se publican los datos FINAL** |
| **19:30** | **Cierre de envíos** (90 min para FINAL) |
| **19:45** | **Entrega del proyecto** (vídeo ≤5 min 720p, repo GitHub, descripción) y presentaciones |
| 20:30 | Revelación y premios |

## 3. Lo más importante: los datos YA están publicados

Todo excepto FINAL se puede descargar hoy (las reglas no lo prohíben, pero conviene confirmarlo con la organización al llegar):

| Archivo | Tamaño |
|---|---|
| https://hackathon.usetalky.com/data/v1/bankrec-docs-and-masters.zip | 10 MB (docs, maestros, 1 mes de muestra con golden, starter y **el evaluador oficial**) |
| https://hackathon.usetalky.com/data/v1/bankrec-train-lite.tar.gz | 460 MB (recomendado: todas las tablas y el golden; extractos en bruto y PDFs solo desde 2024-07) |
| https://hackathon.usetalky.com/data/v1/bankrec-train.tar.gz | 657 MB (completo, 131 k PDFs) |
| https://hackathon.usetalky.com/data/v1/bankrec-live.tar.gz | 87 MB |
| https://hackathon.usetalky.com/examples.json | Un ejemplo resuelto por segmento (36 segmentos) |

`starter/bankrec_score.py` es **exactamente el evaluador del servidor**, así que podemos iterar sobre TRAIN sin gastar envíos.
Esta noche podemos llegar con un pipeline de ~60+ y dedicar el día a la cola larga. **Es nuestra mayor ventaja.**

## 4. Dónde están los puntos (análisis de la muestra 2025-06 con golden)

**Por movimientos** (pesa en Transaction accuracy, el 30 %):
ar_ref_noise 25 % · ar_1to1 11 % · ap_payment_multi 9 % · ap_payment_single 5 % · nonitem_bank_fees 5 % ·
ar_1toN_text 4 % · same_amount_fifo 3 % · short_discount 2,6 % · dd_returned 2,4 % · 1toN_credit_note 2,3 % ·
instalment 2,2 % · same_amount_month 2,2 % · partial 2 % · unidentified 1,8 % + ~30 segmentos más.

**Por pares (movimiento, partida)** (pesa en Allocation F1, el 50 %): aquí manda el 1:N.
| Segmento | % de pares |
|---|---|
| ar_short_dispute (aviso de remesa con deducción) | **23,3 %** |
| ar_ref_noise | 15,5 % |
| ar_dd_batch (remesas SEPA de recibos) | **14,5 %** |
| ap_payment_multi (nuestras remesas de pago) | 8,5 % |
| ar_1toN_credit_note | 6,9 % |
| ar_1toN_text | 6,3 % |
| ar_card_settlement | 3,2 % |
| ar_1toN_remittance | 3,0 % |

**Conclusiones tácticas**
1. **Joins deterministas = puntos baratos**: `direct_debit_batches/`, `card_settlements/`, `cheque_deposits/` y `payment_runs/` dan la lista exacta de partidas. Suman ~30 % de los pares.
2. **Parsear avisos de remesa** (csv/xlsx/pdf/txt), que llegan días después del cobro, vale ~33 % de los pares (dispute + credit notes + remittance).
3. **El macro-F1 de excepciones promedia por clase.** Las 10 clases pesan lo mismo, así que cada una vale unos **2 puntos** de score. Las raras (SHORT_WITHHOLDING, SHORT_BANK_CHARGES, OVERPAYMENT, DUPLICATE_PAYMENT) son puntos fáciles porque sus reglas están escritas exactamente en POLICIES.md.
4. **No dejar ningún movimiento fuera**: los que faltan cuentan como fallo. Ante la duda: UNIDENTIFIED en cobros y la mejor categoría posible en pagos.
5. **No forzar emparejamientos**: hay `unidentified_lookalike` (cobros con el importe exacto de una factura que no tiene nada que ver).

## 5. Reglas del golden que hay que codificar al pie de la letra (POLICIES.md)

- **Importe**: normal, SHORT_DISCOUNT, SHORT_BANK_CHARGES, SHORT_WITHHOLDING, FX_DIFFERENCE y OVERPAYMENT → **importe abierto completo**. PARTIAL y SHORT_DISPUTE → lo realmente aplicado. Los abonos van siempre en **positivo** y sin excepción.
- **SHORT_DISCOUNT**: hay descuento en `payment_terms`, booking_date ≤ fecha de factura + discount_days **+ 3 días de gracia**, y cash = total − pct × total (redondeado). Si se paga fuera de plazo → **PARTIAL**.
- **SHORT_BANK_CHARGES**: transferencia de fuera de SEPA con merma ≤ **60 € equivalentes** y ≤ **2 %**. Por encima → PARTIAL.
- **SHORT_WITHHOLDING**: clientes de MX/BR/MA con retención del 1,25 %, 1,5 % o 10 % del total.
- **FX_DIFFERENCE**: divisa del banco ≠ divisa de la factura. **Sustituye** a vacío, DISCOUNT, WITHHOLDING y BANK_CHARGES en filas de factura; PARTIAL y DISPUTE se mantienen; los abonos van vacíos. Spread 0,1–1,5 % (tolerancia FX 3 %). Tipos BCE en `fx_rates.csv` (en fin de semana se usa el último disponible).
- **PARTIAL**: queda saldo pendiente. Incluye cada plazo salvo el último, el primer pago de una factura con retención de garantía, una factura con "a deducir anticipo" y un descuento tomado fuera de plazo.
- **Sin excepción**: último plazo, liberación de retención, resto de un parcial, liquidación de una disputa, recibos re-presentados y abonos compensados.
- **RETURNED**: devolución de recibo SEPA (R-transaction), cheque devuelto, chargeback o transferencia rechazada. Lleva una fila por cada partida de la operación devuelta.
- **DUPLICATE_PAYMENT**: la partida ya estaba liquidada y la excepción va en el segundo pago. La devolución posterior es CUSTOMER_REFUND o SUPPLIER_REFUND.
- **Quién paga no es siempre el cliente**: matriz (`parent_customer_id`/`group_id`), confirming ("CONFIRMING … ORD. <cliente>"), agencias de cobro ("EXPTE …"), factoring. ES00 paga facturas de ES01, ES02 y PT01 (payment factory).
- **Mismo importe con varias abiertas**: FIFO por fecha de factura. Si el texto nombra un mes ("MANT. MAYO 25"), se liquida la factura de ese mes de servicio, que se emite a principios del mes siguiente.
- **"Anula y sustituye a X"** (solo en el PDF): un pago que cita X liquida la factura de reemplazo. X y su abono nunca se liquidan por banco.
- **Referencias ruidosas**: sin prefijo ni ceros a la izquierda, dígitos transpuestos, truncadas a 16/35/38 caracteres, PO del cliente, número antiguo o sin referencia (solo nombre o IBAN).
- **Categorías**: comisión internacional = línea aparte el mismo día. Las liquidaciones de tarjeta **no** son comisión (liquidan pedidos en bruto). CASH_POOL va en ambos lados. Error del banco y su anulación → BANK_REVERSAL en las dos líneas. Pago de una proforma → CUSTOMER_ADVANCE.

**Drift (se probará en LIVE y FINAL)**: Banco del Moncayo pasa de N43 a camt.053 el 2025-10-01; un gran cliente cambia el layout de sus avisos en 2025-09; nueva filial **ZA01 (ZAR, facturas "KWI…")** desde 2026-01-01, renombrada el 2026-04-01, con `final/masters_final/`; IT01 cambia de cuenta de cobro el 2026-02-01; en 2026 se paga más tarde.

## 6. Arquitectura propuesta

```
loaders (polars/duckdb → parquet cache)
   │
   ├─ 1. Ledger de partidas abiertas (open_items_at_start + facturas y abonos nuevos), saldo vivo y orden cronológico
   ├─ 2. Resolutores deterministas (alta precisión, se ejecutan primero):
   │      DD batches · card settlements · cheque deposits · payment runs (end_to_end_id / run) · devoluciones R
   ├─ 3. Avisos de remesa: parser csv/xlsx/txt/pdf → (pagador, lista de documentos, deducciones, motivo)
   ├─ 4. Identificación de contraparte: IBAN → nombre (fuzzy, trade_name, nombre truncado) → grupo/matriz/confirming
   ├─ 5. Generación de candidatos: referencias fuzzy (normalizar, quitar prefijo/ceros, transposiciones, prefijos truncados, PO)
   │      + importe (exacto / tolerancia / descuento / retención / FX) + subset-sum sobre las abiertas del cliente
   ├─ 6. Reglas de excepción (POLICIES §3) + cláusulas de PDF (plazos, retención, anticipo, anula y sustituye, mes, IBAN)
   ├─ 7. Clasificador de no-partida (bank_code, subcode, texto, signo, contraparte → category), entrenado con TRAIN
   └─ 8. Writer + validador local (mismas reglas que el servidor) + bankrec_score.py con desglose por segmento
```

Principios: un único `run.py --phase {train|live|final}` reproducible y que tarde **< 15 min**, cada módulo con su propio score por segmento, y precisión antes que recall (un par erróneo penaliza tanto la precisión como la accuracy).

Stack: Python 3.11+, polars o duckdb, rapidfuzz, pdfplumber/pypdf (o `pdftotext`), openpyxl y Claude Code en cada portátil. Usaremos LLMs solo para la cola larga (PDFs y textos raros) y con caché, porque 220 k líneas en 90 minutos no se pueden procesar con un LLM línea a línea.

## 7. Reparto de roles (5 personas)

| # | Rol | Responsable de |
|---|---|---|
| **P1** | Core e integración (capitán) | Loaders, ledger de saldos, orquestación, writer y validador, scoring por segmento, envíos LIVE/FINAL, merge de ramas |
| **P2** | Cobros AR por texto | Identificación de pagador, ref_noise, 1:1, 1:N por texto, FIFO o mes, terceros (matriz, confirming, agencia), new_iban, unidentified/lookalike |
| **P3** | Joins estructurados | DD batches y devoluciones/re-presentaciones, tarjetas, cheques, payment runs (AP single, multi, confirming, factored, direct debit), abonos AP compensados |
| **P4** | Remesas y excepciones | Parser de avisos de remesa (dispute, credit notes), SHORT_*, FX, OVERPAYMENT, DUPLICATE, PARTIAL/plazos/retención/anticipo, cláusulas de PDF |
| **P5** | No-partida, drift y narrativa | Clasificador de categorías (fees, cash pool, reversal…), análisis de errores, preparación del drift de FINAL (ZA01, masters_final), **vídeo, repo y descripción** |

Cada persona trabaja en su rama/módulo con una interfaz común: `resolve(txns, ctx) -> rows + set de txn resueltas`. P1 encadena los módulos por prioridad.

## 8. Plan horario

### Hoy (viernes): preparación
- [ ] Todos: descargar docs, train-lite y live; leer README, POLICIES y FORMAT; revisar `examples.json` de sus segmentos.
- [ ] P1: repo GitHub (privado por ahora), estructura, loaders a parquet, `run.py` y evaluador por segmento sobre TRAIN.
- [ ] Todos: comprobar que starter + scorer funcionan en su portátil (necesita unos 8 GB de RAM para TRAIN completo; usar muestras por meses).
- [ ] Cada uno: prototipo de su módulo con la muestra de un mes. **Meta: pipeline > 50 en TRAIN antes de dormir.**
- [ ] Preparar el script `submit.py` (viene en la web, usa la variable `HK_TOKEN`).

### Sábado
| Hora | Qué |
|---|---|
| 09:00–10:00 | Registrar el equipo (lo hace una persona; el código del evento se da allí), **guardar el token** (solo se muestra una vez) y esperar la aprobación. Confirmar con la organización que el trabajo previo es válido. |
| 10:00–10:30 | **Primer envío LIVE** con el pipeline actual para validar el formato de extremo a extremo. |
| 10:30–14:00 | Sprint 1: módulos grandes (joins, remesas, ref_noise, categorías). Envío LIVE cada vez que la mejora en TRAIN sea mayor de 2 puntos. |
| 14:00–14:45 | Comida y merge. Repasar la tabla de errores por segmento y repriorizar. |
| 14:45–17:00 | Sprint 2: cola larga (clases de excepción raras, PDFs, drift, terceros). |
| **17:00** | **Feature freeze.** Ensayo de FINAL: ejecutar en LIVE desde cero y cronometrar. P5 graba el vídeo y redacta la descripción. |
| **18:00** | Descargar FINAL (y `masters_final`), ejecutar el pipeline y hacer sanity checks: 100 % de movimientos cubiertos, distribución de categorías y excepciones parecida a LIVE, ZA01 tratada, validador local OK. |
| **~18:45** | **Envío FINAL único.** No esperar al último minuto: el envío falla rápido si tiene errores de formato, y en empate gana el más temprano. |
| 19:00–19:45 | Entregar el proyecto (vídeo, repo público o con acceso para el jurado, descripción) **antes de las 19:45**. |

## 9. Reglas operativas y riesgos

- **El envío FINAL es único**: probar el formato en LIVE antes. Un fichero con errores de formato no gasta el envío, pero uno válido y malo sí.
- Pedir la URL de subida **ya consume un hueco** de envío aunque no se suba nada, y hay un intervalo mínimo entre envíos y un máximo diario por fase. Toda la sala comparte IP, así que hay que consultar el resultado como mucho cada 5 s.
- Límites del fichero: ≤ 60 MB sin comprimir y ≤ 3 M filas (LIVE tiene unas 750 k asignaciones; mandar `.csv.gz`).
- El token no se comparte ni se sube al repo. Vive en la variable de entorno `HK_TOKEN`.
- No intentar obtener el golden de LIVE o FINAL ni atacar la plataforma (descalificación).
- Usar train-lite si la RAM es justa y trabajar con ventanas de meses.

## 10. Entrega del proyecto (lo que ve el jurado)

- Vídeo ≤ 5 min a 720p (subido o enlace a YouTube oculto, Loom o Drive): lo **clave** de la solución, el enfoque y los resultados.
- Repo `https://github.com/<owner>/<repo>` accesible para el jurado.
- Descripción con ideas clave, stack y modelos, siguientes pasos y miembros.
- Narrativa ganadora: "Explicamos por qué casa cada cosa y marcamos lo que no sabemos": trazabilidad por fila (qué regla o evidencia lo decidió), score por segmento y cómo Claude Code nos aceleró.
