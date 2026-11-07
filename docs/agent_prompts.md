# Prompts para 10 agentes de investigación · ámbito de la Persona A

Cada prompt es el **BLOQUE COMÚN** seguido del **bloque específico** del agente. Pega los dos juntos.
Todos trabajan solo en el terreno de la Persona A: infraestructura e integración, cruces deterministas (clientes:
remesas/tarjetas/cheques; proveedores: todo), categorías / no identificados y FINAL.

---

## BLOQUE COMÚN (va al principio de los 10 prompts)

```
Eres un agente de investigación de la PERSONA A de un equipo en un hackathon de conciliación bancaria ("Close the Books",
Grupo Valtermia, datos sintéticos). Objetivo del equipo: pasar de ~85 a 90+ puntos. Te toca UNA línea de investigación
(al final). Hazla a fondo, con números, y no te salgas del ámbito de A.

ÁMBITO DE LA PERSONA A (lo único que investigas y cambias)
- Infraestructura e integración: src/bankrec/pipeline.py (orden de módulos, ledger, fallback, validador, writer),
  data.py (carga, saldos abiertos), score.py (evaluador local), tools/ (envío, runbook de FINAL).
- Cruces deterministas (src/bankrec/deterministic.py): remesas SEPA de recibos y sus devoluciones/re-presentaciones,
  tarjetas y contracargos, cheques y cheques devueltos, payment runs de proveedores (transfers, aggregated, confirming,
  payment factory ES00, factoring, FX, parciales, duplicados "D"), transferencias devueltas, recibos de proveedores
  (ap_direct_debits) y reembolsos de proveedores (supplier_refunds).
- Categorías y no-partida (src/bankrec/categories.py): clasificador, strong_categories, all_categories, UNIDENTIFIED.
- Preparación de FINAL: drift, tiempo, memoria, robustez.
FUERA de tu ámbito (son de las personas B y C): ar_text.py + src/matching/ (B), remittance.py (C), pdf_clauses.py,
ar_exact.py, ar_multi.py. Puedes LEERLOS y diagnosticar con ellos, pero no los modifiques. Si encuentras un problema ahí,
anótalo en una sección "Para B/C" de tu informe.

REPO Y SETUP
- Repo: https://github.com/liiiaamm/talky-hackathon. Parte de la rama feat/ar-exact-open-items (commit 564867a o posterior).
  git fetch origin && git checkout -b agentA/<N>-<tema> origin/feat/ar-exact-open-items
- Setup (ver README.md):
    uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
    datos de https://hackathon.usetalky.com/data/v1/ (bankrec-docs-and-masters.zip, bankrec-train-lite.tar.gz,
    bankrec-live.tar.gz), extraídos en data/ ; luego: find data -name '._*' -delete ; python eda/00_to_parquet.py
    cd src && python -m bankrec.categories --train      # -> models/categories.joblib (entrenado hasta 2024-12)
    cp ../models/categories.joblib ../models/categories_to2024.joblib
- Reglas del etiquetado: data/POLICIES.md (síguelas AL PIE DE LA LETRA), data/FORMAT.md, data/DATA_DICTIONARY.md, EDA.md.

PUNTUACIÓN
Score = 100 × (0,5·F1 de pares (bank_txn_id,item_id) con importe dentro de max(0,01; 0,5 %) + 0,3·exactitud por movimiento
+ 0,2·macro-F1 de excepciones). Las 10 clases de excepción pesan IGUAL en el macro-F1.
Último envío LIVE (servidor, 203k líneas no etiquetadas): 82,86 = F1 0,867 (P 0,996 / R 0,768) · TxnAcc 0,874 · ExcF1 0,663.
El desglose por segmento del servidor está en submissions/log.jsonl (metrics.segments). Acierto LIVE de segmentos de A:
ap_confirming 0,65 · ap_factored 0,88 · ap_direct_debit 0,94 · ap_supplier_refund 0,94 · ap_duplicate_refund 0,73 ·
ap_duplicate_payment 0,44 · ap_returned_transfer 0,93 · ap_fx 0,97 · ar_dd_returned 0,91 · ar_dd_represented_transfer 0,82 ·
ar_card_chargeback 0,77 · ar_cheque_returned 0,90 · ar_credit_note_paid_out 0,00 · unidentified 0,80 ·
unidentified_lookalike 0,79 · payment_multi/single, dd_batch, card_settlement, cheque_deposit ≈ 0,99-1,00.

PIPELINE (orden actual; el PRIMER resolver que reclama una línea se la queda)
  deterministic: dd_batches, dd_returns, card_settlements, card_chargebacks, cheque_deposits, cheque_returns,
                 payment_runs, payment_returns
  categories:    strong_categories (p ≥ 0,97)
  deterministic: ap_direct_debits, supplier_refunds
  remittance (C) → pdf_clauses → ar_text (B) → ar_exact → ar_multi → categories.all_categories → fallback
data.load(phase, from, to) -> ctx (ctx.bank, ctx.items con open_amount, ctx.all_items, dd/cards/cheques/payruns, maestros,
golden solo en TRAIN). En ventanas de TRAIN, open_amount = saldo al inicio de la ventana (como open_items_at_start en LIVE).
Variables: BANKREC_ORDER=remittance,pdf_clauses,ar_text,ar_exact,ar_multi · BANKREC_SKIP=mod1,mod2 · BANKREC_LEDGER=1 ·
BANKREC_CAT_MODEL=../models/categories_to2024.joblib (para medir sin fuga).

CÓMO MEDIR (siempre las DOS ventanas)
  cd src && export BANKREC_CAT_MODEL=../models/categories_to2024.joblib
  python -m bankrec.pipeline --phase train --from 2025-01-01 --to 2025-06-30 --score
  python -m bankrec.pipeline --phase train --from 2024-07-01 --to 2024-12-31 --score
Foto para análisis de errores (salida del pipeline con el módulo de cada línea + propuestas de cada módulo de cobros):
  python eda/10_snapshot.py 2025-01-01 2025-06-30 h1      # -> out/snap_h1_{sub,alt,gold,pool}.parquet
Para iterar rápido sobre deterministas/categorías: BANKREC_SKIP=ar_text (ar_text tarda ~4,5 min y usa ~7 GB),
pero el número final se da con el pipeline completo. No lances varias ejecuciones completas a la vez.

REGLAS
1. Anti-overfitting: solo reglas de POLICIES.md o patrones estructurales (ids, formatos, ficheros). Un cambio vale SOLO si
   mejora en las dos ventanas. Nada de umbrales afinados a un caso. Piensa en FINAL (2026-H1): filial nueva ZA01 (ZAR, banco
   nuevo, facturas "KWI…"), maestros nuevos en final/masters_final/, IT01 cambia de cuenta de cobro, pagos más lentos.
2. Revisa las suposiciones del código de A: busca reglas que NO se cumplan en todos los casos (contraejemplos en el golden).
3. Precisión antes que recall: una línea mal reclamada por un módulo temprano impide que otro la resuelva bien.
4. Trabaja en tu rama. Los cambios a ficheros de A van como commits pequeños y explicados en tu rama, medidos. No toques
   los módulos de B/C.
5. No envíes nada a la plataforma (no tienes token) y no subas datos, modelos ni out/ al repo.

ENTREGABLE
reports/agentA<N>.md con: (a) hallazgos ordenados por puntos ganables, cada uno con evidencia (conteos y ejemplos
bank_txn_id); (b) suposiciones erróneas encontradas; (c) cambio propuesto con su medición en LAS DOS ventanas (tabla
antes/después: Score, F1, P, R, TxnAcc, ExcF1 y segmentos o clases afectados); (d) riesgos para FINAL; (e) "Para B/C".
Haz push de tu rama y abre un PR contra feat/ar-exact-open-items titulado "[A·<N>] <tema>: +X (H1) / +Y (H2)".
```

---

## A·1 · Payment runs: confirming, factoring, holding, FX y excepciones AP

```
TU INVESTIGACIÓN (A·1): deterministic.payment_runs y sus excepciones. Lado AP: ~30 % de las líneas y ~28 % de los pares.
- Confirming (0,65 LIVE): hoy se agrupa por fecha de vencimiento (máx./mín.) o por un único beneficiario ("N PROVEEDORES").
  Saca los adeudos de confirming no enlazados o mal enlazados y averigua la regla real: ¿vencimiento de la factura, día
  hábil siguiente, agrupación por beneficiario y fecha?
- Factoring (0,88): ¿qué falla cuando el cesionario (Ribagorza Confirming, Norte Pagos, Credifácil…) cobra por el proveedor?
- ap_fx (6 % de los pares, adeudos agregados de hasta 1.540 facturas): PARTIAL frente a FX_DIFFERENCE, abonos en FX, importe
  en la divisa de la factura.
- PARTIAL en AP: acumulado pagado por factura dentro de la fase frente a open_amount. ¿Hay contraejemplos?
- Líneas AP aún sin enlazar: ¿e2e truncado ambiguo, importe distinto, cuenta pagadora distinta?
Mide el impacto en pares (F1) además del de líneas.
```

## A·2 · Resto del lado AP: recibos de proveedores, reembolsos, devoluciones y duplicados

```
TU INVESTIGACIÓN (A·2): ap_direct_debits (0,94 LIVE), supplier_refunds (0,94), ap_duplicate_refund (0,73),
ap_returned_transfer (0,93), ap_duplicate_payment (0,44).
- Recibos de proveedor (SEPA DD, "MREF/RUM … RECIBO …", "Recibos varios"): hoy se exige proveedor en el texto + importe
  exacto + vencimiento más cercano. ¿Se puede identificar al proveedor por el creditor id (CRED+…ZZZ…) o el mandato?
  ¿Qué casos fallan?
- Reembolsos: un APC (abono de proveedor) devuelto por transferencia, frente a SUPPLIER_REFUND (categoría: devolución de un
  pago que hicimos dos veces). ¿Cómo distinguirlos?
- Duplicados AP: el sufijo "D" del e2e solo cubre parte. ¿Qué otros duplicados hay (factura ya pagada que se vuelve a
  pagar)? Ojo con la precisión: una regla "factura ya cerrada → duplicado" dio muchos falsos positivos.
- Transferencias devueltas (RETURN …/DEVOLUCION TRANSFERENCIA …): ¿qué falla en el 7 %?
```

## A·3 · Clientes, deterministas: recibos, devoluciones, re-presentaciones, tarjetas, cheques

```
TU INVESTIGACIÓN (A·3): dd_batches (≈1,00), dd_returns (0,91 LIVE), ar_dd_represented_transfer (0,82),
card_chargebacks (0,77), cheque_returns (0,90), ar_credit_note_paid_out (0,00).
- Devoluciones de recibo: mandato en el texto → recibo de la remesa. ¿Qué falla en el 9 %? (varios recibos del mismo
  mandato con el mismo importe, devolución de un recibo de otra fase, re-presentaciones ya cobradas…).
- Re-presentados: un recibo devuelto que el cliente paga después por transferencia ("PAGO RECIBO 0508…") o se re-presenta
  en otra remesa (re_presentation=Y). ¿Quién reclama hoy esas líneas y con qué partida?
- Contracargos de tarjeta y cheques devueltos: casos sin el número de pedido o de cheque, o con varios candidatos.
- ar_credit_note_paid_out: rappel pagado al cliente por transferencia (nuestro adeudo liquida el ARC). Diseña el resolver
  determinista.
- Comprueba que dd_batches/card/cheque usen all_items y no tengan colisiones de ids entre cuentas o fases.
```

## A·4 · Categorías y no-partida: clasificador, umbrales y UNIDENTIFIED

```
TU INVESTIGACIÓN (A·4): categories.py. Hoy strong_categories (p ≥ 0,97) acierta el 100 %. all_categories da UNIDENTIFIED
(categoría) a los cobros sin resolver y pone la excepción UNIDENTIFIED solo si el modelo predice UNIDENTIFIED
(F1 0,88; unidentified 0,80 y lookalike 0,79 en LIVE).
- Matriz de confusión del clasificador por segmento en las dos ventanas. ¿Qué categorías se confunden (BANK_FEES frente a
  BANK_FEES_intl, INTERCOMPANY frente a INTERNAL_TRANSFER, CASH_POOL en los dos lados, FX_DEAL en las dos patas,
  BANK_REVERSAL en ambas líneas)?
- ¿Bajar el umbral de strong (0,95, 0,90) gana o quita líneas a los módulos de cobros? Mídelo con el pipeline completo.
- Pagos (débitos) sin resolver: hoy reciben la mejor categoría no-ITEM. ¿Hay un patrón mejor?
- UNIDENTIFIED: ¿qué rasgos observables (sin golden) separan un no identificado de un cobro que nadie supo casar?
  ¿Se puede subir el recall sin perder precisión?
- Robustez para FINAL: la entidad ZA01 nunca vista (token ENT_), textos de camt053 nuevos de Moncayo. Entrena con
  --all y valida en una ventana distinta.
```

## A·5 · Ledger compartido y saldos

```
TU INVESTIGACIÓN (A·5): coherencia de saldos entre resolvers (infraestructura de pipeline.py).
- En la foto, cuenta las partidas que aparecen en más de un movimiento: legítimas (PARTIAL, RETURNED, DUPLICATE_PAYMENT,
  cuotas) e ilegítimas (la misma factura liquidada entera dos veces). B midió 320 en 2025-H1 y 344 en LIVE.
- Mide BANKREC_LEDGER=1 (pipeline.apply_ledger) en las dos ventanas y revisa su lógica contra POLICIES §2-3.
  Bug conocido: una partida cerrada no puede reabrirse con un RETURNED porque se elimina de ctx.items (usa ctx.all_items).
- Diseña un ledger cronológico: los resolvers van por módulos, no por fecha, así que un módulo temprano puede consumir con
  un cobro de fecha posterior una factura que un módulo tardío necesita con un cobro anterior. ¿Cuánto pasa? Arréglalo
  dentro de pipeline.py.
- Comprueba que data._open_at reproduce live/open_items_at_start: aplica la misma reconstrucción a la frontera TRAIN→LIVE
  (2025-07-01) y compárala con el fichero oficial, partida a partida.
```

## A·6 · Orden de módulos y árbitro de integración

```
TU INVESTIGACIÓN (A·6): hoy cada línea es del PRIMER resolver que la reclama. Con out/snap_h1_alt.parquet (propuestas de cada
módulo de cobros por separado sobre el mismo pool):
- Para cada línea, ¿qué resolvers aciertan? Calcula el techo de un enrutado perfecto y la distancia al orden actual.
- Busca señales observables (sin golden) que predigan qué propuesta es correcta cuando hay conflicto: tipo de evidencia,
  número de candidatos, suma exacta, aviso de remesa, coincidencia de propuestas entre módulos…
- Diseña un árbitro en pipeline.py: los resolvers proponen y un juez elige, incluida la opción de no reclamar. Usa reglas
  sencillas o un modelo pequeño entrenado SOLO con TRAIN anterior a 2024-07, validado en las dos ventanas.
- Mide órdenes alternativos con BANKREC_ORDER, incluida la posición de strong_categories y de los resolvers AP.
Entrega el mejor orden fijo y, si compensa, el árbitro.
```

## A·7 · Diagnóstico global de pares perdidos (recall 0,768) y reparto por dueño

```
TU INVESTIGACIÓN (A·7): el recall de pares es 0,768 en LIVE (~6,6 pts). Como integrador, atribuye CADA par perdido a su
causa y a su dueño (A, B o C), y arregla solo las causas de A. Con la foto (out/snap_h1_*):
- Recall y precisión de pares por segmento, por módulo y por tamaño del movimiento (1, 2-5, 6-20, 21-100, >100 partidas),
  ponderando por pares.
- Clasifica la pérdida: línea no reclamada, línea reclamada por el módulo equivocado (y cuál habría acertado), partidas de
  menos, importe fuera de tolerancia, categoría en vez de partidas.
- Para los 30 movimientos con más pares perdidos, la causa concreta.
- Entrega una tabla de causas con pares recuperables y dueño, para que el equipo priorice. Arregla las causas de A.
```

## A·8 · Capa de datos, evaluador y validador: paridad con el servidor

```
TU INVESTIGACIÓN (A·8): que lo que medimos en local sea lo que mide el servidor.
- Ejecuta el evaluador OFICIAL (data/starter/bankrec_score.py: load_golden + score_submission) sobre nuestras salidas de las
  dos ventanas y compáralo con src/bankrec/score.py: score, métricas, por segmento. Cualquier diferencia es un bug.
- Validador (pipeline.validate) frente a las reglas del servidor (FORMAT.md + bankrec_score.py): ids, filas duplicadas,
  ítem + categoría, importes, redondeo a 2 decimales, CRLF/BOM. Que un fichero inválido nunca llegue al envío FINAL.
- data.py: revisa las suposiciones (entidad = prefijo del account_id, documentos no únicos entre entidades, _items_from_tables
  para LIVE/FINAL con fases anteriores, partidas de open_items_at_start que no están en nuestras tablas, credit notes "Anul…").
- Calibración local→servidor: v5 local 80,69 → servidor 78,95; v6 83,44 → 82,86. Compara el desglose por segmento del
  servidor (submissions/log.jsonl) con el local para ver qué segmentos se comportan distinto en LIVE (drift).
```

## A·9 · FINAL: drift y runbook de las 18:00

```
TU INVESTIGACIÓN (A·9): FINAL (2026-01..06, 220k líneas) sale a las 18:00, hay UN solo envío y la nota queda oculta hasta las
20:30. Drift conocido (POLICIES §6): ZA01 (adquirida el 2026-01-01, ZAR, banco nuevo, facturas "KWI…", renombrada el
2026-04-01, final/masters_final/), IT01 cambia de cuenta de cobro el 2026-02-01, Moncayo en camt053, clientes renombrados
y pagos más lentos. En LIVE ya hay 2 cobros de ZA01 (anticipos de proforma → CUSTOMER_ADVANCE).
- Revisa todo el código de A en busca de supuestos de TRAIN/LIVE: rutas de maestros, PAYFACTORY, entidades o cuentas
  fijas, cachés por fase (data/parquet/<phase>/…), prefijos de documento, el clasificador ante una entidad nueva.
- Simula el drift: quita una entidad entera del entrenamiento de categorías y mide en ella; renombra clientes; mueve fechas.
- Ensaya tools/prepare_final.sh tratando LIVE como si fuera FINAL (copia live→final en un directorio temporal, incluida una
  carpeta masters_final): tiempo total, pico de RAM, validación y distribuciones.
- Entrega la lista de cosas que pueden romperse con su arreglo y una checklist minuto a minuto de 18:00 a 19:30.
```

## A·10 · Rendimiento y robustez del pipeline

```
TU INVESTIGACIÓN (A·10): el pipeline completo tarda ~6 min en LIVE (ar_text ~4,5 min) con picos de ~8 GB. En FINAL no puede
fallar.
- Perfila tiempo y memoria por módulo. ¿Dónde se puede recortar sin cambiar resultados (cachés de parseo, lecturas parquet
  repetidas, joins grandes, conversiones a pandas)? Verifica que la salida sea idéntica (hash del CSV) antes y después.
- Robustez: si un módulo lanza una excepción, el pipeline debe seguir con los demás y registrarlo (try/except por resolver
  en pipeline.run, con aviso visible). Añade un modo --safe.
- Determinismo: dos ejecuciones deben dar el mismo fichero (orden de filas, desempates estables). Compruébalo.
- Tamaño del envío: ≤ 60 MB sin comprimir y ≤ 3M filas. Hoy LIVE son 628k filas y ~29 MB. FINAL tendrá ~10 % más líneas.
  Comprueba el margen.
- Propón y mide, con el diff a pipeline.py, un `--profile` que imprima tiempo y RAM por módulo.
```
