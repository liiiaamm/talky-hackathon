# Prompts para 10 agentes de investigación

Cada prompt es el **BLOQUE COMÚN** seguido del **bloque específico** del agente. Pega los dos juntos.

---

## BLOQUE COMÚN (va al principio de los 10 prompts)

```
Eres un agente de investigación en un hackathon de conciliación bancaria ("Close the Books", Grupo Valtermia, datos
sintéticos). Objetivo del equipo: pasar de ~85 a 90+ puntos. Te toca UNA línea de investigación (al final). Hazla a fondo,
con números, y no toques lo de los demás.

REPO Y SETUP
- Repo: https://github.com/liiiaamm/talky-hackathon. Parte de la rama feat/ar-exact-open-items (commit 49b988c o posterior).
  git fetch origin && git checkout -b agent/<N>-<tema> origin/feat/ar-exact-open-items
- Setup: README.md. Resumen:
    uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
    datos de https://hackathon.usetalky.com/data/v1/ (bankrec-docs-and-masters.zip, bankrec-train-lite.tar.gz,
    bankrec-live.tar.gz), extraídos en data/ ; luego: find data -name '._*' -delete ; python eda/00_to_parquet.py
    cd src && python -m bankrec.categories --train      # -> models/categories.joblib (entrenado hasta 2024-12)
    cp ../models/categories.joblib ../models/categories_to2024.joblib
- Reglas del juego: data/POLICIES.md (cómo se etiquetó el golden; síguelas AL PIE DE LA LETRA), data/FORMAT.md,
  data/DATA_DICTIONARY.md, EDA.md (exploración previa).

PUNTUACIÓN
Score = 100 × (0,5·F1 de pares (bank_txn_id,item_id) con importe dentro de max(0,01; 0,5 %) + 0,3·exactitud por movimiento
+ 0,2·macro-F1 de excepciones). Las 10 clases de excepción pesan IGUAL en el macro-F1.
Último envío LIVE (servidor, 203k líneas no etiquetadas): 82,86 = F1 0,867 (P 0,996 / R 0,768) · TxnAcc 0,874 · ExcF1 0,663.
→ Los puntos perdidos están en el RECALL de pares (~6,6 pts) y en las excepciones (~6,7 pts); la exactitud solo pierde ~3,8.
F1 por clase (TRAIN 2025-H1): SHORT_DISCOUNT 0,97 · RETURNED 0,96 · SHORT_DISPUTE 0,90 · SHORT_WITHHOLDING 0,90 ·
UNIDENTIFIED 0,88 · FX_DIFFERENCE 0,87 · PARTIAL 0,68 · OVERPAYMENT 0,47 · DUPLICATE_PAYMENT 0,11 · SHORT_BANK_CHARGES 0,00.
El desglose por segmento del servidor está en submissions/log.jsonl (campo metrics.segments).

PIPELINE (src/bankrec/)
pipeline.py ejecuta resolvers en orden; el PRIMERO que reclama una línea se la queda:
  deterministic.py: dd_batches, dd_returns, card_settlements, card_chargebacks, cheque_deposits, cheque_returns,
                    payment_runs, payment_returns
  categories.py:    strong_categories (p ≥ 0,97)
  deterministic.py: ap_direct_debits, supplier_refunds
  remittance.py (avisos de remesa, de C) → pdf_clauses.py (cláusulas PDF) → ar_text.py + src/matching/ (de B)
  → ar_exact.py (exact_amount, adjusted_amount) → ar_multi.py (1:N) → categories.all_categories → fallback
data.py: load(phase, from, to) -> ctx (ctx.bank, ctx.items con open_amount, ctx.all_items, dd/cards/cheques/payruns,
maestros, golden solo en TRAIN). En ventanas de TRAIN, open_amount = saldo al inicio de la ventana (como en LIVE).
score.py: evaluador local, misma matemática que el oficial (data/starter/bankrec_score.py), con desglose por segmento/módulo.
Variables: BANKREC_ORDER=remittance,pdf_clauses,ar_text,ar_exact,ar_multi · BANKREC_SKIP=mod1,mod2 · BANKREC_LEDGER=1 ·
BANKREC_CAT_MODEL=../models/categories_to2024.joblib (para medir sin fuga).

CÓMO MEDIR (siempre las DOS ventanas)
  cd src && export BANKREC_CAT_MODEL=../models/categories_to2024.joblib
  python -m bankrec.pipeline --phase train --from 2025-01-01 --to 2025-06-30 --score
  python -m bankrec.pipeline --phase train --from 2024-07-01 --to 2024-12-31 --score
Foto para análisis de errores (salida del pipeline + propuestas de cada módulo de cobros por separado, ~12 min):
  python eda/10_snapshot.py 2025-01-01 2025-06-30 h1      # -> out/snap_h1_{sub,alt,gold,pool}.parquet
Tarda ~6 min por ventana y usa hasta ~8 GB de RAM (ar_text). No lances varias a la vez si tienes poca memoria.

REGLAS
1. Anti-overfitting: solo reglas que salgan de POLICIES.md o de patrones estructurales (plantillas, formatos). Un cambio
   vale SOLO si mejora en las dos ventanas. Nada de umbrales afinados a un caso. Piensa en FINAL (2026-H1): filial nueva
   ZA01 (ZAR, facturas "KWI…"), maestros nuevos en final/masters_final/, pagos más lentos, clientes renombrados.
2. Revisa las suposiciones: busca reglas del código que NO se cumplan en todos los casos (contraejemplos en el golden).
3. Precisión antes que recall: una línea mal reclamada impide que otro módulo la resuelva bien.
4. NO modifiques pipeline.py, data.py, deterministic.py, score.py ni los módulos de otros. Si hace falta tocarlos,
   proponlo como diff en tu informe. El código nuevo va en src/bankrec/agent<N>_<tema>.py (con RESOLVERS = [...]) o en
   eda/agent<N>_*.py. Para probarlo: BANKREC_ORDER con tu módulo en la posición que propongas.
5. No envíes nada a la plataforma (no tienes token) y no subas datos, modelos ni out/ al repo.

ENTREGABLE
reports/agent<N>.md con: (a) hallazgos ordenados por puntos ganables, cada uno con la evidencia (conteos y ejemplos
bank_txn_id del golden); (b) suposiciones erróneas encontradas; (c) cambio propuesto con su medición en LAS DOS ventanas
(tabla antes/después: Score, F1, P, R, TxnAcc, ExcF1 y las clases afectadas); (d) riesgos para FINAL.
Haz push de tu rama agent/<N>-<tema> y abre un PR contra feat/ar-exact-open-items titulado "[agent N] <tema>: +X / +Y".
```

---

## Agente 1 · Recall de pares: dónde están los pares que faltan

```
TU INVESTIGACIÓN (agente 1): el recall de pares es 0,768 en LIVE. ¿Dónde están los ~23 % de pares (bank_txn_id,item_id)
del golden que no emitimos? Con la foto (out/snap_h1_*):
- Recall y precisión de pares por segmento, por módulo y por tamaño del movimiento (1, 2-5, 6-20, 21-100, >100 partidas).
  Pondera por pares, no por líneas: un solo movimiento con 900 facturas pesa más que 500 movimientos de 1.
- Separa: líneas sin reclamar con muchas partidas, líneas reclamadas con partidas de menos (allocations parciales)
  y líneas reclamadas con importes fuera de tolerancia.
- Para los 30 movimientos con más pares perdidos, explica la causa: aviso de remesa no enlazado, documento no mapeado,
  falta un abono, etc.
- Entrega un ranking de causas con los pares recuperables de cada una y un prototipo para la causa nº 1.
```

## Agente 2 · Ledger compartido y facturas liquidadas dos veces

```
TU INVESTIGACIÓN (agente 2): coherencia de saldos entre resolvers.
- Cuenta en la foto las partidas que aparecen en más de un movimiento, distinguiendo las legítimas (PARTIAL, RETURNED,
  DUPLICATE_PAYMENT, cuotas) de las ilegítimas (la misma factura liquidada entera dos veces).
- Mide BANKREC_LEDGER=1 (pipeline.apply_ledger) en las dos ventanas. Revisa su lógica contra POLICIES §2-3: RETURNED reabre,
  DUPLICATE no resta, las excepciones de "cierre completo" cierran, los parciales restan lo pagado.
- Bug conocido: una partida cerrada no puede reabrirse porque se elimina de ctx.items. Usa ctx.all_items.
- Diseña un ledger cronológico de verdad: los resolvers van por módulos, no por fecha, así que un módulo tardío puede
  necesitar una factura que otro anterior consumió con un cobro de fecha posterior.
- Mide el impacto en ar_dispute_settlement, ar_retention_release y en la última cuota de ar_instalment, que dependen del
  saldo vivo. Propón un diff para pipeline.py.
```

## Agente 3 · Enrutado y orden de módulos (árbitro en vez de "el primero gana")

```
TU INVESTIGACIÓN (agente 3): el pipeline da cada línea al PRIMER resolver que la reclama. Con out/snap_h1_alt.parquet
(propuestas de cada módulo de cobros por separado sobre el mismo pool):
- Para cada línea, ¿qué resolvers aciertan? Calcula el techo de un enrutado perfecto y cuánto lo superamos.
- Busca señales observables (sin golden) que predigan qué módulo acierta: tipo de evidencia, número de candidatos,
  si hay remesa, formato del extracto, segmento probable…
- Diseña un árbitro con reglas sencillas o un modelo pequeño entrenado SOLO con TRAIN anterior a 2024-07 (validar en las dos
  ventanas) que elija entre propuestas en conflicto, incluido "no reclamar".
- Mide también otros órdenes con BANKREC_ORDER. Entrega el mejor orden fijo y, si compensa, el árbitro como módulo.
```

## Agente 4 · SHORT_BANK_CHARGES y FX_DIFFERENCE

```
TU INVESTIGACIÓN (agente 4): dos clases de excepción (cada clase vale ~2 pts del macro-F1).
SHORT_BANK_CHARGES (F1 0,00): POLICIES §3 dice transferencia de fuera de SEPA con merma ≤ 60 EUR equivalentes y ≤ 2 %
de la partida, liquidada entera. Estudia ar_short_bank_charges en el golden: bank_code (XBCT…), orig_amount/orig_currency,
la línea de comisión aparte del mismo día y la conversión a EUR con masters/fx_rates.csv (último tipo disponible si es
festivo). Define cuándo NO es bank charges (por encima de la tolerancia → PARTIAL).
FX_DIFFERENCE (F1 0,87; ar_fx acierta 0,25 en LIVE): cobros en una divisa distinta a la de la factura, con y sin
orig_amount, también /OCMT/…/EXCH/ en MT940 (spread 0,1-1,5 %, tolerancia 3 %). Cubre 1:N en FX y las reglas de
sustitución (FX reemplaza a vacío/DISCOUNT/WHT/BANK_CHARGES; PARTIAL/DISPUTE se mantienen; los abonos van vacíos).
Prototipo en src/bankrec/agent4_fx.py.
```

## Agente 5 · DUPLICATE_PAYMENT, OVERPAYMENT y devoluciones a cliente/proveedor

```
TU INVESTIGACIÓN (agente 5): DUPLICATE_PAYMENT (F1 0,11; ar_duplicate_payment acierta 0 en LIVE; ap_duplicate_payment 0,44)
y OVERPAYMENT (F1 0,47).
- DUPLICATE: una partida ya liquidada que se vuelve a pagar (AR y AP). La excepción va en el segundo pago. ¿Cómo detectarlo
  sin golden? Saldo 0 en open_items_at_start o liquidada antes dentro de la fase, misma referencia o importe, días de
  distancia… En AP mira el sufijo "D" del e2e en payment_runs y lo que pasa sin él. Relación con CUSTOMER_REFUND /
  SUPPLIER_REFUND (la devolución posterior del duplicado).
- OVERPAYMENT: cobro mayor que el saldo de UNA factura (§3). Separa los falsos positivos (líneas 1:N con números abreviados)
  de los verdaderos y estudia la distribución del exceso.
Cuantifica precisión y recall de cada regla en las dos ventanas. Prototipo en src/bankrec/agent5_dup_over.py.
```

## Agente 6 · Familia PARTIAL: cuotas, retenciones, disputas y descuentos tardíos

```
TU INVESTIGACIÓN (agente 6): PARTIAL (F1 0,68) y los segmentos que dependen del saldo vivo: ar_instalment (0,35 LIVE),
ar_partial (0,45), ar_retention (0,33), ar_dispute_settlement (0,31), ar_discount_late (0,60), ar_advance_deduction (0,23).
- Ya existe src/bankrec/pdf_clauses.py (~99 % de acierto en lo que reclama). Mide su COBERTURA: qué casos de esos segmentos
  no reclama y por qué (sin PDF, el PDF de TRAIN-lite falta antes de 2024-07, el importe no coincide, la cuota elegida está
  mal, la liberación de la retención ~12 meses después).
- ¿Cómo distinguir "última cuota" o "resto tras un parcial" (sin excepción) de un PARTIAL, sin PDF?
- Liquidación de disputas: "SETTLEMENT <doc>", importe = resto abierto tras un SHORT_DISPUTE anterior.
- Descuento tardío: la regla de plazo + 3 días de gracia, ¿se aplica igual en todos los casos? Busca contraejemplos.
Prototipo en src/bankrec/agent6_partial.py.
```

## Agente 7 · Avisos de remesa: cobertura y pares perdidos

```
TU INVESTIGACIÓN (agente 7): el módulo remittance.py (de C) acierta el 98 % en lo que reclama. Mide lo que NO reclama:
- Avisos parseados que no se enlazan con ningún cobro (¿por qué: importe, fecha fuera de -5..+20 días, cobro FX,
  referencia distinta?), documentos del aviso que no se mapean a ninguna partida (vouchers del cliente, prefijos,
  facturas antiguas cerradas) y cobros de segmentos de remesa sin aviso asociado.
- Cuánto pesan en pares ar_short_dispute (13 % de los pares del golden) y ar_1toN_remittance (7 %).
- El cambio de layout del gran cliente en 2025-09 (xlsx Voucher | Invoice | PO …): ¿se enlaza bien en LIVE?
- Avisos que llegan días después del cobro, o con varios cobros por aviso o varios avisos por cobro.
Ojo: TRAIN-lite solo trae avisos desde 2024-07, así que mide sobre 2024-H2 y 2025-H1. Entrega las causas con los pares
recuperables y un prototipo para la principal.
```

## Agente 8 · 1:N con abonos, n_to_m y clientes con muchas facturas abiertas

```
TU INVESTIGACIÓN (agente 8): ar_1toN_credit_note (0,33 LIVE), ar_n_to_m (0,15), ar_ref_noise (0,86 sobre 42k líneas, el
segmento más grande) y ar_same_amount_fifo (0,73).
- Para cada segmento, en la foto: ¿falla la identificación del pagador, la búsqueda de subconjuntos (el pool es demasiado
  grande o hay varios subconjuntos válidos), el orden FIFO o el desempate?
- Abonos: rappel trimestral (el PDF dice si se compensa o se paga), abonos de devolución con related_document, abonos de
  proveedor. ¿Cuándo entra un abono en un cobro y cuándo no?
- n_to_m: varios cobros para varias facturas. ¿Hay patrones (pagos a cuenta, plan de pagos)?
- same_amount_fifo: comprueba la regla "la más antigua abierta" de POLICIES §5.6 en el golden. ¿Es por fecha de factura o de
  vencimiento? ¿Qué pasa con las facturas ya pagadas en la misma fase?
Prototipo en src/bankrec/agent8_multi.py (compara con ar_text y ar_multi).
```

## Agente 9 · Auditoría de los módulos de la Persona A (deterministas, categorías, fallback)

```
TU INVESTIGACIÓN (agente 9): los resolvers deterministas y de categorías. Acierto en LIVE:
ar_dd_returned 0,91 · ar_dd_represented_transfer 0,82 · ap_confirming 0,65 · ap_factored 0,88 · ap_direct_debit 0,94 ·
ap_supplier_refund 0,94 · ap_duplicate_refund 0,73 · ap_returned_transfer 0,93 · ar_card_chargeback 0,77 ·
ar_cheque_returned 0,90 · ar_credit_note_paid_out 0,00 · unidentified 0,80 · unidentified_lookalike 0,79.
- Para cada segmento, saca los errores de la foto (deterministic.py / categories.py): ¿línea no enlazada, partida
  equivocada, importe o excepción mal?
- Revisa las suposiciones: identificadores no únicos entre cuentas, cuentas fuera del maestro, el confirming agrupado por
  vencimiento, el factoring (cesionario Ribagorza/Norte Pagos/Credifácil pagando por el proveedor), los re-presentados.
- Rappel pagado al cliente (ar_credit_note_paid_out: nuestra transferencia liquida el ARC).
- UNIDENTIFIED: umbral del clasificador frente a "no hay partida que case" (§3). ¿Se puede predecir mejor?
Entrega los arreglos como diff propuesto para deterministic.py/categories.py, medidos en las dos ventanas.
```

## Agente 10 · Robustez para FINAL (drift, tiempo y ensayo de las 18:00)

```
TU INVESTIGACIÓN (agente 10): FINAL (2026-01..06, 220k líneas) se publica a las 18:00 y hay UN solo envío, oculto hasta
las 20:30. Drift conocido (POLICIES §6): ZA01 (adquirida el 2026-01-01, ZAR, banco nuevo, facturas "KWI…", renombrada el
2026-04-01, final/masters_final/), IT01 cambia de cuenta de cobro el 2026-02-01, Moncayo en camt053, clientes renombrados
y pagos más lentos (más cobros fuera de plazo de descuento y más parciales).
- Revisa dónde el código asume TRAIN/LIVE: rutas de maestros, entidades/cuentas fijas, cachés por fase, prefijos de
  documento, el clasificador de categorías ante una entidad nueva, la lista PAYFACTORY, las listas de países de retención.
- Simula el drift: quita una entidad entera de TRAIN y mide en ella, renombra clientes en una ventana, retrasa
  fechas, etc. Mide la degradación por módulo.
- Ensaya tools/prepare_final.sh contra LIVE como si fuera FINAL (copia live→final en una carpeta temporal): tiempo total,
  pico de RAM, validación y comparación de distribuciones.
- Lista lo que pueda romperse a las 18:00 con su arreglo, y una checklist de 90 minutos.
```
