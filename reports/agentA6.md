# A·6 — Routing: orden de módulos y árbitro

**Conclusión: el enrutado no es donde están los puntos.** Si cada línea fuera al resolver correcto (oráculo con golden), el techo es
**+0,21 puntos** en 2025-H1. El orden actual ya es el mejor de las 120 permutaciones. Un árbitro sencillo con señales observables da
+0,02 (dentro de muestra). **Recomendación: mantener `BANKREC_ORDER=remittance,pdf_clauses,ar_text,ar_exact,ar_multi` y no
integrar ningún árbitro antes de FINAL.**

## Método

- Foto 2025-H1 (`out/snap_h1_*`, código bf9a167 y `categories_to2024.joblib`): propuestas standalone de cada resolver de cobros sobre
  el **mismo pool** de 115.280 líneas, es decir, lo que queda tras deterministic + strong_categories + AP. También están
  `all_categories` sobre todo el pool (equivale a "no reclamar") y el sub/gold del pipeline completo (`/home/liame/claude_hackathon/out/snap_h1_*`).
- `eda/11_routing.py` calcula, para cada par (línea, resolver), si la línea es correcta con la regla oficial de TxnAcc. Después
  rehace cualquier orden o árbitro eligiendo una propuesta por línea y lo puntúa con `bankrec.score`. Solo lee, usa 1,4 GB y tarda 80 s.
- **Validación del replay:** el orden actual rehecho desde las propuestas standalone da **85,16**, y el pipeline real da **85,14**. La
  aproximación "cada resolver propone igual sea cual sea el pool" es buena.

## (a) Hallazgos, ordenados por puntos ganables

| # | Hallazgo | Puntos (H1) |
|---|---|---|
| 1 | **Techo del enrutado perfecto:** 98.976 líneas correctas en el pool, frente a 98.234 del orden actual (+742 líneas, 0,64 % del pool). El oráculo (de las propuestas correctas, la primera según el orden) da 85,37: F1 0,8802 (P 0,9952 / R 0,7890) · TxnAcc 0,9169 · Exc 0,6929. El orden actual da 85,16: F1 0,8786 (P 0,9924 / R 0,7882) · TxnAcc 0,9142 · Exc 0,6902. | **+0,21 máximo** |
| 2 | **El orden fijo actual es el óptimo** entre las 120 permutaciones de grupos. Intercambiar remittance y pdf_clauses da lo mismo (85,16). Con ar_multi antes que ar_exact: 85,15. Con ar_text al final: 85,09. Con remittance detrás de ar_text: entre 85,09 y 85,13. | 0 |
| 3 | **El juego de verdad está fuera del enrutado:** 18.241 líneas del pool no reciben NINGUNA propuesta de cobros y acaban en all_categories. En el pipeline, `cat_rest` tiene 17.638 líneas con un 22 % de acierto, y en 12.869 de ellas el golden tiene partidas. Otras ~16.300 líneas no las acierta ningún resolver. | fuera de A·6 (recall de B/C) |
| 4 | **strong_categories, ap_direct_debits y supplier_refunds no roban líneas:** aciertan el 100 %, 100 % y 99,96 % de 14.237, 5.054 y 2.383 líneas, con 0, 0 y 1 líneas mal cuyo golden tiene partidas. Retrasarlos no puede ganar nada. Ponerlos detrás obligaría a los resolvers de cobros a ver 21.000 líneas más, con más riesgo de FP y más tiempo. | 0 (mantener) |
| 5 | **Árbitro sencillo** (ver más abajo): cambia 88 líneas, de las que ar_text acertaba 12 y exact_amount acierta 66. Resultado: 85,16 → 85,18 (F1 +0,0003, P +0,0006). | +0,02 (solo H1) |

### Conflictos (líneas con propuestas distintas)

La mayoría de las líneas no tienen conflicto: 70.403 reciben 2 propuestas idénticas y 2.488 reciben 3. Solo 2.751 líneas tienen
propuestas distintas. Tabla por pares de resolvers ("A ok" / "B ok" = cuántas veces acierta cada uno):

| A vs B | conflictos | A ok | B ok | ninguno | lo decide el orden actual |
|---|---|---|---|---|---|
| ar_text vs multi_invoice | 1.374 | **1.133** | 109 | 132 | gana ar_text ✔ |
| exact_amount vs multi_invoice | 1.029 | **998** | 10 | 21 | gana exact ✔ |
| ar_text vs exact_amount | 698 | 331 | **336** | 31 | gana ar_text (a cara o cruz) |
| adjusted_amount vs multi_invoice | 616 | 232 | **278** | 106 | gana adjusted (casi empate) |
| adjusted_amount vs ar_text | 375 | 45 | **310** | 20 | gana ar_text ✔ |
| remittances / pdf_clauses vs cualquiera | < 30 por par | — | remit/pdf casi siempre | — | van delante ✔ |

### Señales observables sin golden

- **Coincidencia entre módulos.** Si 2 resolvers proponen el mismo conjunto de partidas, aciertan el 98,6 % (71.917 propuestas). Una
  propuesta en solitario acierta el 84,8 % (25.392). Pero el orden ya respeta casi siempre la mayoría: en ar_text vs exact, cuando
  ar_text tiene 2 o más votos gana 34 a 7, y ya es el primero.
- **Submódulo de ar_text en el conflicto con exact_amount:**
  - `ar_party_subset` de 1 factura: 290 frente a 269, es decir, a cara o cruz. Por segmento: en `ar_same_amount_fifo` gana ar_text
    (164/95) y en `ar_same_amount_month` gana exact (68/127). El desempate entre facturas del mismo importe, FIFO o mes de servicio,
    no se ve desde el router: es lógica de B (ver "Para B/C").
  - `ar_partial_hint`: exact gana 31 a 10.
  - `ar_party_subset` con 2 o más facturas cuando exact tiene un 1:1 exacto: exact gana 35 a 5.
  - `*_disc`, `ar_ref_exact*` y `wht*`: gana siempre ar_text (26/0).
- **Regla del árbitro probada:** "si ar_text reclama con `ar_partial_hint`, o con `ar_party_subset` de 2 o más facturas, y
  exact_amount propone una factura 1:1 de importe exacto, gana exact_amount". Mueve 88 líneas y suma **+0,02**.

## (b) Suposiciones erróneas encontradas

- "El orden decide mucho": falso con los resolvers actuales. Las propuestas coinciden en el 97 % de las líneas disputables. Las
  diferencias entre órdenes que se midieron antes (84,22 frente a 84,63) venían de no tener remittance y pdf_clauses delante y de la
  versión de los módulos de entonces, no de conflictos que queden hoy.
- "Hace falta un árbitro con modelo": el techo total es +0,21. Un modelo entrenado antes de 2024-07 y validado en dos ventanas no
  compensa su riesgo ni su tiempo antes de FINAL.
- "strong_categories debería ir más tarde": no hay ni una línea de strong_categories cuyo golden tenga partidas.

## (c) Cambio propuesto y medición

**Ningún cambio de código en pipeline.py.** Se mantiene el orden fijo actual. El árbitro (+0,02 dentro de muestra en H1) **no se
integra**, por tres motivos: no hay foto standalone de H2 para validarlo, la ganancia es menor que el ruido entre versiones, y
duplicaría el tiempo de ar_text al tener que ejecutar todos los resolvers sobre el pool completo.

| H1 2025 (replay sobre la foto) | Score | F1 | P | R | TxnAcc | ExcF1 |
|---|---|---|---|---|---|---|
| pipeline real (orden actual) | 85,14 | — | — | — | — | — |
| replay del orden actual | 85,16 | 0,8786 | 0,9924 | 0,7882 | 0,9142 | 0,6902 |
| mejor orden alternativo (pdf_clauses, remittance, …) | 85,16 | 0,8786 | 0,9924 | 0,7882 | 0,9142 | 0,6902 |
| árbitro sencillo (regla de arriba) | 85,18 | 0,8789 | 0,9930 | 0,7883 | 0,9144 | 0,6903 |
| oráculo (techo, usa golden) | 85,37 | 0,8802 | 0,9952 | 0,7890 | 0,9169 | 0,6929 |

**H2 2024:** no hay propuestas standalone de H2 (la foto de H2 de la sesión principal solo tiene sub/gold), así que no se ha medido.
Como no se propone ningún cambio, la regla de las dos ventanas no se aplica. La comparativa previa del equipo, con ar_text antes de
ar_exact mejor en las dos ventanas (84,63/84,56 frente a 84,22), coincide con el orden actual. Se han usado 0 ejecuciones con candado
en esta fase.

## (d) Riesgos para FINAL

- **Código distinto:** la foto es de bf9a167. `integ/a-agents-b` añade ap_bulk, credit_note_payouts, y SHORT_BANK_CHARGES y
  DUPLICATE_PAYMENT dentro de ar_text. Las conclusiones estructurales (techo pequeño, orden óptimo) deberían mantenerse. Si se cambia
  el orden en esa rama, conviene repetir `eda/11_routing.py` con una foto nueva: 1 ejecución completa de ~13 min y 7,6 GB de RSS,
  más 80 s de análisis.
- **Coste de un árbitro:** ejecutar todos los resolvers sobre el pool completo cuesta lo mismo que la foto (~12 min frente a ~6 min
  del pipeline, y 7,6 GB). Es otro motivo para no integrarlo en FINAL.
- **Orden actual en FINAL:** no tiene riesgo especial; es el que ya se ha validado.

## (e) Para B/C

- **B (ar_text):** 1:1 entre facturas del mismo importe. Frente a exact_amount en `ar_party_subset` de 1 factura empata (290/269).
  FIFO acierta en `ar_same_amount_fifo` y el mes de servicio en `ar_same_amount_month`. Si ar_text usara la señal de mes de servicio
  de ar_exact (`_service_month`) cuando aparece en el texto, y FIFO solo si no aparece, ganaría como mucho unas 130 líneas (~+0,03).
- **B (ar_text):** `ar_partial_hint` acierta el 62,7 % (2.720 líneas en el pipeline, 1.015 mal con partidas en el golden). Cuando hay
  una factura abierta 1:1 de importe exacto, la PARTIAL suele ser falsa: exact gana 31 a 10.
- **B (ar_text):** `ar_party_subset` con 2 o más facturas frente a un 1:1 exacto pierde 35 a 5. Conviene preferir el 1:1 exacto antes
  que un subconjunto.
- **ar_multi y adjusted_amount:** casi empatan entre ellos (232/278, 106 sin acierto). ar_multi standalone tiene una precisión por
  línea del 85 % y `ar_exact` en el pipeline del 82,6 % (886 líneas mal). Son los dos resolvers con más FP.
- **Recall:** lo que de verdad se puede ganar está en las 18.241 líneas del pool sin ninguna propuesta (12.869 en cat_rest con
  partidas en el golden). No depende del enrutado.

Archivos: `eda/11_routing.py` (análisis reproducible). Salidas locales, que no se suben: `out/routing_analysis_h1.txt`,
`out/routing_h1_lines.parquet`, `out/routing_dig.py`, `out/routing_arbiter.py`.
