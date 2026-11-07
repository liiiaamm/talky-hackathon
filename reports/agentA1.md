# A·1 — Payment runs, confirming y FX

Estado: cambios locales revisables; no envío, commit, push ni PR. Se continuó desde bf9a167 tras el límite de Claude. Las mediciones completas de integración las coordina el agente principal.

## Hallazgos y cambios

1. **Agregados multimoneda: principal pérdida de pares de A.** El resolver sumaba `transfer_amount` de divisas distintas y exigía igualdad con el importe en moneda de la cuenta. `BT-2025-0009406`: adeudo USD 4.160.159,40; suma numérica de transferencias 4.152.530,93, con 374 transferencias USD y 8 EUR. El texto identifica `PR010125010963 382 TRANSFERS`; el golden tiene 984 partidas y el pipeline devolvía FX_DEAL. En H1 hay 43.151 pares perdidos en ap_fx; A7 confirma 40.086 en H2. La corrección acepta este enlace exclusivamente con cuenta + id completo de run + tipo aggregated + divisas múltiples + recuento explícito exacto + un único adeudo candidato de ese run. No se aplica el bypass a agregados monomoneda ni a dos débitos ambiguos.
2. **Divisa de la partida.** Tras el join, `currency` pertenece al payment run y `currency_i` a la factura. `_pr_exception` recibía la primera. Ahora recibe la moneda real de la factura. Se mantiene PARTIAL antes de FX_DIFFERENCE y los abonos APC sin excepción, conforme POLICIES §3.
3. **Confirming y vencimientos de fin de semana.** Dos grupos que vencen sábado y domingo pueden adeudarse juntos el lunes. Se añaden candidatos de vencimiento trasladado al lunes, corroborados por importe exacto y número de proveedores. Los grupos originales mantienen prioridad. Los vencimientos de `all_items` se usan como candidatos adicionales para documentos cerrados al inicio de ventana; no sustituyen a los grupos originales. Ejemplo: `BT-2025-0030761`, run CF010125012772, adeudo del lunes 2025-02-03.
4. **Truncados y duplicados no deterministas.** El l2 original ordenaba solo por e2e antes de elegir un banco por transferencia. Dos adeudos del mismo e2e podían alternarse según el orden del hash. Ahora desempata por e2e, booking_date y bank_txn_id. La política escoge el primer adeudo cronológico; no se infiere DUPLICATE_PAYMENT por el mero hecho de que open_amount sea cero.

## Medición

Se guardan en `out/` propuestas before/after de payment_runs para H1 2025 y H2 2024, y las salidas completas originales del coordinador permiten comparar integración. En inferencia las reglas usan exclusivamente texto y ficheros estructurados; golden se usa para diagnosticar y medir TRAIN durante el desarrollo.

**Los overlays iniciales 86,79/86,31 y combinado 86,94/86,46 están sustituidos y NO son resultados finales.** La primera simulación retenía líneas que el nuevo módulo dejaba de reclamar. Al detectarlo se preservaron los grupos confirming originales y se preparó un overlay conservador que identifica los casos sin propuesta secundaria como `unknown-requires-full`. Las cifras de esta versión y el pipeline completo se incorporarán tras la verificación coordinada.

La evaluación dirigida de la primera versión recuperaba netos 37.507 pares TP en H1 y 32.447 en H2, pero perdía respectivamente 43/32 pares antes correctos; parte venía del cambio de grupos confirming corregido y parte de desempates l2. Es evidencia de magnitud, no una métrica de aceptación de la versión final.

Seis regresiones estructurales pasan: agregado 100 EUR + 100 USD, FX por moneda de factura, rechazo de recuento erróneo, rechazo de débitos duplicados/monomoneda con suma discrepante y confirming sábado+domingo. Comando: `POLARS_MAX_THREADS=4 /home/liame/claude_hackathon/.venv/bin/python tests/test_payment_run_currency.py`.

Los snapshots originales usan `categories_to2024.joblib` (entrenado hasta diciembre de 2024); por tanto la cifra H2 anterior no es una evaluación temporal limpia del clasificador. La aceptación final se medirá por el coordinador con un modelo entrenado como máximo hasta 2024-06-30 y su baseline correspondiente. Los cambios de este informe son deterministas y no entrenan con golden.

## Riesgos para FINAL

El bypass exige la forma explícita `N TRANSFERS`, así que formatos nuevos sin recuento conservan el camino de suma exacta. No convierte sumas de diferentes divisas con un tipo supuesto. Los grupos weekend no incorporan calendarios de festivos nacionales. PAYFACTORY sigue limitado por POLICIES a ES00/ES01/ES02/PT01. Aún quedan casos de factoring con proveedor cedente oculto y referencias/documentos ambiguos, así como duplicados cuyo sufijo D desapareció al truncarse.

## Para B/C

No se cambiaron módulos B/C. Los agregados que antes caían en categorías pasarán a resolverse temprano, de acuerdo con los ficheros estructurados. La validación full debe confirmar que no cambia negativamente el pool de cobros posteriores.
