# A·9: preparación de FINAL (ensayo completo con integ/a-agents-b)

Rama `agentA/9-final-rehearsal` (base `origin/integ/a-agents-b` 53f5256). Los cambios de la primera sesión
(prepare_final.sh endurecido, `00_to_parquet.py --phase`, `12_drift_probe.py`, `tests/test_final_preparation.py`)
ya estaban en integ y eran idénticos a los locales, así que los descarté. Aquí solo hay cosas nuevas:

- `tools/prepare_final.sh`: un único candado global, re-ejecución segura, extracción a prueba de montajes de ensayo, comprobación de cobertura y comparación con LIVE.
- `tools/heavy.sh`: respeta `HEAVY_LOCK_HELD`, y `BANKREC_LOCK_DIR` permite probarlo en un sandbox.
- `src/bankrec/pipeline.py`: `--repair`.
- `src/bankrec/categories.py`: si falta el modelo en FINAL, el error es fatal.
- `tests/test_final_repair.py`: 3 tests. La suite completa pasa: 70 tests OK.

## (a) Qué puede romperse a las 18:00, por gravedad

| # | Gravedad | Riesgo | Estado |
|---|---|---|---|
| 1 | P0 | **`ar_text` tarda ahora 15-17 min, no 4,5.** Con integ 53f5256 mide 606 s en TRAIN H1 (log de la sesión principal), **1.019 s en LIVE** (sesión principal) y **916 s en este ensayo**. Antes de los últimos cambios de B eran 377 s (A·10, H2). El pipeline completo tarda 19-21 min. FINAL tiene más líneas y pagos más lentos: hay que contar con **20-25 min**, y un segundo intento completo ya no cabe con holgura. | Checklist ajustada: lanzar a las 18:00, con paracaídas `BANKREC_SKIP=ar_text` en paralelo. Para B: bisecar (sospechoso: a1cb2e5). |
| 2 | P0 | **Otro proceso con el candado global.** En este ensayo el runbook esperó **32 min** (13:57→14:29) a que acabaran las ejecuciones de la sesión principal. Antes cada paso pedía el candado por separado y otro proceso podía colarse entre parquet, parseo y pipeline. | **Hecho:** prepare_final.sh toma el candado **una vez**, después de extraer, y lo mantiene hasta el final. heavy.sh ve `HEAVY_LOCK_HELD` y no vuelve a bloquear. Probado con contención en un sandbox con candado privado. Además, parar todos los agentes a las 17:50. |
| 3 | P0 | **Validación fallida = no hay fichero.** El pipeline sale con código 1 sin escribir nada, así que un solo fallo de formato nos deja sin envío. | **Hecho:** `--repair` descarta solo las filas culpables: fuera de fase, item_id inválido, importe inválido, par duplicado, fila sin item en una línea con items, categoría o excepción desconocida. Después da la fila fallback a las líneas que se quedan sin fila y vuelve a validar. Se activa con `BANKREC_PIPELINE_ARGS="--safe --repair"`. Tiene tests. Sobre un fichero válido no cambia nada (test). |
| 4 | P0 | **Modelo de categorías ausente.** `strong_categories` y `all_categories` devolvían 0 filas en silencio, y todo caía al fallback (BANK_FEES/UNIDENTIFIED). | **Hecho:** en `final` es `FileNotFoundError`; en las demás fases sale un WARNING visible. |
| 5 | P0 | **Extraer el tarball real sobre un montaje de ensayo** (`data/final` con enlaces a `data/live`): tar escribiría a través de los enlaces. | **Hecho:** el runbook aborta si `data/final` contiene enlaces (probado en el sandbox). |
| 6 | P1 | **Tablas o cachés de un ensayo que sobreviven** en `data/parquet/final`. Antes solo se borraban 2 cachés; si FINAL no trae un fichero que el ensayo sí tenía, su parquet viejo se quedaba. | **Hecho:** al descargar se borra `data/parquet/final` entero (se niega si es un enlace). |
| 7 | P1 | **Re-ejecutar tras un fallo** volvía a descargar, y `curl -C -` sobre un fichero completo puede fallar y cortar el script por `set -e`. | **Hecho:** si el tarball existe y coincide con el SHA256 dado, se salta la descarga. Alternativa: `--skip-download`. |
| 8 | P1 | **Fichero viejo con apariencia de READY:** si el pipeline fallaba, quedaba el `out/final.csv.gz` anterior. | **Hecho:** se borra antes del pipeline. |
| 9 | P1 | **Basura macOS en la raíz.** El tarball de LIVE trae `._live` en el nivel superior, además de `._*` junto a cada fichero y carpeta y avisos `LIBARCHIVE.xattr`. Solo se limpiaba `data/final`. | **Hecho:** se borran también los `data/._*`, se filtran los avisos, se lista el nivel superior del tarball y se imprime `junk left: 0`. |
| 10 | P1 | **Revisión final incompleta:** no se comprobaba la cobertura de FINAL ni se comparaba con LIVE. | **Hecho:** el runbook falla si falta alguna línea, e imprime el reparto por línea y las excepciones junto a `BANKREC_REF` (por defecto `out/live.csv.gz`), más las líneas por entidad. |
| 11 | P1 | **Salida no determinista (~0,3 % de líneas).** El ensayo (LIVE disfrazado de FINAL) y la ejecución LIVE de la sesión principal, con el mismo código y el mismo modelo, difieren en **548 líneas de 203.083**. Son casi todas desempates: mismo importe y otra factura, sobre todo en US01 (406). Causa probable: `data.load` usa `unique("item_id")` sin `maintain_order`, que cambia el orden de `ctx.items` entre ejecuciones y fases, y los resolvers desempatan por orden de fila. | Propuesto, sin aplicar porque no he podido verificarlo en las dos ventanas de TRAIN: `maintain_order=True` + `sort("item_id")` en `data.load`, y medir H1/H2. No rompe FINAL, pero cualquier diferencia de ±0,3 % entre dos ejecuciones es ruido. |
| 12 | P1 | **Entrenar `--all` a las 18:xx** compite por CPU, RAM y tiempo, y no depende de los datos de FINAL. | Propuesto: entrenarlo antes de las 18:00 y lanzar el runbook con `--skip-train`. `models/categories.joblib` no tiene metadata: no consta que sea `--all`. |
| 13 | P2 | `data/masters/fx_rates.csv` acaba el 2025-12-31 (ZAR incluido). Si `masters_final` no trae FX de 2026, se usa el último tipo (no rompe). | Comprobarlo a las 18:04. |
| 14 | P2 | Tarball con una estructura distinta de `final/`. | El runbook para con "Need a real data/final directory"; la lista del nivel superior dice qué mover. |

## (b) Supuestos de fase revisados en el código

- `data._read_master`: cada maestro se lee de `data/final/masters_final/<f>`, con fallback fichero a fichero a `data/masters/`. En el ensayo se leyeron los 7 desde `masters_final`.
- `data.load('final')`: carga el arrastre de `open_items_at_start` enriquecido con las tablas de TRAIN+LIVE (`PHASES[:index]`) más las partidas del propio FINAL. En el ensayo da **1.031.576 partidas, las mismas que `load('live')`**, que es lo esperado porque el falso FINAL son los datos de LIVE.
- `deterministic._acc_entity`: la entidad es el prefijo de `account_id`. Cubre `ZA01-UMVC-ZAR`, que ya aparece en LIVE con 2 líneas, y la segunda cuenta de IT01. `PAYFACTORY` solo se aplica a ES00, que es la regla de POLICIES §5.
- La regex de item_id `^(ARI|ARC|API|APC)-\d{4}-\d{7}$` deja libre el año. Los documentos `KWI…` no pasan por ella y `norm_expr` los conserva.
- Fuera de deterministic no hay entidades ni cuentas fijas en B/C. `customer_matcher._cap_cents` devuelve 0 si falta la divisa en FX.
- Cachés por fase: `data/parquet/<fase>/remittance_lines.parquet` y `pdf_clauses.parquet`, sin versión del parser. `ra_parsed.parquet` ya no lo lee nadie. `ar_text._credit_note_reasons` lee `credit_notes` de todas las fases (`PQ/*`).
- `src_file` en los parquet de lotes es una ruta absoluta, pero ningún módulo lo usa.

## (c) Resultados del ensayo

Montaje: `data/final/` real con enlaces a `data/live/*` y `masters_final/` con copias de `data/masters/*.csv`. `data/parquet/` también es real, con `train/live/sample` como enlaces de solo lectura; `final/` se regenera desde cero.
Comando: `PYTHON_BIN=… BANKREC_CAT_MODEL=/home/liame/claude_hackathon/models/categories.joblib BANKREC_REF=/home/liame/integ_b/out/live_v8_agents.csv.gz /usr/bin/time -v bash tools/prepare_final.sh --skip-download --skip-train`.
Comprobé que los 28 parquet compartidos de train/live tienen el mismo tamaño y mtime antes y después.

| Paso | Tiempo |
|---|---|
| Espera del candado (otras sesiones) | 1.905 s (32 min), solo en el ensayo |
| Parquet FINAL (13 tablas, 444 ficheros de lotes) | 11 s |
| Parseo de PDFs (9.499 → 4.194 cláusulas, 32 s) + avisos | 56 s |
| Pipeline (`total` interno 1.133,7 s) | 1.143 s (19 min). Carga 10,9 s; **ar_text 915,8 s**; strong_categories 40 s; multi_invoice 34 s; resto < 3 s por módulo |
| Tamaño y distribuciones | 1 s |
| **Total sin la espera** | **1.211 s ≈ 20 min** |

- **Pico de RAM**: 6,44 GB (`/usr/bin/time -v`, máximo de los descendientes). El `--profile` del pipeline da un pico de 6,29 GB.
- **VALIDATION: OK**. 678.301 filas, 203.083/203.083 líneas cubiertas, 31,9 MB sin comprimir (7,2 MB en gz): muy por debajo de los límites de 60 MB y 3 M filas. SHA256 `5ab4376d…`.
- **Frente a LIVE con el mismo código** (`live_v8_agents.csv.gz`, 678.270 filas): los módulos hasta `bulk_mixed_runs` dan cifras idénticas. Las diferencias empiezan en `ar_text` (80.145 frente a 80.123 líneas) y son desempates (ver riesgo 11). Reparto por línea: `<ITEMS>` 83,97 % en los dos; UNIDENTIFIED 17.304 frente a 17.300; las demás categorías, idénticas. Excepciones: diferencias de 0-9 filas (OVERPAYMENT 353 frente a 344). Ninguna diferencia viene de la fase: el resultado es el mismo que en LIVE salvo el ruido de desempate.
- **Drift**: el ensayo no puede simular ZA01/KWI ni a los clientes renombrados, porque reutiliza LIVE. La sensibilidad del clasificador ya está medida en `eda/12_drift_probe.py` (primera sesión, en integ): con ENT_ZA01 cambia el 0,23 % de las predicciones y con renombres el 2,6 %, y casi nunca cambia un reclamo fuerte. Queda pendiente un entrenamiento que deje fuera una entidad entera: no hubo turno de candado.

## (d) Checklist 18:00-19:30 (hora de Madrid)

Todo en la copia principal (`cd /home/liame/claude_hackathon`), con la rama de integración congelada.

**Antes de las 18:00**
- 17:00 Congelar el código: `git checkout <rama final> && git pull && git log -1` (anotar el SHA). Tests: `cd src && ../.venv/bin/python -m unittest discover -s ../tests`.
- 17:05 Clasificador de producción (no depende de FINAL):
  `cd src && BANKREC_CAT_MODEL=$PWD/../out/final_categories.joblib ../tools/heavy.sh quick ../.venv/bin/python -m bankrec.categories --train --all`
  y comprobar `cat ../out/final_categories.joblib.metadata.json` → `"all_years": true`.
- 17:20 Última ejecución de LIVE con ese código y ese modelo, a `out/live.csv.gz` (~20 min). Es la referencia de distribuciones que usa el runbook.
- 17:50 **Parar todos los agentes y ejecuciones pesadas.** `cat /tmp/bankrec_locks/holder` debe salir vacío y `free -g` debe dar ≥ 10 GB disponibles.
- 17:55 `ls data/final data/parquet/final` → no debe existir ninguno (si existen, moverlos a `*.old`). `read -rs HK_TOKEN && export HK_TOKEN`. Tener abierto https://hackathon.usetalky.com/config.json para el SHA256.

**18:00-18:07 · descarga y preparación (en el ensayo: parquet 11 s, parseo 56 s)**
- 18:00 Copiar el SHA256 de FINAL de `config.json` y lanzar:
  `BANKREC_CAT_MODEL=$PWD/out/final_categories.joblib /usr/bin/time -v bash tools/prepare_final.sh <SHA256> --skip-train 2>&1 | tee out/final_run.log`
- 18:02 Descarga (~90-100 MB; LIVE pesaba 87 MB) y `OK` del SHA. Si la red falla, relanzar el mismo comando: reanuda, y si el tarball ya coincide se salta la descarga.
- 18:03 En la lista del nivel superior debe salir `final`. Deben salir `got the global lock` al instante y `junk left: 0 … masters_final: …`. Si el candado espera, alguien no ha parado: `cat /tmp/bankrec_locks/holder`.
- 18:04 En otra terminal: `tail -1 data/final/masters_final/fx_rates.csv; grep ZA01 data/final/masters_final/*.csv | head`
- 18:05 Parquet y parseo de PDFs y avisos (~1-2 min).

**18:07-18:30 · pipeline (19 min en el ensayo; contar 20-25 en FINAL)**
- 18:07 Debe salir `loaded final: ~220k bank lines`. Vigilar `PROFILE … status=ok`.
- 18:08 **Paracaídas en paralelo**, en otra terminal, SIN heavy.sh (~2-3 min, ~4 GB; cabe en memoria junto al pipeline si todo lo demás está parado):
  `cd src && BANKREC_SKIP=ar_text BANKREC_CAT_MODEL=$PWD/../out/final_categories.joblib ../.venv/bin/python -m bankrec.pipeline --phase final --safe --repair --out ../out/final_noartext.csv.gz`
  Así a las ~18:12 ya hay un FINAL válido, con peor nota.
- ~18:28 Deben salir `VALIDATION: OK`, `wrote out/final.csv.gz`, la tabla junto a LIVE, el `sha256sum` y `READY`.

**18:30-18:45 · revisión**
- `<ITEMS>` y cada categoría dentro de ±3 puntos de LIVE; excepciones del mismo orden de magnitud; líneas ZA01 presentes.
- Si el reparto es absurdo (casi todo UNIDENTIFIED/BANK_FEES): buscar `WARNING` o `status=failed` en el log y mirar el `pending` tras `ar_text`.
- 18:45 Decisión de envío. Anotar el `sha256sum out/final.csv.gz` y el SHA del código.

**18:45-19:00 · envío único**
- `.venv/bin/python tools/submit.py FINAL out/final.csv.gz --i-am-sure -m "FINAL <git sha> <modelo>"`
- Debe acabar en `received` o `scored`; la nota está oculta hasta las 20:30. Si sale `invalid`, se puede corregir y volver a subir.
- Commit de `submissions/log.jsonl`.

**Planes B (margen hasta 19:30)**
- Si un módulo lanza una excepción: relanzar solo el pipeline con `--safe --repair` (~20 min; decidirlo antes de las 18:50):
  `cd src && BANKREC_CAT_MODEL=$PWD/../out/final_categories.joblib ../.venv/bin/python -m bankrec.pipeline --phase final --profile --safe --repair --out ../out/final.csv.gz`
  También vale `BANKREC_PIPELINE_ARGS="--safe --repair" bash tools/prepare_final.sh --skip-download --skip-train`.
- Si a las 19:05 no hay un `out/final.csv.gz` válido: enviar `out/final_noartext.csv.gz`. **Nunca** llegar a las 19:30 sin haber enviado.

## (e) Para B/C

- **B, tiempo de `ar_text`:** 377 s (H2, A·10) → 606 s (H1) → 916-1.019 s (LIVE) con integ 53f5256. Es el 80 % del pipeline y hace que no quepa un segundo intento completo en FINAL. Sospechoso: a1cb2e5 (índice de sufijos de 6 dígitos y citas abreviadas). Pedimos bisecar, o un tope de candidatos o de tiempo por línea.
- **B, desempates:** 548 líneas (0,27 %, 406 de US01) cambian entre dos ejecuciones con los mismos datos. Conviene desempatar por una clave estable (`item_id`, fecha) y no por el orden de las filas o de un `set`.
- **B:** `ar_text._entity_scope` lee `data/masters/entities.csv` directamente, en vez de `masters_final`. ZA01 entra igualmente por la unión con `ctx.bank.entity`, pero un rol nuevo en `masters_final` no se vería. Arreglo: `data._read_master("entities.csv", ctx.phase)`.
- **C:** `remittance_parse.load_lines` y `pdf_clauses.parse_phase` cachean sin versión del parser. En FINAL no afecta, porque el runbook regenera `data/parquet/final`, pero en TRAIN/LIVE puede quedar una caché de un parser anterior. Propuesta: guardar en el parquet un hash del parser.
- **C:** la ventana aviso−banco es [-5, 20] días. Con pagos más lentos en 2026, comprobar que mide el desfase del aviso y no el vencimiento.
