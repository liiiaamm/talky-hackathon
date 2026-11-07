# talky-hackathon · Close the Books

Conciliación bancaria a escala (Grupo Valtermia, sintético). Para cada línea bancaria: qué partidas abiertas liquida y por qué
importe, o qué es (comisión, nómina, cash pool…), con su excepción (PARTIAL, SHORT_DISCOUNT, FX_DIFFERENCE…).

Score = 100 × (0,5·F1 de asignaciones + 0,3·accuracy por línea + 0,2·macro-F1 de excepciones).
Contexto y estrategia en [PLAN.md](PLAN.md) y [EDA.md](EDA.md). Las reglas de etiquetado están en [data/POLICIES.md](data/POLICIES.md)
y el formato de entrega en [data/FORMAT.md](data/FORMAT.md).

## Setup (una vez)

```bash
git clone https://github.com/liiiaamm/talky-hackathon.git && cd talky-hackathon
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Datos (no van al repo). Se descargan de la web y se extraen dentro de data/
mkdir -p data/_dl && cd data/_dl
curl -LO https://hackathon.usetalky.com/data/v1/bankrec-docs-and-masters.zip
curl -LO https://hackathon.usetalky.com/data/v1/bankrec-train-lite.tar.gz
curl -LO https://hackathon.usetalky.com/data/v1/bankrec-live.tar.gz
cd .. && unzip -o _dl/bankrec-docs-and-masters.zip && tar xzf _dl/bankrec-train-lite.tar.gz && tar xzf _dl/bankrec-live.tar.gz
cd ..
# Debe quedar: data/masters, data/train, data/live, data/sample… (si el zip crea una subcarpeta, mueve su contenido a data/)

python eda/00_to_parquet.py        # csv.gz -> data/parquet/{sample,train,live}  (unos minutos, una sola vez)
cd src && python -m bankrec.categories --train --all && cd ..   # clasificador de categorías -> models/ (no va al repo)
```

## Bucle de trabajo

```bash
cd src
# Iterar rápido: 1 mes con golden (segundos)
python -m bankrec.pipeline --phase sample --score
# Medir de verdad: medio año de TRAIN (~1,5 min, desglose por segmento y por módulo)
python -m bankrec.pipeline --phase train --from 2025-01-01 --to 2025-06-30 --score
# Generar el fichero LIVE (unos 12 s, valida el formato)
python -m bankrec.pipeline --phase live --out ../out/live.csv.gz
```

`src/bankrec/score.py` da **el mismo número** que el evaluador oficial (`data/starter/bankrec_score.py`). Comprobado en `sample`:
35,84 con los dos.

**Baseline actual** (módulos deterministas: remesas DD, tarjetas, cheques, payment runs y devoluciones):

| Conjunto | Score | F1 (P / R) | TxnAcc | ExcF1 |
|---|---|---|---|---|
| sample (1 mes) | 35,84 | 0,48 (0,999 / 0,32) | 0,30 | 0,13 |
| TRAIN 2025-01..06 | 42,39 | 0,54 (0,999 / 0,37) | 0,40 | 0,17 |

La precisión ya es casi 1. **El hueco es el recall**: unas 140k de las 203k líneas de LIVE caen en el `fallback`. Todo el lado AR
(transferencias con el nombre truncado, 1:N, parciales, descuentos, remittances) y las categorías no-item están por hacer.

## Cómo añadir un módulo

El pipeline ([src/bankrec/pipeline.py](src/bankrec/pipeline.py)) ejecuta *resolvers* en orden. **El primero que reclama una línea bancaria se la
queda**, y las líneas que nadie reclama van al `fallback` (créditos → `UNIDENTIFIED`, cargos → `BANK_FEES`).

1. Crea `src/bankrec/<tu_modulo>.py` con una o más funciones `f(ctx, pending) -> pl.DataFrame`:
   - `ctx`: datos de la fase (`ctx.bank`, `ctx.items`, `ctx.customers`… ver [data.py](src/bankrec/data.py)).
   - `pending`: las líneas bancarias que todavía no se ha quedado nadie.
   - Devuelve filas `bank_txn_id, item_id, amount, category, exception, module`. Usa `deterministic.mk_rows(...)` para las filas de
     partida.
2. Expón `RESOLVERS = [f1, f2]` en tu módulo. `remittance` y `ar_text` ya se cargan solos si existen. Para un nombre nuevo, añádelo a
   la lista `for _name in (...)` de `pipeline.py`.
3. Reclama solo las líneas de las que estés seguro: precisión antes que recall. Una línea mal asignada te quita a ti el punto y
   también se lo quita a cualquier módulo posterior que la habría resuelto bien.

## Git

- `main` siempre tiene que correr y validar. Trabaja en una rama (`feat/remittance`, `feat/ar-text`…) y abre un PR cuando tu módulo
  suba el score de TRAIN 2025-01..06. Pon el antes y el después en la descripción del PR.
- Nunca subas datos, modelos, `out/` ni el token. El `.gitignore` ya los excluye.

## Envíos

Solo los hace una persona (quien tenga el token del equipo):

```bash
read -rs HK_TOKEN && export HK_TOKEN          # nunca en código, .env ni historial
python tools/submit.py LIVE out/live.csv.gz -m "qué ha cambiado"
```

- `tools/submit.py` hace comprobaciones locales **antes** de pedir la URL de subida, porque pedirla ya gasta un hueco. Después sube el
  fichero, espera la nota y la apunta en `submissions/log.jsonl`. Haz commit de ese fichero.
- Límites: un envío cada 180 s y 20 al día por fase. En LIVE cuenta el mejor. Enviamos a LIVE solo cuando TRAIN mejore más de unos
  2 puntos.
- **FINAL: un único envío puntuado por equipo**, oculto hasta las 20:30. El script exige `--i-am-sure`. Los datos de FINAL salen a las
  18:00 y el envío cierra a las 19:30. Plan: enviar hacia las 18:45.
