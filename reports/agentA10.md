# A·10: ejecución robusta, perfil y salida reproducible

## Cambios

`pipeline.run(..., safe=False, profile=False)` mantiene el comportamiento de fallo inmediato por defecto. `--safe` captura excepciones de un resolver, imprime nombre y traceback, restaura `ctx.items`, `ctx.all_items` y el ledger cronológico `_ledger` y deja sus movimientos pendientes para los demás resolvers. La propiedad de líneas y la lista de resultados se actualizan después de que el resolver y el ledger terminen correctamente. No captura KeyboardInterrupt/SystemExit ni puede recuperarse de una terminación por el sistema. La restauración del ledger copia dict/list y comparte DataFrames inmutables, evitando duplicar gigabytes.

`--profile` imprime por resolver tiempo con `perf_counter`, RSS actual y máximo histórico del proceso. El máximo es acumulativo: no debe interpretarse como asignación exclusiva del módulo. Lectura `/proc/self/statm` en Linux y `resource.getrusage` para el máximo, sin dependencia adicional.

El writer ordena por las cinco columnas finales y fija `filename=""` y `mtime=0` en gzip. Nombres del archivo, instante de ejecución y orden de entrada ya no cambian el artefacto comprimido. Los importes y filas conservan su contenido.

El CLI termina con código 1 antes de puntuar/escribir si `validate` encuentra errores. El validador rechaza además importes NaN/inf en ítems e importes en filas sin ítem. Esta corrección es dependencia del runbook A9: una validación inválida no puede aparecer como READY.

## Evidencia de paridad

Se serializó el snapshot **completo de H1: 616.747 filas**, una vez con el writer original de HEAD y otra con el nuevo. Tras ordenar el CSV original, ambos tienen exactamente **28.995.908 bytes** y SHA256:

```
a3e6a3807870fe39de3f5c86165f04d65987482df9213723e1d501e87dc896ac
```

Dos salidas gzip nuevas, con nombres distintos y filas de entrada invertidas, son idénticas byte a byte:

```
1a5f5d16bd77a58e181d0ae2bd66963b613132c57b87e04d1b86be4537e90afe
```

Evidencia local: `out/writer_parity.json`. La prueba demuestra paridad de serialización; no demuestra por sí sola que todos los desempates internos de cada resolver sean deterministas en procesos distintos.

## Pruebas

Seis pruebas de `tests/test_pipeline_runtime.py`, todas correctas:

1. Fallo de resolver: conserva banco pendiente/saldos y deja continuar al siguiente, con aviso y perfil.
2. Modo predeterminado conserva fallo inmediato.
3. Fallo del ledger no reclama ni duplica líneas.
4. Fallo restaura eventos previos y elimina ledger creado dentro de un resolver fallido.
5. CLI no llama al writer después de errores de validación.
6. Mismo gzip con nombres y órdenes distintos.

```bash
PYTHONPATH=src POLARS_MAX_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  /home/liame/claude_hackathon/.venv/bin/python -m unittest discover -s tests -p test_pipeline_runtime.py -v
```

El integrador fusionó A10 con el ledger A5 y confirmó 35 pruebas correctas. No se ejecuta el scorer oficial, cuya ejecución no está autorizada por el entorno de esta sesión.

## Tiempo observado y límites

La ejecución completa baseline H2 del integrador tardó **444,12 s**, de los cuales `ar_text` consumió **377,0 s** y categorías fuertes **33,9 s**. `ar_text` es aproximadamente el 84,9% del tiempo total. Se conserva su lógica al pertenecer a B. La instrumentación nueva permite medir RSS y pico para las ejecuciones de integración sin lanzar otras completas redundantes. No atribuyo una mejora de tiempo ni un pico de memoria sin medición correspondiente.

`out/live_v7_jack_pdf.csv.gz`: **628.790 filas, 29.548.078 bytes sin comprimir**. Frente a límites conservadores de 3.000.000 filas y 60.000.000 bytes quedan 2.371.210 filas y 30.451.922 bytes. Con 10% más datos, tamaño estimado 32.502.886 bytes y 691.669 filas; se comprueba el tamaño real en A9 antes de declarar READY. Evidencia `out/size_margin.json`.

## Para B/C y riesgos FINAL

B: priorizar `ar_text` y su construcción de ItemStore/PartyResolver, sin cambiar resultados; evitar dos contextos grandes vivos en el mismo proceso. C: parseo inicial de remesas lanza por defecto tantos procesos como CPU; A9 prepara explícitamente FINAL con cuatro workers para limitar la memoria. Los cachés por fase deben invalidarse al reemplazar datos brutos. `--safe` es recuperación visible de emergencia, no una mejora de score, y el runbook FINAL utiliza por defecto fallo inmediato.

No se han enviado resultados a la plataforma ni publicado datos/modelos/out.
