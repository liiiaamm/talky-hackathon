# Continuación de los nueve agentes de Claude

Se recuperaron los nueve worktrees `agentA/{1,2,3,4,5,6,7,9,10}` en `bf9a167`, sus instrucciones y sus historiales. Todos habían interrumpido la investigación por el límite de uso de Claude; no había cambios de código pendientes. A8 no formaba parte de los nueve.

Se continúan en tres grupos, conservando cada worktree y separando las ejecuciones que cargan datos con `tools/heavy.sh`. Los informes individuales se incorporarán aquí al terminar. No se envía ningún fichero a la plataforma.

La comparación histórica usa `categories_to2024.joblib` para reproducir la referencia original (85,14 en 2025-H1). Ese modelo **vio 2024-H2 durante el entrenamiento**. Se prepara además un modelo entrenado hasta junio de 2024 para la validación temporal sin esa fuga. No se atribuirá carácter fuera de muestra a las métricas históricas de 2024-H2.

El ledger permanece desactivado por defecto: la corrección de reapertura no elimina el sesgo de prioridad entre módulos ni la falta de historia inicial en TRAIN.
