# Instrucciones de la demostración

Guía para preparar y grabar la demo de forma reproducible. El guion hablado está en `DOC/Guion.md`.

## 1. Antes de grabar (10 minutos)

1. **Libera la GPU y la CPU.** En el portátil de desarrollo, Lively Wallpaper (fondo animado con mpv) y Godot
   estaban usando la GPU y bastante CPU, lo que hacía todo mucho más lento. Ciérralos antes de grabar.
2. **Comprueba Ollama y los modelos:**
   ```bash
   ollama list        # deben aparecer qwen2.5:3b, gemma3:4b y nomic-embed-text
   ```
   Si falta alguno: `ollama pull <modelo>`.
3. **Configuración local.** El archivo `.env` (no versionado) contiene `VISION_NUM_GPU=0`, que ejecuta el modelo
   de visión en CPU. En este equipo es más rápido que repartirlo entre GPU y CPU (~50 s por foto frente a
   >10 min). Si cierras las otras aplicaciones y la GPU queda libre, puedes probar a vaciar ese valor.
4. **Arranca la aplicación** (desde la raíz del proyecto):
   ```bash
   .venv\Scripts\python run.py
   ```
   Abre http://localhost:8000. La cabecera debe decir en verde *Ollama OK · qwen2.5:3b · vision gemma3:4b ·
   nomic-embed-text · coordinator: llm*.
5. **Calienta los modelos y prepara el segundo caso.** En la UI ejecuta una vez *Inconsistent evidence (photo
   shows no damage, invoice predates incident)* y espera a que termine (~1 min). Así:
   - los tres modelos quedan cargados en memoria (`keep_alive` de 30 min), y
   - el caso CLM-DEMO-003 aparece en *Previous runs* para la parte del guion que lo muestra.
6. Pulsa **New analysis** para dejar el formulario vacío.
7. Prepara la grabación: navegador a pantalla completa con zoom al 100 % o 110 %, notificaciones desactivadas
   y el guion en otra pantalla. En Windows puedes grabar con la Barra de juego (`Win + Alt + R`) o con OBS Studio.

## 2. Caso principal que hay que usar

**Ceiling leak around a light fitting (main demo, multimodal)**, en la carpeta `data/demo/01_ceiling_leak/`:
- `description.txt` (se rellena sola), `ceiling_stain.jpg` (imagen sintética dibujada) y `repair_estimate.pdf`
  (presupuesto ficticio de 4.350 €).
- Metadatos: CLM-DEMO-001, póliza HOME-55821, 2026-10-08, apartamento, Madrid, 4.350 €.

Se eligió porque usa los tres tipos de entrada, activa un riesgo de seguridad (agua en una lámpara) y tiene
precedentes muy claros en la base de conocimiento.

## 3. Pasos durante la grabación

1. En **Load a demo case…** elige *Ceiling leak around a light fitting*. El formulario se rellena y la foto y el
   PDF se adjuntan como archivos reales; se ven como etiquetas bajo los campos de archivo.
2. Pulsa **Run triage**. Aparece la sección *2 · ReAct execution trace*, que se actualiza cada 1,5 s con los
   pasos reales del backend (con el tiempo transcurrido real, no una barra de progreso inventada).
3. Pasos esperados, con la configuración actual (el orden lo decide el LLM y puede variar entre ejecuciones):
   `analyze_description` → `analyze_image` → `extract_document` → `search_similar_claims` →
   `cross_check_evidence` → `assess_claim` → `finish`. Duración medida: 30–70 s.
4. **Cómo mostrar el bucle ReAct:** en cada fila señala la acción, el agente, la etiqueta *decided by* (`llm`,
   `rules` cuando solo había una acción posible, o `fallback_rules`), la línea *Reason* y la observación.
5. **Cómo mostrar los casos recuperados:** baja hasta *Similar historical claims*. Lo esperado es HC-2024-0040 en
   primera posición (similitud ≈ 0,91) y también HC-2023-0003, con la columna *Why similar* y, debajo,
   *Historical insights*.
6. **Cómo mostrar el informe final:** arriba del informe, la insignia **P1 Critical**, el tipo de daño
   *water leak*, la confianza (*high*, heurística) y el recuadro *Human review recommended*. Después,
   *Priority factors*, *Facts and their sources*, *Evidence consistency checks* (lo esperado: todo *pass* salvo
   *photo authenticity = warn*, porque la imagen es un dibujo), las recomendaciones y las incertidumbres.
7. **Segundo caso:** despliega *Previous runs* y haz clic en **CLM-DEMO-003**. Muestra que se recomienda revisión
   humana: la foto no muestra daño (*fail*), el importe reclamado (6.000 €) no coincide con la factura (2.100 €),
   la factura es anterior al siniestro y el precedente HC-2024-0032 tiene la alerta *fraud indicators*.
8. Opcional: **Download JSON** para enseñar que el informe es estructurado.

## 4. Resultados que deberían aparecer (medidos)

| Caso | Prioridad | Tipo | Revisión humana | Precedente esperado | Tiempo |
|---|---|---|---|---|---|
| 01 Ceiling leak | P1 | water_leak | sí | HC-2024-0040 / HC-2023-0003 | ~31–69 s |
| 02 Storm roof | P1 | storm_wind | sí | HC-2023-0010 / HC-2024-0034 | ~31–58 s |
| 03 Inconsistent | P3 | water_leak | sí (varios avisos, confianza *low*) | HC-2024-0032 | ~25–50 s |
| 04 Minor, text only | P3–P4 | water_leak | según la ejecución | HC-2023-0006 | ~6 s |

Los modelos son pequeños y la temperatura es 0, pero los resultados pueden variar ligeramente entre ejecuciones
(por ejemplo, qwen a veces marca el daño como *en curso*). La prioridad y las comprobaciones siempre muestran su
origen, así que cualquier diferencia se puede explicar.

## 5. Si algo falla durante la grabación

| Problema | Qué hacer |
|---|---|
| Cabecera en rojo (*Ollama unreachable*) | Abre la aplicación de Ollama o ejecuta `ollama serve` y recarga la página. |
| La foto tarda demasiado (> 2 min) | Comprueba que `.env` tiene `VISION_NUM_GPU=0` y que Godot o Lively están cerrados. El tiempo de espera es de 600 s; si se supera, el informe se genera igualmente y registra la limitación. |
| El LLM falla o responde mal | El coordinador pasa a `fallback_rules` y el informe usa la plantilla (`narrative_source = template_fallback`). Es un comportamiento esperado y se puede explicar. |
| Quieres una demo más rápida | Arranca con `COORDINATOR_MODE=rules` (planificador determinista; el mismo bucle sin decisiones del LLM) o usa el caso 04, que no tiene foto. |
| No hay tiempo para ejecutar en directo | En *Previous runs* abre una ejecución guardada (es real; se guardan en `output/runs/`). También: `python scripts/run_case.py data/demo/01_ceiling_leak` en una terminal. |
| Quieres enseñar la validación de entradas | Pulsa **Run triage** con la descripción vacía, o adjunta `data/demo/negative_inputs/unsupported.docx`: aparece un error en rojo con la lista de problemas. |

## 6. Comprobaciones finales antes de entregar

- `.venv\Scripts\python -m pytest` → 52 pruebas superadas.
- Vídeo en inglés, de 5 minutos como máximo, grabado por ti con tu voz (no está grabado: lo tienes que hacer tú).
- Nombres de entrega exigidos por Bain:
  - Código: `Rodrigo_<Apellido>_<País>_Solution` (zip del repositorio sin `.venv`, `output/runs`,
    `output/uploads` ni `data/index`; el `.gitignore` ya los excluye).
  - Vídeo: `Rodrigo_<Apellido>_<País>_Video_Presentation` (por ejemplo, `.mp4`).
  - Sustituye `<Apellido>` y `<País>` por tus datos (el ejemplo de Bain es `Mary_Jane_Madrid_Solution`).
- Para generar el zip: primero haz commit de todos los cambios y después ejecuta
  `git archive --format=zip -o Rodrigo_<Apellido>_<País>_Solution.zip HEAD` (solo incluye los archivos versionados).
