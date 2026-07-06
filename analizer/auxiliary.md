# Auxiliar — Visualización de predicciones sobre test

## ¿Cómo elegir qué 4 imágenes ver?

En la celda 8.5, cambia esta línea:

```python
for row, sample_idx in enumerate(range(n_vis)):
```

Por una lista de índices concretos (0–28, ya que hay 29 imágenes de test):

```python
for row, sample_idx in enumerate([5, 12, 18, 25]):   # ← los que quieras
```

---

## Código auxiliar — Genera todos los plots de 4 en 4

Pega este bloque en una celda nueva del notebook (después de haber entrenado
los modelos y cargado `ALL_MODELS`, `ds_test`, etc.).

Guarda cada grupo de 4 como `predictions_batch_00.png`, `predictions_batch_01.png`…
en `OUTPUT_DIR`. Al final imprime los índices de cada batch para que sepas
qué imagen corresponde a qué número.

```python
# ── Predicciones sobre TODOS los test, de 4 en 4 ──────────────────────────
import math

BATCH_SIZE_VIS = 4
n_test  = len(ds_test)
n_batch = math.ceil(n_test / BATCH_SIZE_VIS)
n_cols  = len(ALL_MODELS) + 2   # RGB | GT | modelo1 | modelo2 | ...

print(f'Test set: {n_test} imágenes → {n_batch} figuras de {BATCH_SIZE_VIS}')
print(f'Guardando en: {OUTPUT_DIR}\n')

for b in range(n_batch):
    idxs = list(range(b * BATCH_SIZE_VIS,
                      min((b + 1) * BATCH_SIZE_VIS, n_test)))
    n_rows = len(idxs)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for row, sample_idx in enumerate(idxs):
        sample = ds_test[sample_idx]
        img_t  = sample['image']
        gt_pan = sample['seg_id_map'].numpy()
        gt_seg = sample['segments_info']
        img_np, gt_color, _, gt_boxes = overlay_panoptic(img_t, gt_pan, gt_seg)

        axes[row][0].imshow(img_np)
        axes[row][0].set_title(f'RGB  (idx {sample_idx})', fontsize=8)
        axes[row][0].axis('off')

        axes[row][1].imshow(gt_color)
        draw_thing_boxes(axes[row][1], gt_boxes)
        axes[row][1].set_title('GT Panoptico', fontsize=8)
        axes[row][1].axis('off')

        for col, (mname, model) in enumerate(ALL_MODELS.items(), start=2):
            pred_pan, pred_segs = model.predict_panoptic(
                img_t.unsqueeze(0).to(DEVICE))
            _, pred_color, _, pred_boxes = overlay_panoptic(
                img_t, pred_pan, pred_segs)
            axes[row][col].imshow(pred_color)
            draw_thing_boxes(axes[row][col], pred_boxes)
            axes[row][col].set_title(mname, fontsize=8)
            axes[row][col].axis('off')

    # Leyenda clases (colores base)
    patches = [mpatches.Patch(color=np.array(CAT_COLORS[c]) / 255,
                              label=CAT_NAMES[c])
               for c in sorted(CAT_NAMES)]
    fig.legend(handles=patches, loc='lower center', ncol=len(CATEGORIES),
               fontsize=8, framealpha=0.8, bbox_to_anchor=(0.5, -0.02))

    title = (f'Batch {b+1}/{n_batch}  —  imgs test {idxs[0]}–{idxs[-1]}\n'
             'Thing: tonos por instancia + bbox ID  |  Stuff: color plano')
    plt.suptitle(title, fontsize=10)
    plt.tight_layout()

    fname = OUTPUT_DIR / f'predictions_batch_{b:02d}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f'  Batch {b+1:02d}: imgs {idxs} → {fname.name}')

print('\nListo. Revisa OUTPUT_DIR:')
print(OUTPUT_DIR)
```
