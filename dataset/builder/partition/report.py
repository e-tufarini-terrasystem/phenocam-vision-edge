"""Render measured dataset and split statistics as a methodological report."""

from collections import Counter

from .allocation import PHASH_THRESHOLD, SEED
from .records import CLASS_NAMES


def split_statistics(assignments):
    result = {}
    for split, records in assignments.items():
        annotations, class_images = Counter(), Counter()
        for record in records:
            annotations.update(record["_classes"])
            class_images.update(record["_classes"].keys())
        result[split] = {
            "images": len(records),
            "positive_images": sum(bool(record["_classes"]) for record in records),
            "negative_images": sum(not record["_classes"] for record in records),
            "annotations": sum(annotations.values()),
            "class_annotations": {CLASS_NAMES[key]: annotations[key] for key in CLASS_NAMES},
            "class_images": {CLASS_NAMES[key]: class_images[key] for key in CLASS_NAMES},
            "sources": dict(sorted(Counter(record["source_dataset"] for record in records).items())),
            "sites": len({record["site_id"] for record in records if record["site_id"]}),
            "cameras": len({record["camera_id"] for record in records if record["camera_id"]}),
        }
    result["test"] = {}
    for field in ("images", "positive_images", "negative_images", "annotations"):
        result["test"][field] = result["test_id"][field] + result["test_ood"][field]
    result["test"]["class_annotations"] = {
        name: result["test_id"]["class_annotations"][name] + result["test_ood"]["class_annotations"][name]
        for name in CLASS_NAMES.values()
    }
    return result


def _split_table(statistics, total):
    labels = (("train", "Train"), ("val", "Validation"), ("test_id", "Test ID"), ("test_ood", "Test OOD"), ("test", "Test totale"))
    lines = ["| Split | Immagini | Annotazioni | % dataset | Positive | Negative | Classi | Siti | Camere |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, label in labels:
        row = statistics[key]
        classes = sum(value > 0 for value in row["class_annotations"].values())
        lines.append(f"| {label} | {row['images']} | {row['annotations']} | {row['images']/total:.2%} | {row['positive_images']} | {row['negative_images']} | {classes} | {row.get('sites', '—')} | {row.get('cameras', '—')} |")
    return "\n".join(lines)


def _class_table(statistics):
    lines = ["| Classe | Totale | Train | Validation | Test ID | Test OOD |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in CLASS_NAMES.values():
        values = [statistics[key]["class_annotations"][name] for key in ("train", "val", "test_id", "test_ood")]
        lines.append(f"| `{name}` | {sum(values)} | {values[0]} | {values[1]} | {values[2]} | {values[3]} |")
    return "\n".join(lines)


def render_report(inventory, phenocam, statistics, verification=None):
    total, annotations = inventory["images"], inventory["annotations"]
    class_lines = []
    for name in CLASS_NAMES.values():
        image_count = inventory["class_images"][name]
        count = inventory["class_annotations"][name]
        class_lines.append(f"| `{name}` | {image_count} | {image_count/total:.2%} | {count} | {count/annotations:.2%} |")
    camera_counts = sorted(phenocam["camera_distribution"].items(), key=lambda item: (-item[1], item[0]))
    top_cameras = ", ".join(f"`{name}` {count}" for name, count in camera_counts[:12])
    verify_note = "Non ancora eseguita." if not verification else f"Esito `{verification['status']}`: {verification['images']} immagini, {verification['annotations']} annotazioni, {verification['cross_split_phash_pairs']} coppie pHash cross-split."
    return f"""# Analisi e split del dataset v3

Data della misura: 2026-09-02. Il report distingue **misure** ottenute dai file, **documentazione** esterna e **raccomandazioni** progettuali. Nessun dato meteo è inferito dalle immagini.

## 1. Composizione originale misurata

L'artifact sorgente contiene **{total} immagini**, **{inventory['positive_images']} positive**, **{inventory['negative_images']} negative intenzionali senza file label**, e **{annotations} annotazioni**. Le immagini multi-oggetto sono {inventory['multi_object_images']}; la media è {inventory['annotations_per_image']:.3f} box per immagine e {inventory['annotations_per_positive_image']:.3f} per positiva. Sorgenti: {inventory['sources']}. Non risultano immagini/label mancanti o non elencate. I duplicati SHA-256 di file, pixel decodificati e file compilati sono tutti zero.

Formato: immagini JPEG e label YOLO, una riga `class x_center y_center width height`, coordinate normalizzate. Le negative omettono correttamente la label, come ammesso dalla [documentazione Ultralytics del formato detection](https://docs.ultralytics.com/datasets/detect/). La configurazione originaria aveva `train`, `operational_dev` e `operational_mining`, ma nessuna validation o test finale.

| Classe | Immagini con classe | % immagini | Annotazioni | % annotazioni |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(class_lines)}

## 2. Provenienza e PhenoCam misurate

Le sorgenti dichiarate dal manifest sono Open Images V7 ({inventory['sources'].get('open_images', 0)}), PhenoCam Network v3 ({inventory['sources'].get('phenocam', 0)}) e due camere operative private ({inventory['sources'].get('internal', 0)}). Solo le righe `source_dataset=phenocam` sono conteggiate come PhenoCam Network; le camere interne non vengono riclassificate per somiglianza.

PhenoCam comprende **{phenocam['images']} immagini**, **{phenocam['cameras']} camere** e **{phenocam['sites']} siti**, dal {phenocam['timestamp_min']} al {phenocam['timestamp_max']}. Massimo osservato: {camera_counts[0][1]} immagini per camera. Prime distribuzioni: {top_cameras}. Stagioni: {phenocam['seasons']}. Gli orari sono quelli dichiarati dalle camere, senza conversione di fuso.

Intervalli consecutivi entro la stessa camera: {phenocam['gaps_minutes']}. Le coppie pHash sono {phenocam['perceptual_pairs']}; a Hamming ≤ {PHASH_THRESHOLD} molte coppie cross-camera sono fotogrammi scuri o uniformi, quindi sono un segnale conservativo di ridondanza e non prova sufficiente per cancellare immagini. Distribuzione di luminanza media misurata su thumbnail: {phenocam['luminance_bins']}. Stagionalità e luminosità sono misurate; neve, nebbia, meteo e altre condizioni ambientali non hanno etichette verificabili. Gli sfondi ricorrenti sono controllati tramite identità di camera e pHash.

Gli script esistenti rilevanti sono `builder/phenocam.py`, `builder/dedup.py`, `builder/embeddings.py` e `builder/finalization/`; il manifest conserva sito, camera, timestamp, sequenza, sorgente, SHA-256, pHash e gruppo di deduplicazione.

## 3. Evidenza documentale

- **Ultralytics:** train/val/test possono essere directory o liste e le label usano coordinate YOLO normalizzate; una negativa può non avere `.txt` ([dataset detection](https://docs.ultralytics.com/datasets/detect/)). La validation è predefinita durante lo sviluppo, mentre un test dichiarato nel YAML si valuta esplicitamente con `split=test` ([model testing](https://docs.ultralytics.com/guides/model-testing)). Lo split deve precedere augmentation e preprocessing derivato ([preprocessing](https://docs.ultralytics.com/guides/preprocessing-annotated-data)).
- **Framework:** `StratifiedGroupKFold` tenta di preservare le classi mantenendo ogni gruppo in un solo fold; la documentazione avverte che la stratificazione è secondaria ai vincoli di gruppo ([scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html#stratifiedgroupkfold)).
- **Letteratura:** Roberts et al. raccomandano block cross-validation quando esistono dipendenze temporali, spaziali o gerarchiche ([Ecography 2017](https://www.wsl.ch/lud/biodiversity_events/papers/Roberts_et_al-2017-Ecography.pdf)). Kapoor e Narayanan richiedono che lo split rispetti le dipendenze nei dati ([Patterns 2023](https://doi.org/10.1016/j.patter.2023.100804)). WILDS separa ID e OOD e misura cali sostanziali sotto distribution shift ([PMLR 2021](https://proceedings.mlr.press/v139/koh21a.html)). Per camere fisse, *Recognition in Terra Incognita* valuta esplicitamente location mai viste perché lo sfondo cambia poco nella stessa camera ([ECCV 2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Beery_Recognition_in_Terra_ECCV_2018_paper.php)).

Queste fonti non prescrivono le percentuali adottate qui: percentuali, gruppi e scelta OOD seguenti sono raccomandazioni derivate dalla composizione misurata.

## 4. Strategie confrontate

| Strategia | Vantaggio | Limite/leakage | Applicabilità v3 |
| --- | --- | --- | --- |
| A — 70/15/15 random stratificato | più dati di evaluation | frame della stessa camera possono attraversare i confini | respinta |
| B — 80/10/10 random | più training | stesso leakage di A; le percentuali non lo risolvono | respinta senza grouping |
| C — sequenza/camera-day | blocca frame ravvicinati | lo stesso background resta in più split | insufficiente da sola |
| D — camera | blocca background e sequenze della camera | classi rare meno bilanciabili | adottata per PhenoCam |
| E — sito | misura nuovi siti | camera e sito coincidono nei metadati PhenoCam correnti | equivalente a D qui |
| F — temporale | simula il futuro | anni e siti sono confusi; Open Images non ha timestamp | non isolabile globalmente |
| G — camera+sito+tempo+classi | leakage basso e classi controllate | richiede ottimizzazione vincolata e due test | adottata |

L'unità minima è il gruppo di duplicazione per Open Images; per PhenoCam è l'intera camera/sito unita a ogni componente pHash≤{PHASH_THRESHOLD}; per i dati interni è il sito/camera. La priorità è non spezzare i gruppi, anche se la distribuzione non fosse perfetta.

## 5. Proposta applicata

Il pubblico (2.000 immagini) è diviso 80/10/10: 1.600 train, 200 validation e 200 TEST-ID. Le 240 immagini operative, provenienti da due camere e periodi mai presenti nel pubblico, formano TEST-OOD. Sul totale: train 71,43%, validation 8,93%, TEST-ID 8,93%, TEST-OOD 10,71%; il test complessivo è 19,64%.

Train serve soltanto ad apprendimento/augmentation; validation a tuning, early stopping e scelta modello; TEST-ID e TEST-OOD restano congelati e non guidano selezione, mining o aggiunta di immagini. TEST-ID mantiene le stesse sorgenti pubbliche ma camere PhenoCam disgiunte; TEST-OOD misura il dominio operativo successivo. Le classi rare sono bilanciate solo dopo i vincoli di gruppo; nessun gruppo viene spezzato per migliorare una percentuale.

{_split_table(statistics, total)}

{_class_table(statistics)}

## 6. Riproducibilità e verifica

Seed fisso: `{SEED}`. L'allocazione usa ottimizzazione intera deterministica sui gruppi indivisibili, con conteggi immagine esatti e deviazione di classi/stagioni/luminosità minimizzata. I file sono materializzati con hard link quando possibile, senza duplicare i byte delle immagini.

Rigenerazione da una copia valida dell'artifact: `cd dataset && .venv/bin/python -m builder.partition build dataset-v3 dataset-v3-rebuilt`. Verifica: `cd dataset && .venv/bin/python -m builder.partition verify dataset-v3`. Viewer: aprire `dataset/viewer.html`, scegliere `dataset-v3`, quindi selezionare Train, Validation, Test, Test ID o Test OOD.

Verifica corrente: {verify_note}
"""
