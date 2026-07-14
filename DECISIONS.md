## [M2-T2] Quiet third-party boundary
- Scelta: silenziare il logger Ultralytics sotto `ERROR` e gli avvisi nativi emessi soltanto durante l'import, ripristinando sempre `stderr`.
- Motivo: lo smoke test locale produceva output di terze parti nonostante `verbose=False`, violando il contratto di successo silenzioso e dei diagnostici esatti.
- Alternative scartate: configurare `task=detect`, vietato dalla specifica; modificare i pacchetti installati, fuori ambito; accettare stream non vuoti, contrario ai criteri di accettazione.
- Reversibile: sì, rimuovendo la gestione dell'import e il livello del logger in `inference.py`.

## [M2-T1] Blocker risolto dalla specifica aggiornata
- Scelta: sostituire i range Nano Banana ritirati con gli snapshot esatti approvati dal nuovo piano.
- Motivo: il planner ha trasformato i conteggi osservati in un contratto di regressione riproducibile, esplicitamente privo di significato di accuracy o ground truth.
- Alternative scartate: tuning della produzione o mantenimento simultaneo dei vecchi range, entrambi vietati dalla specifica aggiornata.
- Reversibile: sì, tramite una successiva modifica esplicita del piano e delle costanti del test.
