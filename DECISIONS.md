## [M0-T4] BLOCCATO
- Motivo: la suite completa fallisce in quattro test preesistenti di `tests/test_selection.py`, rimasti all'aspettativa `person`/`car` dopo che il baseline ha abilitato anche `bicycle`, `motorcycle`, `bus` e `truck` in `classes.py`.
- Opzioni: 1) aggiornare separatamente i test e la documentazione della selezione  2) ripristinare la configurazione classi precedente.
- Consiglio: riallineare i test all'attuale configurazione intenzionale in un piano dedicato.
- Impatto: `unittest discover` non puo restituire 0; i test focalizzati gamma, transazione e CLI e gli snapshot reali restano verificati.

## [M0-T5] Gate di accuratezza esterno non disponibile
- Scelta: registrare `not verified — external verification: annotated dataset and evaluator are not part of the repository` e mantenere `ADAPTIVE_GAMMA_ENABLED = False`.
- Motivo: il repository non contiene il dataset annotato o l'evaluator necessari per confrontare mAP50 e recall.
- Alternative scartate: usare gli snapshot di conteggio o il controllo visivo come metriche di accuratezza, entrambi vietati dalla specifica.
- Reversibile: si, sostituendo lo stato con le metriche prodotte dalla medesima procedura esterna per baseline e variante.

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

## [M2-T3] Verifica Raspberry Pi esterna non disponibile
- Scelta: registrare `not verified — external verification: reference Raspberry Pi 3 is unavailable` e proseguire con i controlli locali e la documentazione.
- Motivo: l'ambiente corrente è macOS arm64, non la board Raspberry Pi 3 di riferimento richiesta dal criterio prestazionale.
- Alternative scartate: applicare il limite alla workstation o presentare timing locali come Raspberry Pi, entrambi vietati dalla specifica.
- Reversibile: sì, sostituendo lo stato con i sei risultati quando il blocco approvato verrà eseguito sulla board di riferimento.
