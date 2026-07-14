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
