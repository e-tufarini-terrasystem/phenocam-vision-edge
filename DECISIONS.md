## [M2-T2] Quiet third-party boundary
- Scelta: silenziare il logger Ultralytics sotto `ERROR` e gli avvisi nativi emessi soltanto durante l'import, ripristinando sempre `stderr`.
- Motivo: lo smoke test locale produceva output di terze parti nonostante `verbose=False`, violando il contratto di successo silenzioso e dei diagnostici esatti.
- Alternative scartate: configurare `task=detect`, vietato dalla specifica; modificare i pacchetti installati, fuori ambito; accettare stream non vuoti, contrario ai criteri di accettazione.
- Reversibile: sì, rimuovendo la gestione dell'import e il livello del logger in `inference.py`.
