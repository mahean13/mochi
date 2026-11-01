# Mochi

A browser game for a preference study of machine creatures. Players choose between two creatures under one question, and their choices breed new generations and train a Bradley-Terry preference model.

## Play locally

Requires Node.js 18 or later. No JavaScript dependencies are required.

```sh
npm run build
python3 -m http.server 8000 --directory web
```

Open http://localhost:8000.

## Deploy to Vercel

Import this GitHub repository into Vercel. The included `vercel.json` builds the game with `npm run build` and serves `web/`. No environment variables are required for the preview.

**Preview only:** ethics approval is pending. `STUDY_OPEN` remains false, and preview picks are excluded from the study. On Vercel and other static hosts, picks are saved in the visitor's browser only. The shared database integration uses the Claude artifact host API and is not available on Vercel; a separate backend must be configured and tested before cross-device study data collection.

## Tests and analysis

```sh
npm test
python3 -m pip install -r analysis/requirements.txt
cd analysis
python3 -m pytest tests
python3 -m mochi_analysis <export-folder> --out results --figures figures
```

The researcher view exports picks, creatures and state. `docs/uml/` contains diagram sources; `docs/figures/` contains rendered figures. Run `node scripts/benchmark.mjs` from the repository root for the engine benchmark.
