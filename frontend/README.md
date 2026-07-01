# RetinaAI Frontend

Premium Next.js interface for the RetinaAI screening prototype.

## Stack

- Next.js 16 App Router
- React 19
- TypeScript
- Tailwind CSS 4
- shadcn-style local primitives
- Framer Motion
- Lucide and Tabler icons
- Recharts
- Three.js
- Geist font

## Commands

```bash
npm install
npm run lint
npm run build
npm run dev -- --hostname 127.0.0.1 --port 3000
```

## Connected Screening API

`POST /api/screen` accepts multipart form data:

- `image`: retinal image file, or
- `demo=true`: use `tests/_self_check/synthetic_retina.png`

The route shells out to:

```bash
python -m src.inference --image <path> --model models/baseline_sklearn.pkl
```

It returns quality, prediction, uncertainty routing, and artifact URLs for the generated report and explanation image.

Set `RETINAAI_PROJECT_ROOT` if the frontend is run outside the default `frontend/` subdirectory.
