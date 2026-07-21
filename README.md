# Resume ATS Tailor (FastAPI)

This project now uses **Python FastAPI** for the backend while serving the existing HTML frontend from the same app.

## Setup

1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows PowerShell
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Configure environment variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key_here
PORT=3000
NODE_ENV=development
```

4. Run the app

```bash
python main.py
```

Open: `http://localhost:3000`

## API Endpoints

- `POST /api/tailor-resume`
- `POST /api/generate-cv`
- `GET /api/test`
- `GET /api/health`

## Notes

- `index.html` is served from `/`
- If a `public/` directory exists, it is served at `/public`
