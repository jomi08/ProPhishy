ProPhishy frontend (Next.js)

This is a minimal Next.js scaffold placed in the `frontend` folder. It exposes a small API route at `/api/hello` and a basic homepage.

Quick start (Windows, cmd.exe):

1. Open a terminal and change to the frontend folder:

```cmd
cd c:\Users\jomim\OneDrive\Desktop\Jofolds\prophishy\frontend
```

2. Install dependencies:

```cmd
npm install
```

3. Start the dev server:

```cmd
npm run dev
```

4. Open http://localhost:3000 in your browser.

Next steps:
- Wire the frontend to your Python backend (e.g., expose an HTTP endpoint `/predict` that accepts email text and returns predictions).
- Add authentication if you plan to access Gmail from the browser.
- Replace the placeholder UI with real pages and components.
