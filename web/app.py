"""RSL Baseball V3 — FastAPI."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import date

from web.predict_v3 import predict_games_v3, predict_consensus
from data.mlb_api import get_schedule

app = FastAPI(title="RSL Baseball V3")

templates_dir = os.path.join(os.path.dirname(__file__), "templates")
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates = Jinja2Templates(directory=templates_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/v3/predictions/{game_date}")
async def api_v3_predictions(game_date: str, force: int = 0, kst: int = 1):
    """V3 투수 매치업 F5 예측 (Free). kst=1이면 KST 날짜 기준."""
    try:
        preds = predict_games_v3(game_date, force=bool(force), kst=bool(kst))
        return JSONResponse(content={"predictions": preds, "date": game_date})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/v3/consensus/{game_date}")
async def api_v3_consensus(game_date: str, force: int = 0, kst: int = 1):
    """V3 Pro: V2+V3 컨센서스 엄선 픽 (Premium). kst=1이면 KST 날짜 기준."""
    try:
        result = predict_consensus(game_date, force=bool(force), kst=bool(kst))
        return JSONResponse(content={"date": game_date, **result})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/schedule/{game_date}")
async def api_schedule(game_date: str):
    try:
        games = get_schedule(game_date)
        return JSONResponse(content=games)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
