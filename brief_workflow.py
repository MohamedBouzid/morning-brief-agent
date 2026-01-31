import asyncio, datetime, os, httpx
from dotenv import load_dotenv
import openai
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import re

load_dotenv()
os.makedirs("reports", exist_ok=True)

clientOpenAi = openai.AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

class State(TypedDict):
    weather: str
    news: str
    brief: str

LAT, LON = 48.8566, 2.3522

async def weather_node(state: State) -> State:
    print(">>> weather_node")
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        f"&timezone=auto"
    )
    async with httpx.AsyncClient() as client:
        r = await client.get(url, timeout=10)
        data = r.json()

    tmin = int(data["daily"]["temperature_2m_min"][0])
    tmax = int(data["daily"]["temperature_2m_max"][0])
    rain = float(data["daily"]["precipitation_sum"][0])
    result = f"{tmin}–{tmax}°C, {rain:.1f}mm rain"
    return {"weather": result}

async def news_node(state: State) -> State:
    print(">>> news_node")
    async with httpx.AsyncClient() as client:
        r = await client.get("https://www.bbc.com/news", timeout=10)
        html = r.text

    title_match = re.search(r'<h[23][^>]*>(.*?)</h[23]>', html)
    title = re.sub(r'<[^>]+>', '', title_match.group(1)) if title_match else "No headline"

    para_match = re.search(r'<p[^>]*>([^<]{50,200})</p>', html)
    para = re.sub(r'<[^>]+>', '', para_match.group(1)) if para_match else ""
    result = f"{title}. {para[:150]}..."
    return {"news": result}

async def reporter_node(state: State) -> State:
    print(">>> reporter_node")

    weather = state["weather"]
    news = state["news"]

    prompt = f"Morning brief:\nWeather: {weather}\nNews: {news}"
    resp = await clientOpenAi.chat.completions.create(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120
    )
    brief = resp.choices[0].message.content
    date = datetime.date.today().isoformat()

    with open(f"reports/{date}.md", "w") as f:
        f.write(brief)

    print("brief written to file")
    return {"brief": brief}

if __name__ == "__main__":
    workflow = StateGraph(State)
    workflow.add_node("weather", weather_node)
    workflow.add_node("news", news_node)
    workflow.add_node("reporter", reporter_node)

    workflow.add_edge(START, "weather")
    workflow.add_edge(START, "news")
    workflow.add_edge("weather", "reporter")
    workflow.add_edge("news", "reporter")
    workflow.add_edge("reporter", END)

    app = workflow.compile()
    async def main():
        print("--- starting workflow ---")
        result = await app.ainvoke({})
        print("--- Workflow execution finished ---")
        print("Report: " + result["brief"])

    asyncio.run(main())