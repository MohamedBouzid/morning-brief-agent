import asyncio
import json
import re
import httpx

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import PromptTemplate

with open('config.json', 'r') as f:
    config = json.load(f)

@tool
async def weather_tool(lat: str, long: str) -> str:
    """Get weather conditions today."""
    print("Fetching weather...")
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={long}&daily=temperature_2m_max,temperature_2m_min"
        )
        data = resp.json()
        min_t = data["daily"]["temperature_2m_min"][0]
        max_t = data["daily"]["temperature_2m_max"][0]
        return f"Weather in {lat},{long}: {min_t}–{max_t}°C"

@tool
async def news_tool() -> str:
    """Get latest news."""
    print("Fetching news...")
    async with httpx.AsyncClient() as client:
        r = await client.get("https://www.bbc.com/news", timeout=10)
        html = r.text

    title_match = re.search(r'<h[23][^>]*>(.*?)</h[23]>', html)
    title = re.sub(r'<[^>]+>', '', title_match.group(1)) if title_match else "No headline"

    para_match = re.search(r'<p[^>]*>([^<]{50,200})</p>', html)
    para = re.sub(r'<[^>]+>', '', para_match.group(1)) if para_match else ""
    result = f"{title}. {para[:150]}..."
    #return {"news": result}    
    return f"Latest news: {result}"

@tool
async def get_location_tool() -> str:
    """Get the current location based on IP address."""
    async with httpx.AsyncClient() as client:
        r = await client.get("http://ip-api.com/json/", timeout=10)
        loc_data = r.json()
        LAT, LON = loc_data['lat'], loc_data['lon']
        location_str = f"Location: {loc_data.get('city', 'Unknown')}, {loc_data.get('country', 'Unknown')} (lat: {LAT}, lon: {LON})"
        return location_str

@tool
async def save_to_file(data: str, filename: str) -> str:
    """Save data to a file."""
    try:
        def write_file():
            with open(filename, 'w') as f:
                f.write(data)
        await asyncio.to_thread(write_file)
        return f"Data saved to {filename}"
    except Exception as e:
        return f"Error saving to file: {e}"

TOOLS=[weather_tool, news_tool, get_location_tool, save_to_file]

llm = ChatOllama(
    base_url=config["model"]["base_url"],
    model=config["model"]["name"],
    temperature=config["model"]["temperature"]
)

# The agent needs a template that includes `{agent_scratchpad}`
template = """You are an intelligent agent that can call tools.

Available Tools:
{{tools}}  # literal text

User: {input}

Begin!

{agent_scratchpad}"""


prompt = PromptTemplate(
    template=template,
    input_variables=["tools", "input", "agent_scratchpad"],
)

agent = create_tool_calling_agent(
    llm=llm,
    tools=TOOLS,
    prompt=prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=TOOLS,
    verbose=True,
    max_concurrency=10,
)

if __name__ == "__main__":
    async def main():
        query = config["query"]
        await agent_executor.ainvoke({"input": query})
    asyncio.run(main())
