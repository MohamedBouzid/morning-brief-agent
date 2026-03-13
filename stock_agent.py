import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import ta  # technical analysis library
import sqlite3
from pathlib import Path

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import PromptTemplate

# Load configuration
with open('stock_config.json', 'r') as f:
    config = json.load(f)


class StockDataManager:
    """Manages stock data collection and storage."""

    def __init__(self, db_path: str = "data/stock_data.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for stock data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS stock_prices (
                    symbol TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (symbol, date)
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS stock_info (
                    symbol TEXT PRIMARY KEY,
                    company_name TEXT,
                    sector TEXT,
                    industry TEXT,
                    market_cap REAL,
                    pe_ratio REAL,
                    last_updated TEXT
                )
            ''')

    def fetch_stock_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Fetch stock data from yfinance."""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)

            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")

            # Store in database
            self._store_price_data(symbol, df)

            return df
        except Exception as e:
            raise Exception(f"Failed to fetch data for {symbol}: {str(e)}")

    def _store_price_data(self, symbol: str, df: pd.DataFrame):
        """Store price data in database."""
        with sqlite3.connect(self.db_path) as conn:
            df_reset = df.reset_index()
            df_reset['symbol'] = symbol
            df_reset['Date'] = df_reset['Date'].dt.strftime('%Y-%m-%d')

            df_reset.to_sql('stock_prices', conn, if_exists='append',
                          index=False, method='multi')

    def get_stock_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Retrieve stock data from database."""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT * FROM stock_prices
                WHERE symbol = ?
                AND date >= date('now', '-{} days')
                ORDER BY date
            '''.format(days)

            df = pd.read_sql_query(query, conn, params=[symbol])

            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.drop('symbol', axis=1, inplace=True)

            return df


class TechnicalAnalyzer:
    """Performs technical analysis on stock data."""

    def __init__(self):
        self.scaler = StandardScaler()

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe."""
        df = df.copy()

        # Basic indicators
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)

        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bollinger.bollinger_hband()
        df['BB_lower'] = bollinger.bollinger_lband()
        df['BB_middle'] = bollinger.bollinger_mavg()

        # Volume indicators
        df['Volume_SMA'] = ta.trend.sma_indicator(df['Volume'], window=20)

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()

        return df

    def generate_signals(self, df: pd.DataFrame) -> Dict[str, any]:
        """Generate trading signals based on technical analysis."""
        signals = {}

        # Trend signals
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Moving average crossover
        if latest['SMA_20'] > latest['SMA_50'] and prev['SMA_20'] <= prev['SMA_50']:
            signals['ma_crossover'] = 'bullish'
        elif latest['SMA_20'] < latest['SMA_50'] and prev['SMA_20'] >= prev['SMA_50']:
            signals['ma_crossover'] = 'bearish'

        # RSI signals
        if latest['RSI'] < 30:
            signals['rsi'] = 'oversold'
        elif latest['RSI'] > 70:
            signals['rsi'] = 'overbought'

        # MACD signals
        if latest['MACD'] > latest['MACD_signal'] and prev['MACD'] <= prev['MACD_signal']:
            signals['macd'] = 'bullish'
        elif latest['MACD'] < latest['MACD_signal'] and prev['MACD'] >= prev['MACD_signal']:
            signals['macd'] = 'bearish'

        # Bollinger Band signals
        if latest['Close'] < latest['BB_lower']:
            signals['bollinger'] = 'buy'
        elif latest['Close'] > latest['BB_upper']:
            signals['bollinger'] = 'sell'

        return signals

    def predict_price_movement(self, df: pd.DataFrame, days_ahead: int = 5) -> Dict[str, any]:
        """Predict future price movement using technical analysis."""
        # Simple momentum-based prediction
        recent_prices = df['Close'].tail(20)
        momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]

        # RSI trend
        rsi_trend = df['RSI'].tail(10).mean()

        # Volume trend
        volume_trend = df['Volume'].tail(10).pct_change().mean()

        prediction = {
            'momentum': momentum,
            'rsi_trend': rsi_trend,
            'volume_trend': volume_trend,
            'confidence': min(abs(momentum) * 100, 100)  # Simple confidence measure
        }

        # Direction prediction
        if momentum > 0.02 and rsi_trend < 60:
            prediction['direction'] = 'up'
            prediction['target_price'] = df['Close'].iloc[-1] * (1 + abs(momentum) * days_ahead / 20)
        elif momentum < -0.02 and rsi_trend > 40:
            prediction['direction'] = 'down'
            prediction['target_price'] = df['Close'].iloc[-1] * (1 - abs(momentum) * days_ahead / 20)
        else:
            prediction['direction'] = 'sideways'
            prediction['target_price'] = df['Close'].iloc[-1]

        prediction['timeframe_days'] = days_ahead

        return prediction


class StockAnalysisAgent:
    """Main stock analysis agent combining all components."""

    def __init__(self):
        self.data_manager = StockDataManager()
        self.technical_analyzer = TechnicalAnalyzer()
        self.llm = ChatOllama(
            base_url=config["model"]["base_url"],
            model=config["model"]["name"],
            temperature=config["model"]["temperature"]
        )

    async def analyze_stock(self, symbol: str) -> Dict[str, any]:
        """Complete stock analysis for a given symbol."""
        try:
            # Fetch data
            df = self.data_manager.fetch_stock_data(symbol, "2y")

            if df.empty:
                return {"error": f"No data available for {symbol}"}

            # Add technical indicators
            df_with_indicators = self.technical_analyzer.add_technical_indicators(df)

            # Generate signals
            signals = self.technical_analyzer.generate_signals(df_with_indicators)

            # Price predictions
            short_term_pred = self.technical_analyzer.predict_price_movement(df_with_indicators, 5)
            medium_term_pred = self.technical_analyzer.predict_price_movement(df_with_indicators, 20)
            long_term_pred = self.technical_analyzer.predict_price_movement(df_with_indicators, 60)

            # Get fundamental data
            fundamentals = await self.get_fundamentals(symbol)

            analysis = {
                "symbol": symbol,
                "current_price": df['Close'].iloc[-1],
                "technical_signals": signals,
                "predictions": {
                    "short_term": short_term_pred,
                    "medium_term": medium_term_pred,
                    "long_term": long_term_pred
                },
                "fundamentals": fundamentals,
                "recommendation": self.generate_recommendation(signals, short_term_pred, fundamentals),
                "risk_assessment": self.assess_risk(df_with_indicators),
                "last_updated": datetime.now().isoformat()
            }

            return analysis

        except Exception as e:
            return {"error": f"Analysis failed for {symbol}: {str(e)}"}

    async def get_fundamentals(self, symbol: str) -> Dict[str, any]:
        """Get fundamental data for a stock."""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info

            fundamentals = {
                "company_name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "pb_ratio": info.get("priceToBook", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "revenue_growth": info.get("revenueGrowth", 0),
                "earnings_growth": info.get("earningsGrowth", 0),
                "debt_to_equity": info.get("debtToEquity", 0)
            }

            return fundamentals

        except Exception as e:
            return {"error": f"Failed to get fundamentals: {str(e)}"}

    def generate_recommendation(self, signals: Dict, prediction: Dict, fundamentals: Dict) -> Dict[str, any]:
        """Generate buy/sell recommendation based on analysis."""
        score = 0

        # Technical signals scoring
        if signals.get('ma_crossover') == 'bullish':
            score += 2
        elif signals.get('ma_crossover') == 'bearish':
            score -= 2

        if signals.get('rsi') == 'oversold':
            score += 1
        elif signals.get('rsi') == 'overbought':
            score -= 1

        if signals.get('macd') == 'bullish':
            score += 1
        elif signals.get('macd') == 'bearish':
            score -= 1

        # Prediction scoring
        if prediction.get('direction') == 'up':
            score += 2
        elif prediction.get('direction') == 'down':
            score -= 2

        # Fundamental scoring
        pe_ratio = fundamentals.get('pe_ratio', 0)
        if 0 < pe_ratio < 25:  # Reasonable P/E
            score += 1
        elif pe_ratio > 40:  # Overvalued
            score -= 1

        debt_to_equity = fundamentals.get('debt_to_equity', 0)
        if debt_to_equity < 1:  # Low debt
            score += 1
        elif debt_to_equity > 2:  # High debt
            score -= 1

        # Generate recommendation
        if score >= 3:
            recommendation = "STRONG_BUY"
            confidence = min(score * 20, 100)
        elif score >= 1:
            recommendation = "BUY"
            confidence = min(score * 15, 80)
        elif score <= -3:
            recommendation = "STRONG_SELL"
            confidence = min(abs(score) * 20, 100)
        elif score <= -1:
            recommendation = "SELL"
            confidence = min(abs(score) * 15, 80)
        else:
            recommendation = "HOLD"
            confidence = 50

        return {
            "action": recommendation,
            "confidence": confidence,
            "score": score,
            "target_price": prediction.get('target_price', 0),
            "timeframe_days": prediction.get('timeframe_days', 5)
        }

    def assess_risk(self, df: pd.DataFrame) -> Dict[str, any]:
        """Assess risk metrics for the stock."""
        returns = df['Close'].pct_change().dropna()

        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)

        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_returns = returns - risk_free_rate/252
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5)

        return {
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "value_at_risk_95": var_95,
            "risk_level": "HIGH" if volatility > 0.4 else "MEDIUM" if volatility > 0.2 else "LOW"
        }


# Tool definitions for the agent
@tool
async def analyze_stock_tool(symbol: str) -> str:
    """Analyze a stock and provide comprehensive analysis with predictions."""
    agent = StockAnalysisAgent()
    analysis = await agent.analyze_stock(symbol.upper())

    if "error" in analysis:
        return f"Error analyzing {symbol}: {analysis['error']}"

    response = f"""
📊 **Stock Analysis: {analysis['symbol']}**

💰 **Current Price:** ${analysis['current_price']:.2f}

🎯 **Recommendation:** {analysis['recommendation']['action']}
📈 **Confidence:** {analysis['recommendation']['confidence']}%
🎯 **Target Price:** ${analysis['recommendation']['target_price']:.2f}
⏰ **Timeframe:** {analysis['recommendation']['timeframe_days']} days

📈 **Predictions:**
• Short-term (5 days): {analysis['predictions']['short_term']['direction'].upper()}
• Medium-term (20 days): {analysis['predictions']['medium_term']['direction'].upper()}
• Long-term (60 days): {analysis['predictions']['long_term']['direction'].upper()}

⚠️ **Risk Assessment:**
• Risk Level: {analysis['risk_assessment']['risk_level']}
• Volatility: {analysis['risk_assessment']['volatility']:.2%}
• Sharpe Ratio: {analysis['risk_assessment']['sharpe_ratio']:.2f}
• Max Drawdown: {analysis['risk_assessment']['max_drawdown']:.2%}

📊 **Technical Signals:**
{chr(10).join([f"• {k}: {v}" for k, v in analysis['technical_signals'].items()])}

🏢 **Fundamentals:**
• Company: {analysis['fundamentals'].get('company_name', 'N/A')}
• Sector: {analysis['fundamentals'].get('sector', 'N/A')}
• P/E Ratio: {analysis['fundamentals'].get('pe_ratio', 'N/A')}
• Market Cap: ${analysis['fundamentals'].get('market_cap', 0):,.0f}
"""

    return response

@tool
async def find_stocks_to_buy() -> str:
    """Find stocks with strong buy signals."""
    # Popular stocks to analyze
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]

    agent = StockAnalysisAgent()
    results = []

    for symbol in symbols:
        try:
            analysis = await agent.analyze_stock(symbol)
            if "error" not in analysis:
                rec = analysis['recommendation']
                if rec['action'] in ['BUY', 'STRONG_BUY']:
                    results.append({
                        'symbol': symbol,
                        'action': rec['action'],
                        'confidence': rec['confidence'],
                        'target_price': rec['target_price'],
                        'current_price': analysis['current_price'],
                        'upside': (rec['target_price'] - analysis['current_price']) / analysis['current_price'] * 100
                    })
        except:
            continue

    if not results:
        return "No strong buy opportunities found at this time."

    # Sort by upside potential
    results.sort(key=lambda x: x['upside'], reverse=True)

    response = "🚀 **Stocks to BUY - Top Opportunities**\n\n"
    for stock in results[:5]:  # Top 5
        response += f"""**{stock['symbol']}**
• Action: {stock['action']} ({stock['confidence']}% confidence)
• Current: ${stock['current_price']:.2f}
• Target: ${stock['target_price']:.2f}
• Upside: +{stock['upside']:.1f}%
• Risk: {agent.assess_risk(agent.data_manager.get_stock_data(stock['symbol']))['risk_level']}

"""

    return response

@tool
async def find_stocks_to_sell() -> str:
    """Find stocks with sell signals."""
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]

    agent = StockAnalysisAgent()
    results = []

    for symbol in symbols:
        try:
            analysis = await agent.analyze_stock(symbol)
            if "error" not in analysis:
                rec = analysis['recommendation']
                if rec['action'] in ['SELL', 'STRONG_SELL']:
                    results.append({
                        'symbol': symbol,
                        'action': rec['action'],
                        'confidence': rec['confidence'],
                        'target_price': rec['target_price'],
                        'current_price': analysis['current_price'],
                        'downside': (analysis['current_price'] - rec['target_price']) / analysis['current_price'] * 100
                    })
        except:
            continue

    if not results:
        return "No strong sell signals found at this time."

    # Sort by downside potential
    results.sort(key=lambda x: x['downside'], reverse=True)

    response = "⚠️ **Stocks to SELL - Potential Declines**\n\n"
    for stock in results[:5]:  # Top 5
        response += f"""**{stock['symbol']}**
• Action: {stock['action']} ({stock['confidence']}% confidence)
• Current: ${stock['current_price']:.2f}
• Target: ${stock['target_price']:.2f}
• Downside: -{stock['downside']:.1f}%

"""

    return response

@tool
async def get_market_overview() -> str:
    """Get overall market analysis and trends."""
    # Major indices
    indices = ["^GSPC", "^IXIC", "^DJI"]  # S&P 500, NASDAQ, Dow Jones

    agent = StockAnalysisAgent()
    market_data = {}

    for symbol in indices:
        try:
            df = agent.data_manager.fetch_stock_data(symbol, "1y")
            if not df.empty:
                current = df['Close'].iloc[-1]
                year_ago = df['Close'].iloc[0]
                change = (current - year_ago) / year_ago * 100

                market_data[symbol] = {
                    'current': current,
                    'change_ytd': change,
                    'volatility': df['Close'].pct_change().std() * np.sqrt(252)
                }
        except:
            continue

    response = "🌍 **Market Overview**\n\n"

    for symbol, data in market_data.items():
        name = "S&P 500" if symbol == "^GSPC" else "NASDAQ" if symbol == "^IXIC" else "Dow Jones"
        response += f"""**{name} ({symbol})**
• Current: {data['current']:.0f}
• YTD Change: {data['change_ytd']:+.1f}%
• Volatility: {data['volatility']:.1%}

"""

    # Market sentiment based on technical analysis
    spx_df = agent.data_manager.get_stock_data("^GSPC", 100)
    if not spx_df.empty:
        spx_technical = agent.technical_analyzer.add_technical_indicators(spx_df)
        signals = agent.technical_analyzer.generate_signals(spx_technical)

        response += "\n📊 **Market Sentiment:**\n"
        if signals.get('ma_crossover') == 'bullish':
            response += "• Trend: BULLISH (MA crossover)\n"
        elif signals.get('ma_crossover') == 'bearish':
            response += "• Trend: BEARISH (MA crossover)\n"

        rsi = spx_technical['RSI'].iloc[-1]
        if rsi < 30:
            response += "• RSI: OVERSOLD (potential bounce)\n"
        elif rsi > 70:
            response += "• RSI: OVERBOUGHT (potential pullback)\n"
        else:
            response += f"• RSI: {rsi:.1f} (neutral)\n"

    return response


# Agent setup
TOOLS = [analyze_stock_tool, find_stocks_to_buy, find_stocks_to_sell, get_market_overview]

template = """You are an expert stock market analysis agent. You can:

1. Analyze individual stocks with technical and fundamental analysis
2. Predict price movements with timeframes and target prices
3. Identify stocks to buy or sell based on comprehensive analysis
4. Provide market overviews and sentiment analysis

Available Tools:
{tools}

Current Query: {input}

Use the tools to provide detailed, actionable stock market insights. Always include specific timeframes and target prices in your recommendations.

Begin!

{agent_scratchpad}"""

prompt = PromptTemplate(
    template=template,
    input_variables=["tools", "input", "agent_scratchpad"],
)

agent = create_tool_calling_agent(
    llm=ChatOllama(
        base_url=config["model"]["base_url"],
        model=config["model"]["name"],
        temperature=config["model"]["temperature"]
    ),
    tools=TOOLS,
    prompt=prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=TOOLS,
    verbose=True,
    max_concurrency=5,
)


if __name__ == "__main__":
    async def main():
        query = config["query"]
        result = await agent_executor.ainvoke({"input": query})
        print("Analysis Complete!")
        print(result.get("output", ""))

    asyncio.run(main())