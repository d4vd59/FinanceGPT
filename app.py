from flask import Flask, render_template, jsonify, request
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import time
import random
import requests
import json
import nltk

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

app = Flask(__name__)

class MarketDataCollector:
    def __init__(self):
        self.top_stocks = [
            'MSFT', 'NVDA', 'AAPL', 'AMZN', 'GOOGL', 'META', 'AVGO', 'TCEHY', 'TSLA', 'TSM',
            'UNH', 'JPM', 'LLY', 'ORCL', 'ASML', 'BABA', 'NFLX', 'MA', 'XOM', 'PLTR',
            'COST', 'KO', 'PFE', 'PG', 'JNJ', 'HD', 'CVX', 'WFC', 'SAP', 'ABBV'
        ]
        
        self.top_cryptos = [
            'BTC-USD', 'ETH-USD', 'USDT-USD', 'XRP-USD', 'BNB-USD', 'SOL-USD', 'ADA-USD', 
            'DOGE-USD', 'DOT-USD', 'MATIC-USD', 'LTC-USD', 'SHIB-USD', 'WLD-USD', 
            'HNT-USD', 'AVAX-USD', 'LINK-USD'
        ]
        
        self.cache_duration = 900
        self.stocks_cache = None
        self.crypto_cache = None
        self.stocks_cache_time = None
        self.crypto_cache_time = None
        
    def is_cache_valid(self, cache_time):
        if cache_time is None:
            return False
        return (datetime.now() - cache_time).total_seconds() < self.cache_duration
    
    def get_cached_data(self, asset_type):
        if asset_type == 'stocks':
            if self.is_cache_valid(self.stocks_cache_time) and self.stocks_cache:
                print(f"✓ Using cached stocks data (age: {(datetime.now() - self.stocks_cache_time).total_seconds():.0f}s)")
                return self.stocks_cache
        elif asset_type == 'crypto':
            if self.is_cache_valid(self.crypto_cache_time) and self.crypto_cache:
                print(f"✓ Using cached crypto data (age: {(datetime.now() - self.crypto_cache_time).total_seconds():.0f}s)")
                return self.crypto_cache
        return None
    
    def cache_data(self, asset_type, data):
        current_time = datetime.now()
        if asset_type == 'stocks':
            self.stocks_cache = data
            self.stocks_cache_time = current_time
        elif asset_type == 'crypto':
            self.crypto_cache = data
            self.crypto_cache_time = current_time
        
        print(f"✓ Cached {asset_type} data at {current_time.strftime('%H:%M:%S')} ({len(data)} items)")
    
    def get_market_data_with_cache(self, asset_type):
        cached_data = self.get_cached_data(asset_type)
        if cached_data:
            return cached_data
        
        print(f"🔄 Fetching fresh {asset_type} data from API...")
        if asset_type == 'stocks':
            symbols = self.top_stocks
        elif asset_type == 'crypto':
            symbols = self.top_cryptos
        else:
            symbols = self.top_stocks
            
        data = self.get_market_data(symbols)
        self.cache_data(asset_type, data)
        return data

    def get_market_data(self, symbols):
        data = {}
        successful_count = 0
        failed_symbols = []
        
        print(f"📊 Fetching data for {len(symbols)} symbols...")
        
        for i, symbol in enumerate(symbols):
            try:
                if i > 0:
                    delay = random.uniform(0.5, 1.5)
                    time.sleep(delay)
                
                print(f"  [{i+1}/{len(symbols)}] Fetching {symbol}...", end=' ')
                
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                if len(hist) == 0:
                    print("❌ No history")
                    failed_symbols.append(symbol)
                    continue
                
                current_price = hist['Close'].iloc[-1]
                
                if len(hist) >= 2:
                    prev_price = hist['Close'].iloc[-2]
                    change_pct = ((current_price - prev_price) / prev_price) * 100
                else:
                    change_pct = 0
                
                try:
                    info = ticker.info
                    if info and isinstance(info, dict):
                        name = info.get('longName') or info.get('shortName') or symbol
                        market_cap = info.get('marketCap', 0)
                    else:
                        name = symbol
                        market_cap = 0
                except:
                    name = symbol
                    market_cap = 0
                
                logo_url = self.get_company_logo(symbol)
                
                data[symbol] = {
                    'price': float(current_price),
                    'change_pct': float(change_pct),
                    'market_cap': int(market_cap) if market_cap else 0,
                    'name': str(name),
                    'logo_url': logo_url
                }
                successful_count += 1
                print(f"✓ ${current_price:.2f}")
                
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}...")
                failed_symbols.append(symbol)
                
                if "429" in str(e) or "Too Many Requests" in str(e):
                    print("⏳ Rate limited - waiting 5 seconds...")
                    time.sleep(5)
        
        print(f"📈 Successfully loaded {successful_count}/{len(symbols)} symbols")
        if failed_symbols:
            print(f"❌ Failed: {', '.join(failed_symbols[:5])}{'...' if len(failed_symbols) > 5 else ''}")
        
        return data
    
    def get_company_logo(self, symbol):
        if symbol.endswith('-USD'):
            crypto_mapping = {
                'BTC-USD': 'https://assets.coingecko.com/coins/images/1/large/bitcoin.png',
                'ETH-USD': 'https://assets.coingecko.com/coins/images/279/large/ethereum.png',
                'USDT-USD': 'https://assets.coingecko.com/coins/images/325/large/Tether.png',
                'XRP-USD': 'https://assets.coingecko.com/coins/images/44/large/xrp-symbol-white-128.png',
                'BNB-USD': 'https://assets.coingecko.com/coins/images/825/large/bnb-icon2_2x.png',
                'SOL-USD': 'https://assets.coingecko.com/coins/images/4128/large/solana.png',
                'ADA-USD': 'https://assets.coingecko.com/coins/images/975/large/cardano.png',
                'DOGE-USD': 'https://assets.coingecko.com/coins/images/5/large/dogecoin.png',
                'DOT-USD': 'https://assets.coingecko.com/coins/images/12171/large/polkadot.png',
                'MATIC-USD': 'https://assets.coingecko.com/coins/images/4713/large/matic-token-icon.png',
                'LTC-USD': 'https://assets.coingecko.com/coins/images/2/large/litecoin.png',
                'SHIB-USD': 'https://assets.coingecko.com/coins/images/11939/large/shiba.png',
                'WLD-USD': 'https://assets.coingecko.com/coins/images/31069/large/worldcoin.jpeg',
                'HNT-USD': 'https://assets.coingecko.com/coins/images/4284/large/Helium_HNT.png',
                'AVAX-USD': 'https://assets.coingecko.com/coins/images/12559/large/Avalanche_Circle_RedWhite_Trans.png',
                'LINK-USD': 'https://assets.coingecko.com/coins/images/877/large/chainlink-new-logo.png'
            }
            return crypto_mapping.get(symbol, None)
        else:
            domain = self.get_company_domain(symbol)
            if domain:
                return f"https://logo.clearbit.com/{domain}"
        return None
    
    def get_company_domain(self, symbol):
        domain_mapping = {
            'MSFT': 'microsoft.com', 'NVDA': 'nvidia.com', 'AAPL': 'apple.com', 
            'AMZN': 'amazon.com', 'GOOGL': 'google.com', 'META': 'meta.com', 
            'AVGO': 'broadcom.com', 'TCEHY': 'tencent.com', 'TSLA': 'tesla.com', 
            'TSM': 'tsmc.com', 'UNH': 'unitedhealthgroup.com', 'JPM': 'jpmorganchase.com', 
            'LLY': 'lilly.com', 'ORCL': 'oracle.com', 'ASML': 'asml.com', 
            'BABA': 'alibaba.com', 'NFLX': 'netflix.com', 'MA': 'mastercard.com', 
            'XOM': 'exxonmobil.com', 'PLTR': 'palantir.com', 'COST': 'costco.com',
            'KO': 'coca-cola.com', 'PFE': 'pfizer.com', 'PG': 'pg.com', 
            'JNJ': 'jnj.com', 'HD': 'homedepot.com', 'CVX': 'chevron.com',
            'WFC': 'wellsfargo.com', 'SAP': 'sap.com', 'ABBV': 'abbvie.com'
        }
        return domain_mapping.get(symbol, None)

class GroqStockPredictor:
    def __init__(self):
        self.api_key = "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"
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.3-70b-versatile"
        
        self.prediction_cache = {}
        self.cache_duration = 3600
    
    def is_cache_valid(self, cache_time):
        if cache_time is None:
            return False
        return (datetime.now() - cache_time).total_seconds() < self.cache_duration
    
    def get_ai_prediction(self, symbol):
        try:
            cache_key = f"groq_{symbol}"
            if cache_key in self.prediction_cache:
                cached_pred, cache_time = self.prediction_cache[cache_key]
                if self.is_cache_valid(cache_time):
                    print(f"✅ Using cached Groq prediction for {symbol}")
                    return cached_pred
    
            context = self.get_stock_context(symbol)
            if not context:
                return None
            
            trend_direction = "bullish" if context['change_7d'] > 2 else "bearish" if context['change_7d'] < -2 else "neutral"
            volatility = "high" if abs(context['change_7d']) > 5 else "low" if abs(context['change_7d']) < 1 else "medium"
            
            prompt = f"""You are a professional stock analyst. Analyze {symbol} based on the following data:

Current Price: ${context['current_price']}
7-day change: {context['change_7d']}%
30-day change: {context['change_30d']}%
Recent trend: {trend_direction}
Volatility: {volatility}

Provide a realistic analysis. Consider:
- If 7-day change is very negative (< -5%), lean toward bearish/sell
- If 7-day change is very positive (> 5%), consider if it's overextended
- If change is moderate, provide balanced view
- Price targets should reflect realistic expectations, not always growth

Respond with ONLY this JSON format (replace values with your analysis):
{{
    "price_target_7d": [calculate realistic 7-day target],
    "price_target_30d": [calculate realistic 30-day target],
    "confidence_7d": [1-100 based on data quality],
    "confidence_30d": [1-100 based on uncertainty],
    "direction": "[bullish/bearish/neutral based on analysis]",
    "risk_level": "[low/medium/high based on volatility]",
    "recommendation": "[buy/sell/hold based on analysis]",
    "key_factors": ["factor1", "factor2", "factor3"],
    "reasoning": "Your analytical reasoning here"
}}"""
        
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an expert financial analyst. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 500
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"🤖 Asking Groq AI about {symbol} (attempt {attempt + 1}/{max_retries})...")
                    
                    timeout = 15 + (attempt * 10)
                    response = requests.post(self.api_url, json=data, headers=headers, timeout=timeout)
                    
                    if response.status_code == 200:
                        response_json = response.json()
                        
                        if "choices" in response_json and len(response_json["choices"]) > 0:
                            ai_response = response_json["choices"][0]["message"]["content"]
                            
                            try:
                                prediction = json.loads(ai_response)
                                
                                prediction.update({
                                    'symbol': symbol,
                                    'current_price': context['current_price'],
                                    'timestamp': datetime.now().isoformat(),
                                    'ai_model': self.model,
                                    'powered_by': 'Groq AI'
                                })
                                
                                self.prediction_cache[cache_key] = (prediction, datetime.now())
                                
                                print(f"✅ Groq AI prediction for {symbol}: {prediction['direction']}")
                                return prediction
                                
                            except json.JSONDecodeError as e:
                                print(f"❌ JSON parsing error: {e}")
                                if attempt == max_retries - 1:
                                    return None
                                continue
                        else:
                            print("❌ No valid response from Groq")
                            if attempt == max_retries - 1:
                                return None
                            continue
                            
                    elif response.status_code == 429:
                        print(f"⏳ Rate limited - waiting {5 * (attempt + 1)} seconds...")
                        time.sleep(5 * (attempt + 1))
                        continue
                        
                    else:
                        print(f"❌ Groq API Error: {response.status_code}")
                        if attempt == max_retries - 1:
                            return None
                        continue
                        
                except requests.exceptions.Timeout:
                    print(f"⏰ Timeout on attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"⏳ Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print("❌ All retry attempts failed due to timeout")
                        return None
                        
                except requests.exceptions.ConnectionError:
                    print(f"🌐 Connection error on attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        return None
                    
            return None
            
        except Exception as e:
            print(f"❌ Groq prediction error: {e}")
            return None