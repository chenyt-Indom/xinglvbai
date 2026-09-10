"""中国天气智能体服务：集成wttr.in实时观测 + 高德空气质量 + DeepSeek天气对话"""
import httpx
from config import AMAP_KEY, AMAP_WEATHER_URL, DEEPSEEK_KEY, DEEPSEEK_URL

CITY_MAP = {
    "北京": "Beijing", "上海": "Shanghai", "广州": "Guangzhou", "深圳": "Shenzhen",
    "杭州": "Hangzhou", "成都": "Chengdu", "重庆": "Chongqing", "南京": "Nanjing",
    "武汉": "Wuhan", "西安": "Xian", "苏州": "Suzhou", "长沙": "Changsha",
    "天津": "Tianjin", "厦门": "Xiamen", "青岛": "Qingdao", "大连": "Dalian",
    "昆明": "Kunming", "三亚": "Sanya", "桂林": "Guilin", "丽江": "Lijiang",
    "哈尔滨": "Harbin", "拉萨": "Lhasa", "乌鲁木齐": "Urumqi", "贵阳": "Guiyang",
}


async def get_observation(city: str) -> dict:
    """中国天气智能体-实况观测：综合wttr.in实时数据 + 高德天气"""
    en = CITY_MAP.get(city, city)
    result = {"success": True, "city": city, "source": "中国天气智能体"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            wttr = (await client.get(f"https://wttr.in/{en}?format=j1", follow_redirects=True)).json()
            c = wttr.get("current_condition", [{}])[0]
            result["observation"] = {
                "temp": f"{c.get('temp_C', '')}°C", "feels_like": f"{c.get('FeelsLikeC', '')}°C",
                "weather": (c.get("weatherDesc", [{}])[0].get("value", "") if c.get("weatherDesc") else ""),
                "humidity": f"{c.get('humidity', '')}%", "wind_dir": c.get("winddir16Point", ""),
                "wind_speed": f"{c.get('windspeedKmph', '')}km/h", "pressure": f"{c.get('pressure', '')}hPa",
                "visibility": f"{c.get('visibility', '')}km", "uv": c.get("uvIndex", ""),
                "cloud_cover": f"{c.get('cloudcover', '')}%",
            }
            wd = wttr.get("weather", [])
            if wd:
                t = wd[0]
                result["forecast_summary"] = {
                    "date": t.get("date", ""), "max_temp": f"{t.get('maxtempC', '')}°C",
                    "min_temp": f"{t.get('mintempC', '')}°C",
                    "sunrise": (t.get("astronomy", [{}])[0].get("sunrise", "") if t.get("astronomy") else ""),
                    "sunset": (t.get("astronomy", [{}])[0].get("sunset", "") if t.get("astronomy") else ""),
                }
    except Exception:
        result["observation"] = {"error": "wttr.in不可用"}
    return result


async def get_air_quality(city: str) -> dict:
    """中国天气智能体-空气质量：通过高德天气API获取"""
    result = {"success": False, "city": city, "source": "中国天气智能体"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(AMAP_WEATHER_URL, params={"key": AMAP_KEY, "city": city, "extensions": "all"})
            data = resp.json()
            if data.get("status") == "1":
                for f in data.get("forecasts", []):
                    for c in f.get("casts", []):
                        aqi = c.get("aqi", "")
                        if aqi:
                            aqi_val = int(aqi) if aqi.isdigit() else 0
                            levels = [(50, "优", "空气质量令人满意", "#52C41A"), (100, "良", "空气质量可接受", "#FAAD14"),
                                      (150, "轻度污染", "敏感人群症状有轻度加剧", "#FF7A45"),
                                      (200, "中度污染", "进一步加剧敏感人群症状", "#FF4D4F"),
                                      (300, "重度污染", "心脏病和肺病患者症状显著加剧", "#9900FF")]
                            level, desc, color = "严重污染", "健康人群有明显强烈症状", "#7D0000"
                            for limit, lv, ds, cl in levels:
                                if aqi_val <= limit: level, desc, color = lv, ds, cl; break
                            return {"success": True, "city": city, "source": "中国天气智能体",
                                    "aqi": aqi_val, "level": level, "description": desc, "color": color,
                                    "pm25": c.get("pm25", ""), "pm10": c.get("pm10", ""), "date": c.get("date", "")}
    except Exception:
        pass
    return result


async def get_weather_chat(city: str, query: str = "") -> dict:
    """中国天气智能体-AI天气对话：使用DeepSeek生成智能天气建议"""
    result = {"success": True, "city": city, "source": "中国天气智能体"}
    en = CITY_MAP.get(city, city)
    ctx = ""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            wttr = (await client.get(f"https://wttr.in/{en}?format=j1", follow_redirects=True)).json()
            c = wttr.get("current_condition", [{}])[0]
            ctx = f"当前{city}天气：{c.get('weatherDesc', [{}])[0].get('value', '')}，气温{c.get('temp_C', '')}°C，体感{c.get('FeelsLikeC', '')}°C，湿度{c.get('humidity', '')}%，风速{c.get('windspeedKmph', '')}km/h。"
            if wttr.get("weather"):
                t = wttr["weather"][0]
                ctx += f"今日最高{t.get('maxtempC', '')}°C，最低{t.get('mintempC', '')}°C。"
    except Exception:
        pass
    prompt = f"""你是中国天气智能体助手。当前天气：{ctx}
用户：{query or f'请根据{city}天气给出行建议'}
请提供穿衣、出行、健康方面建议，语言亲切自然，200字以内。"""
    try:
        # 每日费用保护：与行程生成共用同一预算，防止被刷
        from deepseek_service import check_deepseek_budget, record_deepseek_usage
        budget_err = check_deepseek_budget()
        if budget_err:
            result["reply"] = budget_err
            return result
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(DEEPSEEK_URL, json={
                "model": "deepseek-chat",
                "messages": [{"role": "system", "content": "你是中国天气智能体，提供专业天气分析和生活建议。"},
                             {"role": "user", "content": prompt}],
                "max_tokens": 500, "temperature": 0.7,
            }, headers={"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"})
            data = resp.json()
            result["reply"] = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage", {})
            record_deepseek_usage(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))
    except Exception:
        result["reply"] = f"当前{city}天气数据获取中，请稍后再试。出行请注意查看实时天气，合理安排行程。"
    result["weather_context"] = ctx
    return result