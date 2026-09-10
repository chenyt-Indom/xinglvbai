"""配置常量：API密钥和URL，从环境变量读取"""
import os
from dotenv import load_dotenv

# 加载 .env 文件（本地开发用）
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
# 每日DeepSeek费用上限（元），防止无人使用时被刷接口导致费用失控；用尽后当天拒绝AI调用
DEEPSEEK_DAILY_BUDGET = float(os.getenv("DEEPSEEK_DAILY_BUDGET", "3"))
AMAP_KEY = os.getenv("AMAP_API_KEY", "")
AMAP_POI_URL = "https://restapi.amap.com/v3/place/text"
AMAP_WEATHER_URL = "https://restapi.amap.com/v3/weather/weatherInfo"
AMAP_GEO_URL = "https://restapi.amap.com/v3/geocode/geo"
AMAP_REGEO_URL = "https://restapi.amap.com/v3/geocode/regeo"
IMG_BASE = "https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image"
VARIFLIGHT_KEY = os.getenv("VARIFLIGHT_API_KEY", "")
VARIFLIGHT_URL = "https://ai.variflight.com/api/v1/mcp/data"