"""飞常准(Variflight) REST API 服务：航班查询+火车票查询+中转方案"""
import httpx
import asyncio
from config import VARIFLIGHT_KEY, VARIFLIGHT_URL

# MCP工具名 → REST端点名映射（飞常准API的MCP工具名与REST端点名不同，需映射）
MCP_TO_REST_ENDPOINT = {
    "searchFlightsByDepArr": "flights",
    "searchFlightsByNumber": "flight",
    "getFlightTransferInfo": "transfer",
    "flightHappinessIndex": "happiness",
    "getRealtimeLocationByAnum": "realtimeLocation",
    "getFutureWeatherByAirport": "futureAirportWeather",
    "searchFlightItineraries": "searchFlightItineraries",
    "getFlightPriceByCities": "getFlightPriceByCities",
    "trainStanTicket": "trainStanTicket",
}


def _fmt_duration(raw: str) -> str:
    """将分钟数或秒数转换为 'XhYmin' 格式"""
    if not raw:
        return ""
    try:
        minutes = int(raw)
        if minutes > 1000:  # 秒数
            minutes = minutes // 60
        hours = minutes // 60
        mins = minutes % 60
        if hours > 0 and mins > 0:
            return f"{hours}h{mins}min"
        elif hours > 0:
            return f"{hours}h"
        else:
            return f"{mins}min"
    except (ValueError, TypeError):
        return str(raw)


async def _call_variflight(endpoint: str, params: dict) -> dict:
    """调用飞常准API通用方法，使用REST格式
    参数必须嵌套在params键下
    包含重试机制：HTTP 403/429/502/503及连接错误时自动重试最多3次"""
    if not VARIFLIGHT_KEY:
        return {"success": False, "error": "VARIFLIGHT_API_KEY未配置", "data": []}
    rest_endpoint = MCP_TO_REST_ENDPOINT.get(endpoint, endpoint)
    max_retries = 3
    last_error = ""
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                common_headers = {
                    "X-VARIFLIGHT-KEY": VARIFLIGHT_KEY,
                    "Content-Type": "application/json",
                }
                resp = await client.post(
                    VARIFLIGHT_URL,
                    headers=common_headers,
                    json={"endpoint": rest_endpoint, "params": params},
                )
                if resp.status_code == 401:
                    return {"success": False, "error": "飞常准API Key无效", "data": []}
                # 403 多为余额不足/无数据权限，重试无意义且会把整个请求拖到几分钟，直接返回
                if resp.status_code == 403:
                    return {"success": False, "error": "飞常准API余额不足或无数据权限", "data": []}
                # 可重试的错误码（限流/网关抖动）
                if resp.status_code in (429, 502, 503):
                    last_error = f"飞常准API HTTP错误: {resp.status_code}"
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        print(f"[VARIFLIGHT] {last_error}，{wait_time}s后重试({attempt+1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    return {"success": False, "error": last_error, "data": []}
                if resp.status_code != 200:
                    return {"success": False, "error": f"飞常准API HTTP错误: {resp.status_code}", "data": []}
                data = resp.json()
                # 支持 code=200 和 code=0 两种成功响应
                if data.get("code") == 200 or data.get("code") == 0:
                    inner = data.get("data", data)
                    # 🔴 检测 data 中的 error_code（API返回code=200但data中包含错误信息）
                    if isinstance(inner, dict) and inner.get("error_code"):
                        error_code = inner.get("error_code")
                        error_msg = inner.get("error", "未知错误")
                        # error_code=10 表示"暂无数据"，不是错误，返回空数据即可
                        if error_code == 10:
                            print(f"[VARIFLIGHT] 暂无数据: {error_msg}")
                            return {"success": True, "data": []}
                        # 其他error_code（如3=参数错误, 4=无权限）才是真正的错误
                        print(f"[VARIFLIGHT] API返回错误: error_code={error_code}, error={error_msg}")
                        return {"success": False, "error": f"飞常准API错误: {error_msg}", "data": []}
                    return {"success": True, "data": inner}
                resp_preview = str(data)[:200]
                print(f"[VARIFLIGHT] REST响应格式异常: {resp_preview}")
                return {"success": False, "error": f"飞常准API返回错误: {data.get('message', data.get('error', ''))}", "data": []}
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as e:
            last_error = f"连接失败: {type(e).__name__}"
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"[VARIFLIGHT] {last_error}，{wait_time}s后重试({attempt+1}/{max_retries})")
                await asyncio.sleep(wait_time)
                continue
        except Exception as e:
            last_error = f"REST调用失败: {type(e).__name__}: {str(e)[:100]}"
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"[VARIFLIGHT] {last_error}，{wait_time}s后重试({attempt+1}/{max_retries})")
                await asyncio.sleep(wait_time)
                continue
        break
    return {"success": False, "error": f"飞常准API调用失败: {last_error}", "data": []}


async def _try_flight_query(dep_iata: str, arr_iata: str, date: str) -> dict:
    """尝试查询航班（单个机场对），返回航班列表
    优先使用depcity/arrcity（城市代码），回退使用dep/arr（机场代码）
    策略间有延迟避免频率限制"""
    from feichangzhun_service import get_primary_airport_iata
    print(f"[VARIFLIGHT] 查询航班: {dep_iata}→{arr_iata}, date={date}")
    # 策略1：使用depcity/arrcity（城市代码）
    result = await _call_variflight("searchFlightsByDepArr", {"depcity": dep_iata, "arrcity": arr_iata, "date": date})
    # 处理dict格式错误响应（如{"error_code": 10, "error": "暂无数据"}）
    if result.get("success") and isinstance(result.get("data"), dict):
        error_info = result.get("data", {})
        if error_info.get("error_code"):
            print(f"[VARIFLIGHT] depcity查询返回错误: {error_info.get('error','')}，回退dep/arr")
            result = {"success": False, "data": []}
    if not result.get("success") or not result.get("data"):
        # 策略间延迟1秒，避免连续请求触发频率限制
        await asyncio.sleep(1)
        # 策略2：回退到dep/arr（机场代码）
        dep_airport = get_primary_airport_iata(dep_iata) or dep_iata
        arr_airport = get_primary_airport_iata(arr_iata) or arr_iata
        print(f"[VARIFLIGHT] 城市代码查询无结果，尝试机场代码: dep={dep_airport}, arr={arr_airport}")
        result = await _call_variflight("searchFlightsByDepArr", {"dep": dep_airport, "arr": arr_airport, "date": date})
    # 策略2也返回dict错误时，标记为无数据
    if result.get("success") and isinstance(result.get("data"), dict) and result.get("data", {}).get("error_code"):
        print(f"[VARIFLIGHT] dep/arr查询也返回错误: {result.get('data', {}).get('error','')}")
        result = {"success": False, "data": [], "error": result.get("data", {}).get("error", "")}
    print(f"[VARIFLIGHT] 航班查询结果: success={result.get('success')}, data_type={type(result.get('data')).__name__}, data_len={len(result.get('data', [])) if isinstance(result.get('data'), list) else 'N/A'}, error={result.get('error', '')[:100]}")
    flights = []
    if result.get("success"):
        raw_data = result.get("data", [])
        if isinstance(raw_data, list):
            for f in raw_data:
                dep_time = f.get("FlightDeptimePlanDate", f.get("dep", ""))
                arr_time = f.get("FlightArrtimePlanDate", f.get("arr", ""))
                if " " in str(dep_time):
                    dep_time = dep_time.split(" ")[-1][:5]
                if " " in str(arr_time):
                    arr_time = arr_time.split(" ")[-1][:5]
                raw_dur = f.get("FlightDuration", f.get("duration", ""))
                duration = _fmt_duration(raw_dur)
                flights.append({
                    "num": f.get("FlightNo", f.get("flightNumber", f.get("fnum", ""))),
                    "dep": dep_time,
                    "arr": arr_time,
                    "duration": duration,
                    "from_airport": f.get("FlightDepAirport", f.get("departureAirport", "")),
                    "to_airport": f.get("FlightArrAirport", f.get("arrivalAirport", "")),
                    "price": f.get("price", ""),
                    "airline": f.get("FlightCompany", f.get("airline", "")),
                    "_source": "飞常准实时API",
                })
    elif result.get("error"):
        print(f"[VARIFLIGHT] 航班查询失败: {result.get('error')[:200]}")
    return {"success": result.get("success"), "flights": flights, "error": result.get("error", "")}


async def search_flights_by_route(dep_city: str, arr_city: str, date: str) -> dict:
    """按出发/到达城市查询直飞航班（优先使用城市代码depcity/arrcity，兼容性更好）
    支持多机场查询和邻近城市校准：如果主机场无数据，尝试查询邻近枢纽城市的机场
    反复校准：主查询失败时自动扩展搜索范围"""
    from feichangzhun_service import get_primary_airport_iata, get_nearest_hub, get_iata, _clean_city_name

    # 优先使用城市代码（BJS/SHA等），兼容性更好
    dep_iata = get_iata(dep_city) or get_primary_airport_iata(dep_city)
    arr_iata = get_iata(arr_city) or get_primary_airport_iata(arr_city)

    if not dep_iata or not arr_iata:
        # 尝试获取邻近枢纽的机场代码
        dep_hub = get_nearest_hub(dep_city)
        arr_hub = get_nearest_hub(arr_city)
        if not dep_iata and dep_hub.get("has_hub"):
            dep_iata = dep_hub.get("hub_iata", "")
            print(f"[VARIFLIGHT] {dep_city}无机场，使用邻近枢纽{dep_hub['hub_city']}的机场代码: {dep_iata}")
        if not arr_iata and arr_hub.get("has_hub"):
            arr_iata = arr_hub.get("hub_iata", "")
            print(f"[VARIFLIGHT] {arr_city}无机场，使用邻近枢纽{arr_hub['hub_city']}的机场代码: {arr_iata}")
        if not dep_iata or not arr_iata:
            return {"success": False, "error": f"无法识别机场代码：{dep_city}或{arr_city}", "flights": []}

    # 主查询：使用城市代码（depcity/arrcity），_try_flight_query已内置双策略回退
    result = await _try_flight_query(dep_iata, arr_iata, date)
    flights = result.get("flights", [])

    # 🔴 反复校准：如果主查询无结果，尝试扩展搜索邻近城市机场
    if not flights:
        flights = await _calibrate_nearby_airport_search(dep_city, arr_city, date, dep_iata, arr_iata)
        # 标记为经过校准扩展搜索
        if flights:
            for f in flights:
                if "_calibrated" not in f:
                    f["_calibrated"] = True
                    f["_calibrated_source"] = "飞常准API邻近城市校准搜索"

    return {"success": bool(flights), "error": result.get("error", ""),
            "flights": flights, "_source": "飞常准实时API（含邻近城市校准）"}


async def _calibrate_nearby_airport_search(dep_city: str, arr_city: str, date: str,
                                           dep_iata: str, arr_iata: str) -> list:
    """反复校准：主查询无结果时，扩展搜索邻近城市机场
    尝试多种机场组合，策略间有延迟避免频率限制"""
    from feichangzhun_service import get_nearest_hub, get_primary_airport_iata, _clean_city_name
    all_flights = []

    # 策略1：尝试出发城市邻近枢纽
    dep_hub = get_nearest_hub(dep_city)
    if dep_hub.get("has_hub") and dep_hub["hub_iata"] != dep_iata:
        alt_dep_iata = dep_hub["hub_iata"]
        print(f"[CALIBRATE] 策略1：尝试出发邻近枢纽 {dep_hub['hub_city']}({alt_dep_iata}) → {arr_iata}")
        alt_result = await _try_flight_query(alt_dep_iata, arr_iata, date)
        alt_flights = alt_result.get("flights", [])
        if alt_flights:
            for f in alt_flights:
                f["_source"] = f"飞常准实时API（邻近枢纽{dep_hub['hub_city']}出发）"
            all_flights.extend(alt_flights)

    # 策略2：尝试到达城市邻近枢纽
    if not all_flights:
        await asyncio.sleep(1.5)
        arr_hub = get_nearest_hub(arr_city)
        if arr_hub.get("has_hub") and arr_hub["hub_iata"] != arr_iata:
            alt_arr_iata = arr_hub["hub_iata"]
            print(f"[CALIBRATE] 策略2：尝试{dep_iata} → 到达邻近枢纽 {arr_hub['hub_city']}({alt_arr_iata})")
            alt_result = await _try_flight_query(dep_iata, alt_arr_iata, date)
            alt_flights = alt_result.get("flights", [])
            if alt_flights:
                for f in alt_flights:
                    f["_source"] = f"飞常准实时API（邻近枢纽{arr_hub['hub_city']}到达）"
                all_flights.extend(alt_flights)

    # 策略3：尝试出发和到达都使用邻近枢纽
    if not all_flights:
        await asyncio.sleep(1.5)
        dep_hub2 = get_nearest_hub(dep_city)
        arr_hub2 = get_nearest_hub(arr_city)
        if dep_hub2.get("has_hub") and arr_hub2.get("has_hub"):
            alt_dep = dep_hub2["hub_iata"]
            alt_arr = arr_hub2["hub_iata"]
            if alt_dep != dep_iata or alt_arr != arr_iata:
                print(f"[CALIBRATE] 策略3：尝试双向邻近枢纽 {dep_hub2['hub_city']}({alt_dep}) → {arr_hub2['hub_city']}({alt_arr})")
                alt_result = await _try_flight_query(alt_dep, alt_arr, date)
                alt_flights = alt_result.get("flights", [])
                if alt_flights:
                    for f in alt_flights:
                        f["_source"] = f"飞常准实时API（双向邻近枢纽校准）"
                    all_flights.extend(alt_flights)

    if all_flights:
        print(f"[CALIBRATE] 邻近城市校准搜索完成，共找到{len(all_flights)}条航班")
    else:
        print(f"[CALIBRATE] 所有邻近城市校准策略均未找到航班")

    return all_flights


async def _calibrate_nearby_train_search(dep_city: str, arr_city: str, date: str) -> list:
    """高铁邻近城市扩展搜索：当主城市查无高铁线路时，扩展搜索邻近城市
    策略1：邻近出发城市 → 原到达城市
    策略2：原出发城市 → 邻近到达城市
    策略3：邻近出发城市 → 邻近到达城市"""
    from feichangzhun_service import get_train_nearby_cities, _clean_city_name

    dep_nearby = get_train_nearby_cities(dep_city)
    arr_nearby = get_train_nearby_cities(arr_city)
    all_trains = []

    # 策略1：尝试邻近出发城市 → 原到达城市（最多尝试3个邻近城市）
    if dep_nearby:
        for alt_dep in dep_nearby[:1]:
            print(f"[CALIBRATE-TRAIN] 策略1：尝试邻近出发 {alt_dep} → {arr_city}")
            result = await search_train_tickets(alt_dep, arr_city, date)
            alt_trains = result.get("trains", [])
            if alt_trains:
                for t in alt_trains:
                    t["_source"] = f"飞常准实时API（邻近城市{alt_dep}出发）"
                    t["_calibrated"] = True
                all_trains.extend(alt_trains)
                print(f"[CALIBRATE-TRAIN] 策略1成功：{alt_dep}→{arr_city} 找到{len(alt_trains)}条")
                break
            await asyncio.sleep(0.8)

    # 策略2：尝试原出发城市 → 邻近到达城市
    if not all_trains and arr_nearby:
        for alt_arr in arr_nearby[:1]:
            print(f"[CALIBRATE-TRAIN] 策略2：尝试 {dep_city} → 邻近到达 {alt_arr}")
            result = await search_train_tickets(dep_city, alt_arr, date)
            alt_trains = result.get("trains", [])
            if alt_trains:
                for t in alt_trains:
                    t["_source"] = f"飞常准实时API（邻近城市{alt_arr}到达）"
                    t["_calibrated"] = True
                all_trains.extend(alt_trains)
                print(f"[CALIBRATE-TRAIN] 策略2成功：{dep_city}→{alt_arr} 找到{len(alt_trains)}条")
                break
            await asyncio.sleep(0.8)

    # 策略3：尝试邻近出发 → 邻近到达（双向邻近）
    if not all_trains and dep_nearby and arr_nearby:
        for alt_dep in dep_nearby[:1]:
            found = False
            for alt_arr in arr_nearby[:1]:
                if alt_dep == alt_arr:
                    continue
                print(f"[CALIBRATE-TRAIN] 策略3：尝试双向邻近 {alt_dep} → {alt_arr}")
                result = await search_train_tickets(alt_dep, alt_arr, date)
                alt_trains = result.get("trains", [])
                if alt_trains:
                    for t in alt_trains:
                        t["_source"] = f"飞常准实时API（双向邻近城市校准）"
                        t["_calibrated"] = True
                    all_trains.extend(alt_trains)
                    print(f"[CALIBRATE-TRAIN] 策略3成功：{alt_dep}→{alt_arr} 找到{len(alt_trains)}条")
                    found = True
                    break
                await asyncio.sleep(0.8)
            if found:
                break

    if all_trains:
        print(f"[CALIBRATE-TRAIN] 邻近城市高铁搜索完成，共找到{len(all_trains)}条")
    else:
        print(f"[CALIBRATE-TRAIN] 所有邻近城市高铁策略均未找到车次")

    return all_trains


async def verify_flight_number(flight_num: str, date: str, dep: str = "", arr: str = "") -> dict:
    """验证航班号是否真实存在
    优先不带dep/arr参数查询（兼容性最好），失败后回退带dep/arr参数"""
    from feichangzhun_service import get_primary_airport_iata

    # 策略1：不带dep/arr参数查询（兼容性最好，不会因机场代码不匹配而漏查）
    result = await _call_variflight("searchFlightsByNumber", {"fnum": flight_num, "date": date})
    if result.get("success"):
        data = result.get("data", {})
        if isinstance(data, list) and len(data) > 0:
            for f in data:
                if f.get("FlightNo", f.get("flightNumber", f.get("fnum", ""))) == flight_num:
                    return {"valid": True, "data": f, "_source": "飞常准实时验证"}
        if isinstance(data, dict) and (data.get("FlightNo") or data.get("flightNumber") or data.get("fnum")):
            return {"valid": True, "data": data, "_source": "飞常准实时验证"}
        # 如果data是dict且包含error_code（如{"error_code": 10, "error": "暂无数据"}），回退到策略2
        if isinstance(data, dict) and data.get("error_code"):
            print(f"[VARIFLIGHT] 航班{flight_num}无dep/arr查询返回错误: {data.get('error','')}，尝试带dep/arr回退")

    # 策略2：带dep/arr参数回退查询
    if dep or arr:
        params = {"fnum": flight_num, "date": date}
        if dep:
            params["dep"] = get_primary_airport_iata(dep) or dep
        if arr:
            params["arr"] = get_primary_airport_iata(arr) or arr
        result2 = await _call_variflight("searchFlightsByNumber", params)
        if result2.get("success"):
            data2 = result2.get("data", {})
            if isinstance(data2, list) and len(data2) > 0:
                for f in data2:
                    if f.get("FlightNo", f.get("flightNumber", f.get("fnum", ""))) == flight_num:
                        return {"valid": True, "data": f, "_source": "飞常准实时验证"}
            if isinstance(data2, dict) and (data2.get("FlightNo") or data2.get("flightNumber") or data2.get("fnum")):
                return {"valid": True, "data": data2, "_source": "飞常准实时验证"}

    return {"valid": False, "error": "航班不存在或未找到", "_source": "飞常准实时验证"}


async def search_train_tickets(dep_city: str, arr_city: str, date: str) -> dict:
    """查询火车票（高铁/动车/普通列车）
    参数使用飞常准API文档规定的 cdep/carr（中文城市名）"""
    # 清洗城市名（去除"市"等后缀）
    from feichangzhun_service import _clean_city_name
    dep_clean = _clean_city_name(dep_city)
    arr_clean = _clean_city_name(arr_city)
    result = await _call_variflight("trainStanTicket", {
        "cdep": dep_clean, "carr": arr_clean, "date": date
    })
    trains = []
    if result.get("success"):
        raw_data = result.get("data", [])
        # 火车票API返回嵌套结构: {"code": 0, "data": [...]}
        if isinstance(raw_data, dict):
            raw_data = raw_data.get("data", raw_data)
        if isinstance(raw_data, list):
            for t in raw_data:
                # 取最低票价作为参考价格
                seat_lists = t.get("seatLists", [])
                price = ""
                if seat_lists:
                    prices = [s.get("seatPrice", 0) for s in seat_lists if s.get("seatPrice")]
                    if prices:
                        price = str(min(prices))
                trains.append({
                    "num": t.get("trainNumber", t.get("num", "")),
                    "dep": t.get("fromTime", t.get("departureTime", t.get("dep", ""))),
                    "arr": t.get("toTime", t.get("arrivalTime", t.get("arr", ""))),
                    "duration": _fmt_duration(t.get("useTime", t.get("duration", ""))),
                    "from_station": t.get("fromStation", t.get("departureStation", "")),
                    "to_station": t.get("toStation", t.get("arrivalStation", "")),
                    "type": t.get("trainType", ""),
                    "price": price,
                    "_source": "飞常准实时API",
                })
    else:
        print(f"[VARIFLIGHT] 火车票查询失败: {result.get('error', '')[:200]}")
    return {"success": result.get("success"), "error": result.get("error", ""),
            "trains": trains, "_source": "飞常准实时API"}


async def search_flight_itineraries(dep_city: str, arr_city: str, date: str) -> dict:
    """查询可购航班行程（含价格），使用城市代码（depCityCode/arrCityCode为城市代码）
    MCP searchFlightItineraries 参数: depCityCode(城市代码如BJS), arrCityCode(城市代码如SHA), depDate(日期)"""
    from feichangzhun_service import get_iata
    dep_city_code = get_iata(dep_city)
    arr_city_code = get_iata(arr_city)
    if not dep_city_code or not arr_city_code:
        return {"success": False, "error": "无法识别城市代码", "itineraries": []}
    result = await _call_variflight("searchFlightItineraries", {
        "depCityCode": dep_city_code, "arrCityCode": arr_city_code, "depDate": date
    })
    return {"success": result.get("success"), "error": result.get("error", ""),
            "data": result.get("data", {}), "_source": "飞常准实时API"}


async def get_flight_transfer(dep_city: str, arr_city: str, date: str) -> dict:
    """查询中转航班方案，使用城市代码（depcity/arrcity而非机场代码）
    MCP getFlightTransferInfo 参数: depcity(城市代码), arrcity(城市代码), depdate(日期)"""
    from feichangzhun_service import get_iata
    dep_city_code = get_iata(dep_city)
    arr_city_code = get_iata(arr_city)
    if not dep_city_code or not arr_city_code:
        return {"success": False, "error": "无法识别城市代码", "transfers": []}
    result = await _call_variflight("getFlightTransferInfo", {
        "depcity": dep_city_code, "arrcity": arr_city_code, "depdate": date
    })
    return {"success": result.get("success"), "error": result.get("error", ""),
            "data": result.get("data", {}), "_source": "飞常准实时API"}


async def get_full_route_data(dep_city: str, arr_city: str, date: str, user_transport_mode: str = "") -> dict:
    """获取完整路线数据：航班+火车票+中转方案（并行查询）
    飞常准API是唯一数据源，绝不使用本地预存数据
    user_transport_mode: 用户选择的出行方式，用于过滤不匹配的交通类型
    包含反复校准机制：对查询结果进行交叉验证，确保数据准确性"""
    import asyncio
    tasks = [
        search_flights_by_route(dep_city, arr_city, date),
        search_train_tickets(dep_city, arr_city, date),
        get_flight_transfer(dep_city, arr_city, date),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    flights_result = results[0] if not isinstance(results[0], Exception) else {"success": False, "flights": []}
    trains_result = results[1] if not isinstance(results[1], Exception) else {"success": False, "trains": []}
    transfer_result = results[2] if not isinstance(results[2], Exception) else {"success": False, "data": {}}

    flights = flights_result.get("flights", [])
    trains = trains_result.get("trains", [])

    # 🔴 反复校准：对查询到的航班进行抽样验证，确保数据真实性
    if flights and len(flights) > 0:
        flights = await _calibrate_flight_results(flights, date, dep_city, arr_city)
        print(f"[VARIFLIGHT] 航班校准完成：{len(flights)}条有效航班")

    # 🔴 反复校准：对查询到的火车票进行抽样验证
    if trains and len(trains) > 0:
        trains = await _calibrate_train_results(trains, date, dep_city, arr_city)
        print(f"[VARIFLIGHT] 火车票校准完成：{len(trains)}条有效车次")

    # 🔴 邻近城市扩展搜索：如果主城市无高铁数据，尝试搜索邻近城市
    if not trains:
        trains = await _calibrate_nearby_train_search(dep_city, arr_city, date)
        if trains:
            print(f"[VARIFLIGHT] 邻近城市高铁搜索成功：{len(trains)}条车次")

    has_flights = bool(flights)
    has_trains = bool(trains)
    has_data = has_flights or has_trains

    # 🔴 飞常准API是唯一数据源，绝不回退本地预存数据
    if not has_data:
        print(f"[VARIFLIGHT] API无{dep_city}-{arr_city}实时数据，不启用本地回退")

    return {
        "success": flights_result.get("success") or trains_result.get("success"),
        "flights": flights,
        "trains": trains,
        "transfers": transfer_result.get("data", {}),
        "_source": "飞常准实时API（已校准）",
        "_no_data": not has_data,
        "_no_data_message": "飞常准API暂无该路线实时班次数据，请优先选择大巴/自驾等低成本出行方式，或自行在携程查询实时航班" if not has_data else "",
        "flight_error": flights_result.get("error", ""),
        "train_error": trains_result.get("error", ""),
    }


async def _calibrate_flight_results(flights: list, date: str, dep_city: str, arr_city: str) -> list:
    """反复校准航班结果：抽样验证航班号真实性，过滤无效数据
    对前3个航班进行验证，确保返回的航班数据真实可靠"""
    if not flights:
        return flights
    calibrated = []
    # 对前3个航班进行抽样验证（避免过多API调用）
    sample_size = min(3, len(flights))
    verified_nums = set()
    for i in range(sample_size):
        f = flights[i]
        flight_num = f.get("num", "")
        if not flight_num or flight_num in verified_nums:
            calibrated.append(f)
            continue
        try:
            verify_result = await verify_flight_number(flight_num, date, dep_city, arr_city)
            if verify_result.get("valid"):
                f["_calibrated"] = True
                f["_calibrated_source"] = "飞常准API交叉验证通过"
                calibrated.append(f)
                verified_nums.add(flight_num)
                print(f"[CALIBRATE] 航班{flight_num}验证通过")
            else:
                print(f"[CALIBRATE] 航班{flight_num}验证失败：{verify_result.get('error','')}，已剔除")
                # 不加入calibrated，剔除无效航班
        except Exception as e:
            # 验证失败时保留原数据但标记为未校准
            f["_calibrated"] = False
            f["_calibrated_note"] = f"校准异常: {str(e)}"
            calibrated.append(f)
            print(f"[CALIBRATE] 航班{flight_num}校准异常: {e}")
    # 未抽样的航班直接保留
    for i in range(sample_size, len(flights)):
        flights[i]["_calibrated"] = False
        flights[i]["_calibrated_note"] = "未抽样校准"
        calibrated.append(flights[i])
    return calibrated


async def _calibrate_train_results(trains: list, date: str, dep_city: str, arr_city: str) -> list:
    """反复校准火车票结果：通过重新查询验证数据一致性
    对火车票进行二次查询校准，确保数据准确"""
    if not trains:
        return trains
    # 火车票校准：重新查询一次，对比结果一致性
    try:
        retry_result = await search_train_tickets(dep_city, arr_city, date)
        retry_trains = retry_result.get("trains", [])
        if retry_trains:
            retry_nums = {t.get("num", "") for t in retry_trains}
            calibrated = []
            for t in trains:
                train_num = t.get("num", "")
                if train_num in retry_nums:
                    t["_calibrated"] = True
                    t["_calibrated_source"] = "飞常准API二次查询校准通过"
                    calibrated.append(t)
                else:
                    print(f"[CALIBRATE] 车次{train_num}二次查询未找到，保留但标记未校准")
                    t["_calibrated"] = False
                    t["_calibrated_note"] = "二次查询未确认"
                    calibrated.append(t)
            return calibrated
    except Exception as e:
        print(f"[CALIBRATE] 火车票校准异常: {e}")
    # 校准失败时标记所有车次
    for t in trains:
        t["_calibrated"] = False
        t["_calibrated_note"] = "校准未执行"
    return trains