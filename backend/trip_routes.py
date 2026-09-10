"""行程生成与重新生成接口路由"""
import json
import time
import httpx
import asyncio
from collections import defaultdict, deque
from fastapi import APIRouter, Request
from models import TripRequest
from config import AMAP_KEY
from amap_service import amap_poi_search, amap_weather, amap_geocode, fill_coordinates
from deepseek_service import call_deepseek, build_trip_prompt, build_booking_prompt, build_regenerate_prompt, build_retry_prompt
from image_service import fill_images, fill_booking_images
from feichangzhun_service import judge_transport, search_flights, build_flight_query_text, get_nearest_hub, get_transfer_routes, get_amap_city_distance
from route_service import calculate_self_drive_plan, calculate_transit_to_station, calculate_station_to_hotel
from transport_validation import (
    _validate_transport_airports, _validate_cross_card_consistency,
    _check_transport_type_consistency, _ensure_transport_data,
    _detect_mixed_transport_from_text, _enforce_mixed_transport,
    _detect_transport_mode_from_text,
)

router = APIRouter()

# ============ 每IP限流（生成行程会消耗DeepSeek费用，防止被他人/爬虫刷爆） ============
_RATE_WINDOW = 3600  # 1小时窗口
_RATE_LIMIT = 5      # 每IP每小时最多5次
_rate_hits = defaultdict(deque)  # ip -> deque[timestamp]


def _check_rate_limit(ip: str) -> str:
    """返回空串=放行；否则返回错误提示"""
    now = time.time()
    q = _rate_hits[ip]
    while q and q[0] < now - _RATE_WINDOW:
        q.popleft()
    if len(q) >= _RATE_LIMIT:
        return f"操作过于频繁，请{_RATE_WINDOW // 60}分钟后再试（每IP每小时最多生成{_RATE_LIMIT}次）"
    q.append(now)
    return ""


@router.post("/api/generate-trip")
async def generate_trip(req: TripRequest, request: Request):
    """生成旅行攻略：高德POI+天气 → DeepSeek itinerary → DeepSeek booking → 返回完整JSON"""
    try:
        # 每IP限流：防止公开站点被刷爆DeepSeek费用
        rate_err = _check_rate_limit(request.client.host if request.client else "")
        if rate_err:
            return {"success": False, "error": rate_err}

        dest = req.destination.strip()
        days = max(1, min(31, req.days))
        start_date = req.start_date or ""
        end_date = req.end_date or ""

        # 1. 并行查询高德 POI 和天气
        poi_tasks = [
            amap_poi_search(f"{dest}景点", dest),
            amap_poi_search(f"{dest}美食", dest),
        ]
        if req.interests:
            poi_tasks.append(amap_poi_search(f"{dest}{' '.join(req.interests)}", dest))
        weather_task = amap_weather(dest)

        poi_results = await asyncio.gather(*poi_tasks)
        weather_data = await weather_task

        all_pois = []
        for r in poi_results:
            if isinstance(r, list):
                all_pois.extend(r)

        # 去重并按评分降序排列（高评分优先），取前20个
        seen_names = set()
        unique_pois = []
        for p in all_pois:
            if p["name"] not in seen_names:
                seen_names.add(p["name"])
                unique_pois.append(p)
        unique_pois.sort(key=lambda x: float(x["rating"]) if x["rating"] else 0, reverse=True)
        ranked_pois = unique_pois[:20]

        # 1.5 出发/返程交通规划（自驾/公共交通）
        transport_info = {}
        if req.departure_city:
            # 判断交通工具（优先使用高德API距离，回退预存数据）
            ti = judge_transport(req.departure_city, dest)
            # 高德API补充精确距离数据
            try:
                amap_dist = await get_amap_city_distance(req.departure_city, dest)
                if amap_dist.get("success") and amap_dist.get("distance_km", 0) > 0:
                    transport_info["amap_distance"] = amap_dist
                    dist_km = amap_dist["distance_km"]
                    ti["estimated_distance"] = f"{dist_km}km"
                    transport_info["transport"] = ti
            except Exception as e:
                print(f"[AMAP] 距离查询失败，使用预存数据: {e}")
                transport_info["transport"] = ti
            # 查询真实航班/火车班次数据（传入日期校准和高德API精确距离）
            amap_dist_km = transport_info.get("amap_distance", {}).get("distance_km", 0) if transport_info.get("amap_distance") else 0
            mode_map = {"plane": "飞机", "train": "高铁", "taxi": "打车", "selfdrive": "自驾"}
            user_mode_cn = mode_map.get(req.transport_mode, "") if req.transport_mode else ""
            # 🔴 飞常准API是唯一数据源，不调用本地预存数据
            schedule = {"flights": [], "trains": [], "_verified": "", "_source": "等待飞常准API查询",
                        "_no_data": True, "_no_data_note": "该路线未查询到实时班次，请等待飞常准API结果"}
            transport_info["route_schedule"] = schedule
            # 查询中转方案（当无直飞时）
            transfer_info = get_transfer_routes(req.departure_city, dest, start_date)
            transport_info["transfer_info"] = transfer_info
            # 检查出发/目的城市是否需要去邻近枢纽
            dep_hub = get_nearest_hub(req.departure_city)
            dest_hub = get_nearest_hub(dest)
            transport_info["dep_hub"] = dep_hub
            transport_info["dest_hub"] = dest_hub
            # 计算前往机场/车站的时间
            if ti.get("need_flight"):
                to_station = await calculate_transit_to_station(req.departure_city, "airport")
                from_station = await calculate_station_to_hotel(dest, "airport")
                transport_info["to_station"] = to_station
                transport_info["from_station"] = from_station
                # 生成航班查询指引
                transport_info["flight_query_text"] = build_flight_query_text(
                    req.departure_city, dest, start_date)
            else:
                to_station = await calculate_transit_to_station(req.departure_city, "train")
                from_station = await calculate_station_to_hotel(dest, "train")
                transport_info["to_station"] = to_station
                transport_info["from_station"] = from_station
                # 生成火车票查询指引
                transport_info["flight_query_text"] = build_flight_query_text(
                    req.departure_city, dest, start_date)
            # 如果用户在首页选择了特定出行方式，强制覆盖交通判断
            if req.transport_mode:
                user_mode = mode_map.get(req.transport_mode, "")
                if user_mode:
                    ti["mode"] = user_mode
                    ti["reason"] = f"用户指定{user_mode}出行"
                    ti["need_flight"] = (user_mode == "飞机")
                    transport_info["transport"] = ti
                    print(f"[TRANSPORT] 用户选择出行方式: {req.transport_mode} → {user_mode}")
            # 自驾模式：计算自驾路线
            if req.is_self_drive:
                sd_plan = await calculate_self_drive_plan(
                    req.departure_city, dest, start_date, end_date)
                transport_info["self_drive_plan"] = sd_plan
            # 飞常准API实时查询航班和火车票（多渠道验证）
            if ti.get("need_flight"):
                flight_result = await search_flights(req.departure_city, dest, start_date)
                transport_info["flight_data"] = flight_result.get("flights", [])
                transport_info["flight_query"] = flight_result.get("query", {})
            # 并行查询：飞常准API火车票 + 完整路线数据
            from variflight_service import get_full_route_data
            vf_result = await get_full_route_data(req.departure_city, dest, start_date, user_transport_mode=user_mode_cn)
            # 🔴 返程方向单独查询（目的地→出发地），用于确保返程交通数据正确
            vf_return = await get_full_route_data(dest, req.departure_city, end_date or start_date, user_transport_mode=user_mode_cn)
            transport_info["variflight_data"] = {
                "flights": vf_result.get("flights", []),
                "trains": vf_result.get("trains", []),
                "transfers": vf_result.get("transfers", {}),
                "success": vf_result.get("success"),
                "_no_data": vf_result.get("_no_data", False),
                "_no_data_message": vf_result.get("_no_data_message", ""),
                "source": vf_result.get("_source", "飞常准API"),
                # 🔴 返程方向数据
                "return_flights": vf_return.get("flights", []),
                "return_trains": vf_return.get("trains", []),
                "return_success": vf_return.get("success"),
            }
            # 🔴 飞常准API是唯一数据源，直接用API数据替换route_schedule
            if vf_result.get("success"):
                vf_flights_filtered = vf_result.get("flights", [])
                vf_trains_filtered = vf_result.get("trains", [])
                if user_mode_cn == "飞机":
                    vf_trains_filtered = []
                elif user_mode_cn == "高铁":
                    vf_flights_filtered = []
                schedule = {
                    "flights": vf_flights_filtered,
                    "trains": vf_trains_filtered,
                    "_verified": f"{start_date}",
                    "_source": "飞常准实时API（唯一数据源）",
                    "_no_data": not (vf_flights_filtered or vf_trains_filtered),
                    "_no_data_note": "飞常准API未返回该路线实时班次，请勿编造班次号" if not (vf_flights_filtered or vf_trains_filtered) else "",
                }
                transport_info["route_schedule"] = schedule

        # 2. 调用 DeepSeek 生成行程
        prompt = build_trip_prompt(dest, days, req.budget, req.interests, ranked_pois, weather_data,
                                   start_date, end_date, req.travelers, req.budget_type, req.pace,
                                   req.is_self_drive, req.departure_city, transport_info, req.transport_mode,
                                   req.travel_group)
        dynamic_tokens = 4000 + days * 500
        try:
            raw = await call_deepseek("你是一个专业的旅行规划师，只输出JSON格式数据。", prompt, dynamic_tokens)
        except httpx.HTTPStatusError as e:
            err_msg = "AI服务调用失败"
            if e.response.status_code == 402:
                err_msg = "DeepSeek API 余额不足，请充值后重试"
            elif e.response.status_code == 401:
                err_msg = "DeepSeek API Key 无效"
            elif e.response.status_code == 429:
                err_msg = "请求过于频繁，请稍后重试"
            return {"success": False, "error": err_msg}
        except RuntimeError as e:
            # 每日预算保护等自定义错误，向用户透出真实原因
            return {"success": False, "error": str(e)}
        except Exception:
            return {"success": False, "error": "AI服务调用失败，请重试"}

        raw_clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            trip_data = json.loads(raw_clean)
        except Exception:
            return {"success": False, "error": "AI返回数据格式异常，请重试"}

        # 后处理验证与重生成：飞常准API严格校验班次真实性，不准确则重新生成（最多1次，控制DeepSeek费用）
        validation_result = {"valid": True, "issues": [], "fabricated": []}
        max_retries = 1
        for retry_attempt in range(max_retries + 1):
            try:
                # 将用户选择的出行方式注入trip_data，供验证函数使用
                user_mode_map = {"plane": "飞机", "train": "高铁", "taxi": "打车", "selfdrive": "自驾"}
                trip_data["_transport_mode"] = user_mode_map.get(req.transport_mode, "") if req.transport_mode else ""
                validation_result = _validate_transport_airports(
                    trip_data, req.departure_city, dest, req.start_date,
                    transport_info.get("variflight_data"),
                    transport_info.get("route_schedule"),
                    user_transport_mode=trip_data.get("_transport_mode", ""))
            except Exception as e:
                print(f"[WARN] 交通验证异常（不影响主流程）: {e}")
                validation_result = {"valid": True, "issues": [], "fabricated": []}

            if validation_result.get("valid") and not validation_result.get("fabricated"):
                break  # 验证通过

            if retry_attempt < max_retries and (validation_result.get("fabricated") or not validation_result.get("valid")):
                print(f"[RETRY] 第{retry_attempt+1}/{max_retries}次重生成：验证不通过 - 编造班次={validation_result.get('fabricated', [])} 问题={validation_result.get('issues', [])}")
                retry_prompt = build_retry_prompt(
                    prompt, validation_result, transport_info,
                    req.departure_city, dest, req.start_date,
                    user_transport_mode=trip_data.get("_transport_mode", ""))
                try:
                    retry_raw = await call_deepseek(
                        "你是一个专业的旅行规划师，只输出JSON格式数据。必须严格使用飞常准API提供的真实班次！",
                        retry_prompt, dynamic_tokens)
                    retry_clean = retry_raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                    trip_data = json.loads(retry_clean)
                except Exception as e:
                    print(f"[WARN] 重生成失败: {e}")
                    break
            else:
                break

        # 确保往返交通信息始终存在
        try:
            _ensure_transport_data(trip_data, req.departure_city, dest,
                                   transport_info.get("variflight_data"), req.days,
                                   transport_info.get("route_schedule"))
        except Exception as e:
            print(f"[WARN] 交通数据补全失败（不影响主流程）: {e}")
        # 🔴 最终校验：确保交通类型与用户选择一致
        _check_transport_type_consistency(trip_data, trip_data.get("departure_transport", {}), trip_data.get("return_transport", {}))
        # 交叉验证：确保行程卡片与交通卡片时间一致性
        try:
            _validate_cross_card_consistency(trip_data)
        except Exception as e:
            print(f"[WARN] 交叉验证失败（不影响主流程）: {e}")

        # 3. 补全景点坐标（带异常保护，不影响主流程）
        try:
            await fill_coordinates(trip_data, dest)
        except Exception as e:
            print(f"[WARN] 坐标补全失败（不影响主流程）: {e}")

        # 4. 调用 DeepSeek 查询订票/酒店信息
        itinerary = trip_data.get("itinerary", [])
        booking_prompt = build_booking_prompt(dest, start_date, end_date, req.budget, itinerary,
                                              req.departure_city, transport_info,
                                              trip_data.get("departure_transport"),
                                              trip_data.get("return_transport"),
                                              transport_info.get("variflight_data"))
        try:
            booking_raw = await call_deepseek("你是一个旅行服务顾问，只输出JSON格式数据。", booking_prompt, 3000)
            booking_clean = booking_raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            booking_info = json.loads(booking_clean)
        except Exception:
            booking_info = {"flights": [], "hotels": [], "tickets": [], "booking_tips": []}

        # 6. 为酒店和门票补全坐标
        for hotel in booking_info.get("hotels", []):
            if hotel.get("name") and not hotel.get("location"):
                hotel["location"] = await amap_geocode(hotel["name"], dest)
        for change in booking_info.get("hotel_changes", []):
            if change.get("to_hotel") and not change.get("location"):
                change["location"] = await amap_geocode(change["to_hotel"], dest)
        for ticket in booking_info.get("tickets", []):
            if ticket.get("spot") and not ticket.get("location"):
                ticket["location"] = await amap_geocode(ticket["spot"], dest)

        # 7. 为景点和天气获取真实图片URL（带超时，不阻塞主流程）
        try:
            await asyncio.wait_for(fill_images(trip_data, dest), timeout=25.0)
        except (asyncio.TimeoutError, Exception):
            pass  # 图片获取超时或失败不阻塞攻略生成
        # 为酒店获取真实门面图片（带超时）
        try:
            await asyncio.wait_for(fill_booking_images(booking_info, dest), timeout=25.0)
        except (asyncio.TimeoutError, Exception):
            pass  # 酒店图片获取超时或失败不阻塞攻略生成

        trip_data["booking_info"] = booking_info
        trip_data["transport_info"] = transport_info
        return {"success": True, "data": trip_data}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"服务异常：{str(e)}"}


@router.post("/api/regenerate-trip")
async def regenerate_trip(request: Request):
    """根据用户新需求重新生成旅行计划，重点参考用户输入"""
    try:
        # 每IP限流：防止公开站点被刷爆DeepSeek费用
        rate_err = _check_rate_limit(request.client.host if request.client else "")
        if rate_err:
            return {"success": False, "error": rate_err}

        body = await request.json()
        user_input = body.get("user_input", "").strip()
        trip_data = body.get("trip_data", {})
        is_self_drive = body.get("is_self_drive", False)
        transport_mode = body.get("transport_mode", "")
        departure_city = body.get("departure_city", "")
        travel_group = body.get("travel_group", "")

        # 🔴 从用户输入文本中检测交通方式变更（覆盖首页选择的交通方式）
        detected_mode = _detect_transport_mode_from_text(user_input)
        # 🔴 检测混合交通方式（出发和返程不同交通工具）
        mixed_transport = _detect_mixed_transport_from_text(user_input)
        if mixed_transport.get("has_mixed"):
            print(f"[REGENERATE] 检测到混合交通方式: 出发={mixed_transport['departure_cn']}, 返程={mixed_transport['return_cn']}")
            transport_mode = mixed_transport["departure_mode"]  # 用出发方式作为主模式
            is_self_drive = (mixed_transport["departure_mode"] == "selfdrive")
        elif detected_mode:
            print(f"[REGENERATE] 从用户输入检测到交通方式变更: '{detected_mode}'，覆盖原transport_mode='{transport_mode}'")
            transport_mode = detected_mode
            is_self_drive = (detected_mode == "selfdrive")

        if not user_input:
            return {"success": False, "error": "请输入新计划需求"}

        dest = trip_data.get("destination", "")
        days = trip_data.get("days", 3)
        start_date = trip_data.get("start_date", "")
        end_date = trip_data.get("end_date", "")
        old_itinerary = trip_data.get("itinerary", [])

        if not dest:
            return {"success": False, "error": "缺少目的地信息"}

        # 获取最新天气
        try:
            weather_data = await amap_weather(dest)
        except Exception:
            weather_data = []

        # 获取交通判断信息（用于AI选择交通工具）
        transport_info = {}
        ti = {"mode": "公共交通", "reason": "默认出行方式", "need_flight": False}  # 默认值防止未定义
        if departure_city:
            try:
                ti = judge_transport(departure_city, dest)
                transport_info["transport"] = ti
                # 高德API补充精确距离数据
                try:
                    amap_dist = await get_amap_city_distance(departure_city, dest)
                    if amap_dist.get("success") and amap_dist.get("distance_km", 0) > 0:
                        transport_info["amap_distance"] = amap_dist
                        ti["estimated_distance"] = f"{amap_dist['distance_km']}km"
                        transport_info["transport"] = ti
                except Exception as e:
                    print(f"[AMAP] 重新生成时距离查询失败: {e}")
            except Exception:
                pass
            # 如果用户在首页选择了特定出行方式，强制覆盖交通判断（重新生成时）
            if transport_mode:
                mode_map = {"plane": "飞机", "train": "高铁", "taxi": "打车", "selfdrive": "自驾"}
                user_mode = mode_map.get(transport_mode, "")
                if user_mode:
                    ti["mode"] = user_mode
                    ti["reason"] = f"用户指定{user_mode}出行"
                    ti["need_flight"] = (user_mode == "飞机")
                    transport_info["transport"] = ti
                    print(f"[TRANSPORT] 重新生成：用户选择出行方式: {transport_mode} → {user_mode}")
            try:
                amap_dist_km_reg = transport_info.get("amap_distance", {}).get("distance_km", 0) if transport_info.get("amap_distance") else 0
                mode_map = {"plane": "飞机", "train": "高铁", "taxi": "打车", "selfdrive": "自驾"}
                user_mode_cn = mode_map.get(transport_mode, "") if transport_mode else ""
                # 🔴 飞常准API是唯一数据源，不调用本地预存数据
                schedule_reg = {"flights": [], "trains": [], "_verified": "", "_source": "等待飞常准API查询",
                                "_no_data": True, "_no_data_note": "该路线未查询到实时班次，请等待飞常准API结果"}
                transport_info["route_schedule"] = schedule_reg
            except Exception:
                pass
            try:
                transfer_info = get_transfer_routes(departure_city, dest, start_date)
                transport_info["transfer_info"] = transfer_info
            except Exception:
                pass
            try:
                dep_hub = get_nearest_hub(departure_city)
                transport_info["dep_hub"] = dep_hub
            except Exception:
                pass
            try:
                dest_hub = get_nearest_hub(dest)
                transport_info["dest_hub"] = dest_hub
            except Exception:
                pass
            # 飞常准API实时查询航班和火车票（多渠道验证，与generate_trip一致）
            if ti.get("need_flight"):
                try:
                    flight_result = await search_flights(departure_city, dest, start_date)
                    transport_info["flight_data"] = flight_result.get("flights", [])
                    transport_info["flight_query"] = flight_result.get("query", {})
                except Exception:
                    pass
            try:
                from variflight_service import get_full_route_data
                # 🔴 混合交通方式时查询全部数据（不限制单一交通方式）
                vf_mode = None if mixed_transport.get("has_mixed") else user_mode_cn
                vf_result = await get_full_route_data(departure_city, dest, start_date, user_transport_mode=vf_mode)
                # 🔴 返程方向单独查询（目的地→出发地），用于确保返程交通数据正确
                vf_return = await get_full_route_data(dest, departure_city, end_date or start_date, user_transport_mode=vf_mode)
                transport_info["variflight_data"] = {
                    "flights": vf_result.get("flights", []),
                    "trains": vf_result.get("trains", []),
                    "transfers": vf_result.get("transfers", {}),
                    "success": vf_result.get("success"),
                    "_no_data": vf_result.get("_no_data", False),
                    "_no_data_message": vf_result.get("_no_data_message", ""),
                    "source": vf_result.get("_source", "飞常准API"),
                    # 🔴 返程方向数据
                    "return_flights": vf_return.get("flights", []),
                    "return_trains": vf_return.get("trains", []),
                    "return_success": vf_return.get("success"),
                }
                # 🔴 飞常准API是唯一数据源，直接用API数据替换route_schedule
                if vf_result.get("success"):
                    vf_flights_filtered = vf_result.get("flights", [])
                    vf_trains_filtered = vf_result.get("trains", [])
                    # 🔴 混合交通方式时保留全部数据，否则按单一模式过滤
                    if not mixed_transport.get("has_mixed"):
                        if user_mode_cn == "飞机":
                            vf_trains_filtered = []
                        elif user_mode_cn == "高铁":
                            vf_flights_filtered = []
                    schedule = {
                        "flights": vf_flights_filtered,
                        "trains": vf_trains_filtered,
                        "_verified": f"{start_date}",
                        "_source": "飞常准实时API（唯一数据源）",
                        "_no_data": not (vf_flights_filtered or vf_trains_filtered),
                        "_no_data_note": "飞常准API未返回该路线实时班次，请勿编造班次号" if not (vf_flights_filtered or vf_trains_filtered) else "",
                    }
                    transport_info["route_schedule"] = schedule
            except Exception:
                pass

        # 构建regenerate prompt
        prompt = build_regenerate_prompt(dest, days, user_input, old_itinerary,
                                         weather_data, start_date, end_date,
                                         is_self_drive, departure_city, transport_info, transport_mode, mixed_transport, travel_group)
        # 调用DeepSeek
        try:
            raw = await call_deepseek("你是一个专业的旅行规划师，只输出JSON格式数据。", prompt, 6000)
        except httpx.HTTPStatusError as e:
            err_msg = "AI服务调用失败"
            if e.response.status_code == 402:
                err_msg = "DeepSeek API 余额不足，请充值后重试"
            elif e.response.status_code == 401:
                err_msg = "DeepSeek API Key 无效"
            elif e.response.status_code == 429:
                err_msg = "请求过于频繁，请稍后重试"
            return {"success": False, "error": err_msg}
        except RuntimeError as e:
            # 每日预算保护等自定义错误，向用户透出真实原因
            return {"success": False, "error": str(e)}
        except Exception:
            return {"success": False, "error": "AI服务调用失败，请重试"}

        # 解析DeepSeek返回的JSON
        raw_clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            new_trip = json.loads(raw_clean)
        except Exception:
            return {"success": False, "error": "AI返回数据格式异常，请重试"}

        # 后处理验证与重生成：飞常准API严格校验班次真实性，不准确则重新生成（最多1次，控制DeepSeek费用）
        validation_result = {"valid": True, "issues": [], "fabricated": []}
        max_retries = 1
        for retry_attempt in range(max_retries + 1):
            try:
                # 将用户选择的出行方式注入trip_data，供验证函数使用
                user_mode_map = {"plane": "飞机", "train": "高铁", "taxi": "打车", "selfdrive": "自驾"}
                # 🔴 混合交通方式时不设置_transport_mode（避免强制统一往返类型）
                if mixed_transport.get("has_mixed"):
                    new_trip["_transport_mode"] = ""
                    new_trip["_mixed_transport"] = mixed_transport
                else:
                    new_trip["_transport_mode"] = user_mode_map.get(transport_mode, "") if transport_mode else ""
                validation_result = _validate_transport_airports(
                    new_trip, departure_city, dest, start_date,
                    transport_info.get("variflight_data"),
                    transport_info.get("route_schedule"),
                    user_transport_mode=new_trip.get("_transport_mode", ""))
            except Exception as e:
                print(f"[WARN] 重新生成交通验证异常（不影响主流程）: {e}")
                validation_result = {"valid": True, "issues": [], "fabricated": []}

            if validation_result.get("valid") and not validation_result.get("fabricated"):
                break

            if retry_attempt < max_retries and (validation_result.get("fabricated") or not validation_result.get("valid")):
                print(f"[RETRY] 重新生成第{retry_attempt+1}/{max_retries}次重试：验证不通过 - 编造班次={validation_result.get('fabricated', [])} 问题={validation_result.get('issues', [])}")
                retry_prompt = build_retry_prompt(
                    prompt, validation_result, transport_info,
                    departure_city, dest, start_date,
                    user_transport_mode=new_trip.get("_transport_mode", ""))
                try:
                    retry_raw = await call_deepseek(
                        "你是一个专业的旅行规划师，只输出JSON格式数据。必须严格使用飞常准API提供的真实班次！",
                        retry_prompt, 6000)
                    retry_clean = retry_raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                    new_trip = json.loads(retry_clean)
                except Exception as e:
                    print(f"[WARN] 重新生成重试失败: {e}")
                    break
            else:
                break

        # 🔴 混合交通方式：强制校验出发和返程交通类型
        _enforce_mixed_transport(new_trip, mixed_transport)
        # 确保往返交通信息始终存在
        try:
            _ensure_transport_data(new_trip, departure_city, dest,
                                   transport_info.get("variflight_data"), days,
                                   transport_info.get("route_schedule"))
        except Exception as e:
            print(f"[WARN] 重新生成交通数据补全失败（不影响主流程）: {e}")
        # 🔴 最终校验：确保交通类型与用户选择一致
        _check_transport_type_consistency(new_trip, new_trip.get("departure_transport", {}), new_trip.get("return_transport", {}))
        # 交叉验证：确保行程卡片与交通卡片时间一致性
        try:
            _validate_cross_card_consistency(new_trip)
        except Exception as e:
            print(f"[WARN] 重新生成交叉验证失败（不影响主流程）: {e}")

        # 补全坐标
        try:
            await fill_coordinates(new_trip, dest)
        except Exception:
            pass  # 坐标补全失败不影响主流程

        # 获取图片
        try:
            await asyncio.wait_for(fill_images(new_trip, dest), timeout=25.0)
        except (asyncio.TimeoutError, Exception):
            pass

        # 重新生成booking_info（基于新itinerary和交通班次，确保与上方的交通安排是同一班次）
        new_itinerary = new_trip.get("itinerary", [])
        booking_budget = trip_data.get("budget", "中等")
        booking_prompt = build_booking_prompt(dest, start_date, end_date, booking_budget, new_itinerary,
                                              departure_city, transport_info,
                                              new_trip.get("departure_transport"),
                                              new_trip.get("return_transport"),
                                              transport_info.get("variflight_data"))
        try:
            booking_raw = await call_deepseek("你是一个旅行服务顾问，只输出JSON格式数据。", booking_prompt, 3000)
            booking_clean = booking_raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            booking_info = json.loads(booking_clean)
        except Exception:
            booking_info = trip_data.get("booking_info", {})
        # 为booking_info中的酒店和门票补全坐标
        for hotel in booking_info.get("hotels", []):
            if hotel.get("name") and not hotel.get("location"):
                try:
                    hotel["location"] = await amap_geocode(hotel["name"], dest)
                except Exception:
                    pass
        for change in booking_info.get("hotel_changes", []):
            if change.get("to_hotel") and not change.get("location"):
                try:
                    change["location"] = await amap_geocode(change["to_hotel"], dest)
                except Exception:
                    pass
        for ticket in booking_info.get("tickets", []):
            if ticket.get("spot") and not ticket.get("location"):
                try:
                    ticket["location"] = await amap_geocode(ticket["spot"], dest)
                except Exception:
                    pass
        # 为booking_info中的酒店获取图片
        try:
            await asyncio.wait_for(fill_booking_images(booking_info, dest), timeout=25.0)
        except (asyncio.TimeoutError, Exception):
            pass
        new_trip["booking_info"] = booking_info
        new_trip["transport_info"] = transport_info

        return {"success": True, "data": new_trip}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"重新生成失败：{str(e)}"}


@router.post("/api/verify-flight")
async def verify_flight(request: Request):
    """验证航班号/车次号是否真实存在（飞常准API）"""
    try:
        body = await request.json()
        flight_num = body.get("flight_number", "").strip()
        date = body.get("date", "").strip()
        dep = body.get("dep", "").strip()
        arr = body.get("arr", "").strip()
        if not flight_num or not date:
            return {"success": False, "valid": False, "error": "缺少航班号或日期"}
        from variflight_service import verify_flight_number
        result = await verify_flight_number(flight_num, date, dep, arr)
        return {"success": True, "valid": result.get("valid", False),
                "data": result.get("data", {}), "source": result.get("_source", "")}
    except Exception as e:
        return {"success": False, "valid": False, "error": str(e)}