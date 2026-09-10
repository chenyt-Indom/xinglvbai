"""DeepSeek AI 服务：API调用和提示词构建"""
import asyncio
import json
import os
import time
import httpx
from config import DEEPSEEK_KEY, DEEPSEEK_URL, DEEPSEEK_DAILY_BUDGET

# ============ 每日费用保护（防止无人使用时被刷接口导致费用失控） ============
_USAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".deepseek_usage.json")
# 按峰值价格保守估算（元/百万tokens）：输入未命中3元、输出9元，宁可早停也不超支
_INPUT_PRICE, _OUTPUT_PRICE = 3.0, 9.0


def _load_daily_usage() -> dict:
    try:
        with open(_USAGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("date") == time.strftime("%Y-%m-%d"):
            return data
    except Exception:
        pass
    return {"date": time.strftime("%Y-%m-%d"), "cost": 0.0, "calls": 0, "tokens_in": 0, "tokens_out": 0}


def _save_daily_usage(data: dict) -> None:
    try:
        with open(_USAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass


def check_deepseek_budget() -> str:
    """检查今日费用是否超限；未超限返回空串，超限返回错误提示"""
    usage = _load_daily_usage()
    if usage["cost"] >= DEEPSEEK_DAILY_BUDGET:
        return (f"DeepSeek今日预算已用完（¥{usage['cost']:.2f}，上限¥{DEEPSEEK_DAILY_BUDGET:.2f}），"
                f"请明天再试或调高DEEPSEEK_DAILY_BUDGET")
    return ""


def record_deepseek_usage(prompt_tokens: int, completion_tokens: int) -> None:
    """按usage记录当日费用（估算值），并持久化到本地文件"""
    usage = _load_daily_usage()
    est = prompt_tokens / 1_000_000 * _INPUT_PRICE + completion_tokens / 1_000_000 * _OUTPUT_PRICE
    usage["cost"] += est
    usage["calls"] += 1
    usage["tokens_in"] += prompt_tokens
    usage["tokens_out"] += completion_tokens
    _save_daily_usage(usage)
    print(f"[DEEPSEEK] 本次消耗 tokens(in={prompt_tokens}, out={completion_tokens}) 估算+¥{est:.4f} | "
          f"今日累计 ¥{usage['cost']:.2f}/{DEEPSEEK_DAILY_BUDGET:.2f} 共{usage['calls']}次")


async def call_deepseek(system_prompt: str, user_prompt: str, max_tokens: int = 4000) -> str:
    """调用 DeepSeek API，返回生成的文本内容（带每日费用上限保护，超时不重试避免重复扣费）"""
    # 每日预算保护：超限直接拒绝，不发起请求
    budget_err = check_deepseek_budget()
    if budget_err:
        raise RuntimeError(budget_err)

    last_error = None
    # 说明：超时后DeepSeek服务端可能已生成完成并按量计费，此时重试会造成同一内容重复扣费，
    # 因此超时不再重试（连接类错误不影响计费，仍重试3次）
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(
                    DEEPSEEK_URL,
                    headers={
                        "Authorization": f"Bearer {DEEPSEEK_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.7, "max_tokens": max_tokens,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            content = data["choices"][0]["message"]["content"]
            # 记录本次费用（DeepSeek返回的usage为实际计费token数）
            usage = data.get("usage", {})
            record_deepseek_usage(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))
            return content
        except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
            # 网络连接错误（请求未送达，不会计费）：重试
            last_error = e
            if attempt < 2:
                wait_s = (attempt + 1) * 3  # 3s, 6s 递增等待
                print(f"[DEEPSEEK] 第{attempt+1}次调用失败({type(e).__name__})，{wait_s}秒后重试...")
                await asyncio.sleep(wait_s)
        except httpx.TimeoutException as e:
            # 超时：服务端可能已生成并计费，重试=重复扣费，直接抛出
            raise e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < 2:
                # 限流：不产生费用，按1/3/9秒递增等待后重试
                last_error = e
                wait_s = 3 ** (attempt + 1)
                print(f"[DEEPSEEK] 请求过于频繁(429)，{wait_s}秒后重试...")
                await asyncio.sleep(wait_s)
            elif e.response.status_code >= 500 and attempt < 2:
                last_error = e
                wait_s = (attempt + 1) * 3
                print(f"[DEEPSEEK] 第{attempt+1}次调用失败(HTTP {e.response.status_code})，{wait_s}秒后重试...")
                await asyncio.sleep(wait_s)
            else:
                raise  # 4xx错误直接抛出
    # 3次重试全部失败
    raise last_error


def build_trip_prompt(dest: str, days: int, budget: str, interests: list,
                      poi_data: list, weather_data: list, start_date: str, end_date: str,
                      travelers: int = 1, budget_type: str = "total", pace: int = 50,
                      is_self_drive: bool = False, departure_city: str = "",
                      transport_info: dict = None, transport_mode: str = "",
                      travel_group: str = "") -> str:
    """构建行程生成提示词，包含POI数据、天气、人数、预算类型、节奏、自驾和JSON格式要求"""
    interest_str = "、".join(interests) if interests else "综合体验"
    poi_str = "\n".join([
        f"- {p['name']}（{p.get('type','景点')}，坐标：{p.get('location','')}，评分：{p.get('rating','无')}，商区：{p.get('business_area','')}）"
        for p in poi_data[:20]
    ]) if poi_data else "（请根据常识推荐）"
    weather_str = "\n".join([
        f"  {w['date']}: {w['dayweather']} {w['daytemp']}°C~{w['nighttemp']}°C"
        for w in weather_data[:days+2]  # 多取2天天气用于出发/返程
    ]) if weather_data else "（请根据常识判断）"
    date_info = f"\n【出行日期】{start_date} 至 {end_date}（共{days}天）" if start_date else ""

    # 🔴 用户指定出行方式：最高优先级，覆盖所有自动判断
    transport_mode_map = {"plane": "飞机", "train": "高铁", "taxi": "打车", "selfdrive": "自驾"}
    user_transport_mode = transport_mode_map.get(transport_mode, "")
    if user_transport_mode:
        is_self_drive = True  # 用户指定了方式，按用户选择处理

    # 出发/返程交通信息
    transport_section = ""
    if departure_city:
        if user_transport_mode:
            # 用户指定了具体出行方式：强制使用该方式
            transport_section = f"\n【🔴 用户指定出行方式 - 最高优先级 - 绝对不可更改】{user_transport_mode}"
            if user_transport_mode == "自驾":
                transport_section += f"\n【出行方式】自驾从{departure_city}出发"
                if transport_info:
                    sd = transport_info.get("self_drive_plan", {})
                    if sd.get("total_distance"):
                        transport_section += f"\n【自驾路线】全程{sd.get('total_distance','')}，预计驾驶{sd.get('total_duration_min',0)}分钟，当前路况{sd.get('traffic','畅通')}"
                    if sd.get("stopover"):
                        so = sd["stopover"]
                        transport_section += f"\n【沿途过夜】{so.get('suggestion','')}，第一天驾驶{so.get('day1_drive','')}，第二天驾驶{so.get('day2_drive','')}"
                    transport_section += f"\n【出发建议】{sd.get('suggested_departure','')}"
                transport_section += "\n第一天和最后一天需包含出发/返程自驾规划，计算好驾驶时间，不要安排太紧凑，留足休息时间"
            elif user_transport_mode == "飞机":
                transport_section += f"\n【🔴 不可更改的出行方式】用户指定飞机出行，departure_transport和return_transport的type字段必须为'飞机'，绝对禁止填写'高铁'、'火车'、'动车'或其他任何交通方式！"
                transport_section += f"\n必须安排{departure_city}到{dest}的往返航班"
                transport_section += "\n必须从飞常准API实时数据中选择真实航班，不可编造航班号"
                transport_section += "\n即使飞常准API无数据，type也必须填'飞机'，flight_number可留空，duration填估算飞行时间"
                # 如果出发城市没有机场，提示可以从邻近城市出发
                if transport_info:
                    dep_hub = transport_info.get("dep_hub", {})
                    if dep_hub.get("has_hub"):
                        transport_section += f"\n【⚠️ 出发城市无机场】{departure_city}本地无大型机场，但可以从邻近城市{dep_hub['hub_city']}的{dep_hub.get('hub_airport', '机场')}出发"
                    dest_hub = transport_info.get("dest_hub", {})
                    if dest_hub.get("has_hub"):
                        transport_section += f"\n【⚠️ 目的地无机场】{dest}本地无大型机场，航班将降落在邻近城市{dest_hub['hub_city']}的{dest_hub.get('hub_airport', '机场')}"
            elif user_transport_mode == "高铁":
                transport_section += f"\n【🔴 不可更改的出行方式】用户指定高铁出行，departure_transport和return_transport的type字段必须为'高铁'，绝对禁止填写'飞机'、'航班'或其他任何交通方式！"
                transport_section += f"\n必须安排{departure_city}到{dest}的往返高铁/动车"
                transport_section += "\n必须从飞常准API实时数据中选择真实车次，不可编造车次号"
                transport_section += "\n即使飞常准API无数据，type也必须填'高铁'，flight_number可留空，duration填估算高铁时间"
            elif user_transport_mode == "打车":
                transport_section += f"\n【出行方式】用户指定打车出行，{departure_city}到{dest}使用打车/出租车"
                transport_section += "\n无需安排航班或高铁，只标注打车预估费用和耗时即可"
        elif is_self_drive:
            transport_section = f"\n【出行方式】自驾从{departure_city}出发"
            if transport_info:
                sd = transport_info.get("self_drive_plan", {})
                transport_section += f"\n【自驾路线】全程{sd.get('total_distance','')}，预计驾驶{sd.get('total_duration_min',0)}分钟，当前路况{sd.get('traffic','畅通')}"
                if sd.get("stopover"):
                    so = sd["stopover"]
                    transport_section += f"\n【沿途过夜】{so.get('suggestion','')}，第一天驾驶{so.get('day1_drive','')}，第二天驾驶{so.get('day2_drive','')}"
                transport_section += f"\n【出发建议】{sd.get('suggested_departure','')}"
            transport_section += "\n第一天和最后一天需包含出发/返程自驾规划，计算好驾驶时间，不要安排太紧凑，留足休息时间"
        else:
            transport_section = f"\n【出发城市】{departure_city}"
            if transport_info:
                ti = transport_info.get("transport", {})
                transport_section += f"\n【推荐交通】{ti.get('mode','')} - {ti.get('reason','')}"
                # 邻近枢纽提示
                dep_hub = transport_info.get("dep_hub", {})
                dest_hub = transport_info.get("dest_hub", {})
                if dep_hub.get("has_hub"):
                    transport_section += f"\n【出发地枢纽】{dep_hub['note']}"
                if dest_hub.get("has_hub"):
                    transport_section += f"\n【目的地枢纽】{dest_hub['note']}"
                if transport_info.get("to_station"):
                    ts = transport_info["to_station"]
                    transport_section += f"\n【前往机场/车站】驾车约{ts.get('drive_min',0)}分钟，公交约{ts.get('transit_min',0)}分钟，{ts.get('advice','')}"
                if transport_info.get("from_station"):
                    fs = transport_info["from_station"]
                    transport_section += f"\n【到达后交通】从机场/车站到酒店，驾车约{fs.get('drive_min',30)}分钟，公交约{fs.get('transit_min',45)}分钟"
                if transport_info.get("flight_query_text"):
                    transport_section += f"\n{transport_info['flight_query_text']}"
                # 真实航班/火车班次数据
                schedule = transport_info.get("route_schedule", {})
                if schedule.get("flights") or schedule.get("trains"):
                    transport_section += "\n【以下是真实可选的航班/火车班次，必须从中选择，不可随意编造！】"
                    if schedule.get("flights"):
                        transport_section += "\n可选航班："
                        for f in schedule["flights"]:
                            transport_section += f"\n  ✈ {f['num']}：{f['dep']}出发 → {f['arr']}到达（{f['duration']}）"
                    if schedule.get("trains"):
                        transport_section += "\n可选火车/高铁："
                        for t in schedule["trains"]:
                            transport_section += f"\n  🚄 {t['num']}：{t['dep']}出发 → {t['arr']}到达（{t['duration']}）"
                    transport_section += "\n【强制要求-班次严格匹配】你必须从以上列出的真实班次中选择一个，并且严格使用该班次的全部信息！"
                    transport_section += "\n  ① flight_number字段：必须填写所选班次的航班号/车次号（如CA1515、G1），禁止编造"
                    transport_section += "\n  ② departure_time字段：必须填写所选班次的出发时间（如08:00），与上面列出的时间一致"
                    transport_section += "\n  ③ arrival_time字段：必须填写所选班次的到达时间（如10:05），与上面列出的时间一致"
                    transport_section += "\n  ④ duration字段：必须使用'航班号/车次号 + 耗时'格式（如'G1次 4h29min'或'CA1515 2h10min'），不可只写耗时"
                    transport_section += "\n  ⑤ 以上4个字段必须来自同一个班次，不可交叉混用！如果选择的班次是早上8:00出发，departure_time就必须是8:00，不能改成其他时间"
                    transport_section += "\n  ⑥ 返程如果走相同路线，也必须从以上班次中反向选择合适时间的班次，同样严格匹配"
                    transport_section += "\n  ⑦ 【年份校验-最高优先级】以上班次为本年实时运营数据，出行日期必须在本年或次年！如果出行日期是其他年份，说明数据可能不适用，此时flight_number必须留空，station也必须留空，只填写交通方式类型和估算耗时！"
                    transport_section += "\n  ⑧ 【机场名规则】机场名以飞常准API返回的为准，API返回的机场名即为有效运营机场！"
                    # 无飞常准API数据时的致命警告
                    if schedule.get("_no_data"):
                        transport_section += "\n【致命警告-飞常准API无实时数据】该路线飞常准API未返回实时班次！"
                        transport_section += "\n  ① flight_number字段必须留空字符串''，绝对禁止编造任何航班号/车次号！"
                        transport_section += "\n  ② station字段必须留空字符串''，绝对禁止编造任何机场/车站名！"
                        if user_transport_mode:
                            transport_section += f"\n  ③ 交通方式type必须填写'{user_transport_mode}'（用户已指定），绝对禁止改为其他交通方式！"
                        else:
                            transport_section += "\n  ③ 只填写交通方式类型（如'飞机'或'高铁'）"
                        transport_section += "\n  ④ duration只写估算耗时（如'约3小时'）"
                        transport_section += "\n  ⑤ 在note中建议用户自行在携程查询实时航班号"
                # 中转方案
                transfer_info = transport_info.get("transfer_info", {})
                if transfer_info.get("transfer_options"):
                    transport_section += "\n【中转方案参考】如果无合适直飞航班，可考虑以下中转方案："
                    transfer_note = transfer_info.get("_note", "")
                    if transfer_note:
                        transport_section += f"\n  {transfer_note}"
                    for to in transfer_info["transfer_options"]:
                        transport_section += f"\n  方案：经{to['transfer_city']}中转"
                        if to["leg1"]["flights"]:
                            transport_section += f"\n    第一段 {to['leg1']['from']}→{to['leg1']['to']}："
                            for f in to["leg1"]["flights"][:2]:
                                transport_section += f"\n      ✈ {f['num']}：{f['dep']}→{f['arr']}（{f['duration']}）"
                        if to["leg2"]["flights"]:
                            transport_section += f"\n    第二段 {to['leg2']['from']}→{to['leg2']['to']}："
                            for f in to["leg2"]["flights"][:2]:
                                transport_section += f"\n      ✈ {f['num']}：{f['dep']}→{f['arr']}（{f['duration']}）"
                        transport_section += f"\n    中转提示：{to['note']}"
                # 飞常准API实时数据（唯一数据源，优先级最高，覆盖静态数据）
                vf_data = transport_info.get("variflight_data", {})
                if vf_data.get("success"):
                    vf_flights = vf_data.get("flights", [])
                    vf_trains = vf_data.get("trains", [])
                    # 🔴 根据用户交通方式过滤：用户选飞机时不显示火车，选高铁时不显示航班
                    if user_transport_mode == "飞机":
                        vf_trains = []
                    elif user_transport_mode == "高铁":
                        vf_flights = []
                    if vf_flights or vf_trains:
                        transport_section += "\n\n【🔴 飞常准API实时数据 - 唯一权威数据源 - 必须100%严格使用！】"
                        transport_section += "\n以下班次为飞常准API实时查询结果，是当前日期唯一真实可购的班次。之前的静态数据全部作废，只使用以下数据！"
                        if vf_flights:
                            transport_section += "\n🚀 实时航班（飞常准API验证，真实可购）："
                            for f in vf_flights[:10]:
                                airport_info = ""
                                if f.get('from_airport') and f.get('to_airport'):
                                    airport_info = f"  [{f.get('from_airport','')} → {f.get('to_airport','')}]"
                                price_info = f"  ¥{f.get('price','')}" if f.get('price') else ""
                                transport_section += f"\n  ✈ {f['num']}：{f.get('dep','')}→{f.get('arr','')}（{f.get('duration','')}）{airport_info}{price_info}"
                        if vf_trains:
                            transport_section += "\n🚄 实时火车票（飞常准API验证，真实可购）："
                            for t in vf_trains[:10]:
                                station_info = ""
                                if t.get('from_station') and t.get('to_station'):
                                    station_info = f"  [{t.get('from_station','')} → {t.get('to_station','')}]"
                                price_info = f"  ¥{t.get('price','')}" if t.get('price') else ""
                                transport_section += f"\n  🚄 {t['num']}：{t.get('dep','')}→{t.get('arr','')}（{t.get('duration','')}）{station_info}{price_info}"
                        transport_section += "\n\n【🔴 飞常准API强制使用规则 - 违反将导致行程无效！】"
                        transport_section += "\n  ① 以上飞常准API数据是唯一可用的真实班次，之前任何静态数据全部忽略！"
                        transport_section += "\n  ② flight_number必须从上表中精确选择，一字不差！"
                        transport_section += "\n  ③ departure_time必须与上表中该班次的dep时间完全一致！"
                        transport_section += "\n  ④ arrival_time必须与上表中该班次的arr时间完全一致！"
                        transport_section += "\n  ⑤ duration必须使用'班次号 + 耗时'格式（如'MU5102 2h5min'），耗时与上表一致！"
                        transport_section += "\n  ⑥ station必须使用上表中的机场/车站名，禁止编造！飞常准API返回的机场名即为有效名！"
                        transport_section += "\n  ⑦ 去程和返程各选一个班次，4个字段（flight_number/departure_time/arrival_time/duration）必须来自同一班次！"
                        transport_section += "\n  ⑧ 🔴 绝对禁止编造班次号！如果上表中没有任何班次数据，flight_number必须留空字符串''！"
                        transport_section += "\n  ⑧ 如果上表中有多个班次，优先选择上午出发的班次（休闲节奏选08:00后），确保到达后有充足时间！"
                        transport_section += "\n  ⑨ 绝对禁止编造任何不在上表中的航班号/车次号！绝对禁止修改上表中的时间！"
            transport_section += f"\n第一天必须包含从{departure_city}出发前往{dest}的交通规划，最后一天必须包含从{dest}返回{departure_city}的交通规划。"
            transport_section += """
【重要-出发时间灵活规则】
1. 出发时间不固定，需根据当天航班/车次时刻表决定，可以是上午、下午、傍晚甚至晚上出发
2. 如果选择飞机：需先查询当天所有直飞航班，选择最合适的时间段（考虑票价、时长、到达时间）
3. 如果选择火车：需查询当天高铁/动车班次，优先选择耗时短、到达时间合理的车次
4. 根据用户偏好选择合适的交通方式，有飞机/高铁时优先推荐飞机/高铁
5. 到达时间规划：如果下午到达，当天可安排1个晚间景点；如果傍晚/晚上到达，当天仅安排入住酒店
6. 跨天到达处理：如果航班/火车在次日凌晨到达，需在departure_transport中标注"次日XX:XX到达"，并规划好到达后的交通和住宿
7. 必须计算从机场/车站到酒店的交通方式、时间和费用（打车/地铁/机场大巴），在departure_transport的note中说明
8. 预留充足缓冲：飞机起飞前2小时到达机场，火车发车前1小时到达车站，加上从住处到机场/车站的时间
9. 返程同理：最后一天需根据返程航班/车次时间倒推最晚出发时间，确保不误机/误车
10. 如果出发当天没有合适的航班/车次，可考虑提前一天出发，并在第一天安排轻松的活动
11. 【大巴/自驾出行】如果选择大巴或自驾，不需要填写flight_number（留空），只需要填写type、出发时间、预估耗时即可，station填写大巴站或直接留空"""
            transport_section += "\n【🔴 行程卡片与交通卡片一致性-最高优先级】"
            transport_section += "\n  出发交通卡片和Day1行程卡片必须时间连贯，不可相互独立！"
            transport_section += "\n  ① Day1的第一个景点开始时间必须 ≥ departure_transport.arrival_time + 从机场/车站到酒店的时间（约1小时）"
            transport_section += "\n  ② 如果departure_transport.arrival_time是下午（如14:00），Day1上午和下午不应有景点，最多安排1个晚间景点"
            transport_section += "\n  ③ 如果departure_transport.arrival_time是晚上（如20:00），Day1全天不应有景点，只能安排入住酒店"
            transport_section += "\n  ④ 返程交通卡片和最后一天行程卡片必须时间连贯！最后一天最后一个景点的结束时间必须 ≤ return_transport.departure_time - 酒店到车站时间 - 提前到站缓冲（飞机2h/火车1h）"
            transport_section += "\n  ⑤ 如果return_transport.departure_time是上午（如10:00），最后一天不应安排任何景点"
            transport_section += "\n  ⑥ 如果return_transport.departure_time是下午（如15:00），最后一天最多安排1个上午景点"
            transport_section += "\n  ⑦ departure_transport和return_transport中的时间必须与Day1/最后一天的行程时间互相呼应，不可出现矛盾！"
            transport_section += "\n【跨天到达示例】如选择晚上20:00航班，飞行2小时，22:00到达机场，打车30分钟到酒店，则第一天行程为：上午准备出发→下午前往机场→晚上航班→到达后入住酒店，不安排游览。"

    # 人数描述
    people_info = ""
    if travelers > 1:
        people_info = f"\n【旅行人数】{travelers}人"
        if travelers <= 2:
            people_info += "（情侣/密友出行，安排浪漫私密体验）"
        elif travelers <= 4:
            people_info += "（家庭/小团体出行，注意协调口味和节奏）"
        elif travelers <= 8:
            people_info += "（中型团体，需考虑分组活动和拼车拼桌）"
        else:
            people_info += "（大型团体，优先安排大巴接送、团餐预订、分批游览）"

    # 预算类型
    budget_info = f"\n【预算】{budget}"
    if budget_type == "aa":
        budget_info += f"（此为AA制每人预算，实际总预算=每人预算×{travelers}人={budget}×{travelers}，请以总预算为准规划但不在输出中显示计算过程）"

    # 节奏描述（连续值 0-100）
    if pace <= 10:
        pace_desc = "极慢节奏：每天只安排1个核心景点，上午10点后出发，大量自由时间，深度体验为主，适合度假放空"
    elif pace <= 25:
        pace_desc = "慢节奏：每天1-2个景点，上午9:30后出发，每个景点预留充足时间，下午可自由探索"
    elif pace <= 40:
        pace_desc = "偏休闲节奏：每天2个景点，上午9点出发，适当安排休息时间，游览与放松兼顾"
    elif pace <= 55:
        pace_desc = "适中节奏：每天2-3个景点，早上9点出发，合理分配时间，兼顾游览和休息"
    elif pace <= 70:
        pace_desc = "偏紧凑节奏：每天3个景点，早上8:30出发，充分利用白天时间，行程较充实"
    elif pace <= 85:
        pace_desc = "快节奏：每天3-4个景点，早上8点出发，紧密安排行程，适合打卡式旅行"
    else:
        pace_desc = "极快节奏：每天4-5个景点，早上7:30前出发，最大化游览效率，适合特种兵式旅行"
    pace_info = f"\n【游玩节奏】{pace_desc}"
    pace_info += "\n【重要】游玩节奏优先于所有其他因素！请严格按照此节奏安排每天行程，人数影响为次要考虑。"
    if pace <= 40:
        pace_info += "\n【休闲节奏约束-最高优先级】此用户选择了偏休闲/慢节奏，所有出行时间不得早于上午8:00！上午景点最早8:00开始，出发时间最早8:00（绝对不能是7:00、7:30等）。"

    # 出行人群偏好
    travel_group_info = ""
    if travel_group == "youth":
        travel_group_info = """
【出行人群-青少年-最高优先级】
用户选择了"青少年"出行人群，请按以下原则安排行程：
1. 景点选择：优先安排年轻人喜爱的热门打卡地、网红景点、主题乐园、户外探险、运动体验类项目
2. 住宿选择：推荐青旅、民宿、特色胶囊酒店等性价比高且有社交氛围的住宿
3. 餐饮推荐：以本地特色小吃、网红餐厅、夜市美食为主，注重性价比和体验感
4. 节奏建议：可安排偏紧凑的节奏，年轻人精力充沛，可适当增加景点密度
5. 交通建议：优先推荐公共交通（地铁/公交），减少打车费用，适合背包客风格
6. 预算控制：整体预算偏经济型，控制住宿和交通成本，更多预算留给体验项目
7. 避免安排：不适合老年人的高强度爬山、需要门票的高消费项目需标注性价比分析
8. 安全提示：户外探险类活动需标注安全注意事项"""
    elif travel_group == "senior":
        travel_group_info = """
【出行人群-中老年人-最高优先级】
用户选择了"中老年人"出行人群，请按以下原则安排行程：
1. 景点选择：优先安排文化历史类景点（博物馆、古迹、寺庙）、自然风光（公园、湖泊）、温泉养生类项目，避免高强度爬山和刺激性游乐项目
2. 住宿选择：推荐安静舒适、交通便利的酒店，优先选择有电梯、离景区近的住宿，避免青年旅舍和嘈杂地段
3. 餐饮推荐：以清淡健康、卫生可靠的正餐为主，推荐老字号餐厅、品质中餐，避免辛辣刺激和街边摊
4. 节奏建议：必须采用慢节奏！每天最多2个景点，上午9:30后出发，下午留足休息时间，晚上不安排活动或只安排轻松散步
5. 交通建议：优先推荐打车/包车/景区电瓶车，减少步行距离，避免长时间公共交通换乘
6. 预算控制：预算适当放宽，优先考虑舒适度和便利性，住宿和交通可占较高比例
7. 健康关怀：每个景点标注是否有无障碍设施、休息区、医疗点，行程中预留充足的休息时间
8. 避免安排：不建议安排爬山、长时间步行（单次不超过1小时）、高温时段户外活动、夜间活动
9. 保险建议：提示用户购买旅行意外险，特别是包含医疗救援的险种"""
    elif travel_group == "family":
        travel_group_info = """
【出行人群-全家出行-最高优先级】
用户选择了"全家出行"（含老人和小孩），请按以下原则安排行程：
1. 景点选择：兼顾各年龄段需求，优先选择老少皆宜的景点（主题乐园、动物园、海洋馆、科技馆、自然风光），避免纯成人向的网红打卡地和极限运动
2. 住宿选择：推荐家庭房/套房/公寓式酒店，有厨房和洗衣机更佳，靠近商圈或景区，方便就餐和购物
3. 餐饮推荐：以家庭聚餐为主，推荐有包间、有儿童座椅的餐厅，口味兼顾老人清淡和小孩喜好，避免排长队的网红店
4. 节奏建议：必须采用慢节奏！每天最多2-3个景点，上午9:00后出发，中午安排午休时间（12:00-14:00不安排景点），下午5:00前结束游览
5. 交通建议：优先推荐包车/自驾/打车，避免频繁换乘公共交通，确保有安全座椅等儿童设施
6. 预算控制：以家庭为单位计算，住宿和餐饮占比较高，门票预算考虑儿童/老人优惠票
7. 亲子设施：每个景点标注是否有儿童游乐区、母婴室、无障碍通道、家庭卫生间
8. 老人关怀：避免长时间步行（单次不超过1.5小时），预留充足休息时间，注意温差变化
9. 安全提示：标注景点安全等级，提醒看管好儿童，注意人流密集区域的安全
10. 弹性安排：行程中预留1-2个自由活动时段，方便家庭成员根据体力情况灵活调整"""

    return f"""你是一个资深旅行规划师。请根据以下真实数据，生成一份 {days} 天的{dest}行程。

【目的地】{dest}
【天数】{days}天{date_info}{people_info}{budget_info}{pace_info}{travel_group_info}{transport_section}
【兴趣】{interest_str}

【高德 POI 真实数据（按评分降序排列，评分越高越热门）】
{poi_str}

【天气预报】
{weather_str}

请严格按照以下 JSON 格式输出（不要输出其他内容）：
{{
  "destination": "{dest}",
  "days": {days},
  "start_date": "{start_date}",
  "end_date": "{end_date}",
  "departure_transport": {{
    "type": "出发交通方式（高铁/飞机/自驾/大巴）",
    "flight_number": "航班号或车次号（如CA1515、G1等，必须从提供的真实班次中选择，自驾则为空字符串）",
    "departure_time": "建议出发时间（如8:00，留足缓冲，也可以是14:00、20:00等任意时刻）",
    "station": "出发站/机场名称（必须使用真实民用机场名称，禁止使用军用/已关闭机场）",
    "arrival_time": "预计到达目的地时间（跨天到达标注"次日XX:XX"）",
    "duration": "交通耗时（必须使用'航班号/车次号+耗时'格式，如'G1次 4h29min'或'CA1515 2h10min'）",
    "cost": "预估费用",
    "cross_day": false,
    "station_to_hotel": "从机场/车站到酒店的方式和时间（如：打车30分钟约50元，或地铁X号线转X号线约45分钟）",
    "note": "注意事项（如提前多久到站、中转信息、是否有合适的航班/车次）",
    "transfers": [
      {{
        "step": 1,
        "type": "交通工具类型（飞机/高铁/大巴/出租车/地铁）",
        "flight_number": "中转段航班号/车次号（如CA1234，如无则留空）",
        "from_station": "出发站/机场",
        "to_station": "到达站/机场",
        "departure_time": "中转段出发时间",
        "arrival_time": "中转段到达时间",
        "duration": "中转段耗时",
        "transfer_time": "中转等待时间（如'1小时30分钟'），必须≥45分钟（飞机）或≥30分钟（火车）",
        "note": "中转说明"
      }}
    ],
    "vehicle_image": "交通工具配图关键词（如'airplane','high_speed_rail','bus'等）"
  }},
  "return_transport": {{
    "type": "返程交通方式",
    "flight_number": "航班号或车次号（必须从提供的真实班次中选择，自驾则为空字符串）",
    "departure_time": "建议出发时间（根据最后一天游玩安排倒推）",
    "station": "出发站/机场名称（必须使用真实民用机场名称）",
    "arrival_time": "预计到达时间（跨天到达标注"次日XX:XX"）",
    "duration": "交通耗时（必须使用'航班号/车次号+耗时'格式）",
    "cost": "预估费用",
    "cross_day": false,
    "station_to_hotel": "从酒店到机场/车站的方式和时间",
    "note": "注意事项",
    "transfers": [],
    "vehicle_image": "交通工具配图关键词"
  }},
  "weather": [
    {{"date": "日期", "weather": "天气", "temp": "温度", "wind": "风力", "icon": "天气图标emoji"}}
  ],
  "itinerary": [
    {{
      "day": 1,
      "date": "日期",
      "weather": {{"desc": "天气", "temp": "温度", "icon": "emoji"}},
      "morning": {{"spot": "景点名", "time_slot": "游玩时间段（如8:00-10:30）", "duration": "建议时长（如2.5小时）", "reason": "推荐理由", "location": "坐标", "need_booking": false, "route_detail": "", "recommended_routes": []}},
      "afternoon": {{"spot": "景点名", "time_slot": "游玩时间段（如13:00-15:30）", "duration": "建议时长（如2.5小时）", "reason": "推荐理由", "location": "坐标", "need_booking": false, "route_detail": "", "recommended_routes": []}},
      "evening": {{"spot": "景点名", "time_slot": "游玩时间段（如18:30-20:30）", "duration": "建议时长（如2小时）", "reason": "推荐理由", "location": "坐标", "need_booking": false, "route_detail": "", "recommended_routes": []}},
      "lunch": "午餐推荐",
      "dinner": "晚餐推荐",
      "transport": "交通建议"
    }}
  ],
  "budget_breakdown": {{"交通": "金额", "住宿": "金额", "餐饮": "金额", "门票": "金额", "其他": "金额"}},
  "tips": ["贴士1", "贴士2", "贴士3"]
}}

要求：
【第零条-往返交通必填-最高优先级，优先级高于所有其他规则！】
0. departure_transport和return_transport绝对不可缺失！它们是行程的骨架，缺失则整个行程无效！
   a) departure_transport：必须包含从{departure_city}到{dest}的完整交通规划，包含type、flight_number（从真实班次选）、departure_time、arrival_time、duration（必须用"班次号+耗时"格式）、station（真实民用机场/车站名）、cost、station_to_hotel、note
   b) return_transport：必须包含从{dest}返回{departure_city}的完整交通规划，字段同上
   c) 第一天itinerary必须包含出发交通的完整描述：什么时间从家出发→前往机场/车站（方式+耗时）→乘坐什么班次（具体航班号/车次号）→到达时间→如何到酒店（方式+耗时）。出发交通占用时间必须从第一天可以游玩的时间中扣除！
   d) 最后一天itinerary必须包含返程交通的完整描述：什么时间从酒店出发→前往机场/车站（方式+耗时）→乘坐什么班次→到达时间。返程交通占用时间必须从最后一天可以游玩的时间中扣除！
   e) 如果交通时间占了半天以上，第一天/最后一天只安排0-1个景点或不安排景点
   f) 交通耗时估算：高铁约300km/h，飞机飞行时间约800km/h（不含候机2小时+机场到市区1小时），大巴约80km/h，自驾约100km/h。根据用户偏好和飞常准API数据选择合适的交通方式
   g) station字段必须填写真实运营的机场/车站名称，禁止使用已关闭/军用的机场
   h) 【换乘必须全部列出】如果前往目的地需要换乘多种交通工具（如：家→打车到地铁站→地铁→高铁站→高铁→目的地机场→机场大巴→酒店），transfers数组中必须按顺序列出每一段换乘！每段必须填写：step序号、type（交通工具类型）、flight_number（飞机/火车必填班次号，其他留空）、from_station、to_station、departure_time（具体发车时间）、arrival_time、duration、transfer_time（换乘等待时间）。飞机和火车/高铁必须注明班次编号和发车时间，公交/地铁/打车/大巴等只需注明交通方式类型和耗时。
   i) 【游玩必须在到达之后-致命规则】第一天所有景点的time_slot开始时间必须晚于到达酒店的时间（arrival_time + station_to_hotel耗时）！绝对不能出现上午9:00在景点游玩，但航班下午2:00才到达的情况！如果到达时间是下午或晚上，第一天只能安排0-1个晚间景点或不安排景点。最后一天所有景点的time_slot结束时间必须早于出发去机场/车站的时间（departure_time - 前往机场/车站耗时 - 提前1-2小时候机/候车）！

1. 【热门景点优先】优先选择评分高（≥4.0）的著名景点，确保行程中至少80%的景点来自高评分榜单。POI列表已按评分降序排列，排在前面的优先选择
2. 优先使用高德POI真实数据，location坐标必须填写
3. need_booking 标记需要提前预约的景点（如故宫、莫高窟、迪士尼等热门景点设为true）
4. 结合天气预报合理安排室内外活动
5. 上午/下午各安排1个景点，晚上安排1个
6. 【最高优先级-时间规划】每个景点必须填写 time_slot 字段，给出当天在当地的具体游玩时间段（如"9:00-11:30"），必须严格遵循以下规则：
   a) 上午时段：最早8:00开始，最晚12:00结束，上午景点游玩时间建议2-3小时
   b) 下午时段：最早13:00开始，最晚17:30结束，下午景点游玩时间建议2-3.5小时
   c) 晚上时段：最早18:00开始，最晚21:30结束，晚上景点游玩时间建议1.5-2.5小时
   d) 午餐时间：12:00-13:00为午餐和休息时间，不可安排景点游玩
   e) 晚餐时间：17:30-18:30为晚餐时间，不可安排景点游玩
   f) 景点间交通损耗：相邻景点之间必须预留至少30-60分钟交通和休息时间（根据距离远近），上午景点的结束时间与下午景点的开始时间间隔至少1小时，下午景点的结束时间与晚上景点的开始时间间隔至少1小时
   g) 大景点时间：大型景区（如故宫、黄山、九寨沟、张家界等）上午游玩时间建议3-4小时，下午建议3-4小时
   h) 小景点时间：博物馆、寺庙、公园等小景点建议1.5-2.5小时
   i) 节奏适配：极慢/慢节奏需将游玩时间增加30%，并增加景点间休息时间至60-90分钟；快/极快节奏可适当缩短游玩时间但不可低于1小时，景点间间隔至少30分钟
   j) 所有时间使用24小时制，跨天使用"次日XX:XX"格式
7. 【最高优先级】景点名绝对不能重复！整个{days}天行程中，每个独立景点名只能出现一次，不允许任何例外。即使同一大型景区，不同天也必须用不同子区域命名（如"张家界(金鞭溪-袁家界)"和"张家界(天子山-杨家界)"），确保景点名不重复，避免走回头路
8. 【交通便利性】安排景点顺序时需考虑地理位置和商区分布，将同一商区/相邻区域的景点安排在同一天，减少不必要的交通时间。每天景点之间距离不宜过远
9. 对于大型景区（如张家界、黄山、九寨沟、故宫、颐和园等），每天安排不同区域/入口，按地理位置顺序游览，不走回头路：
   - route_detail 字段中描述当天的具体游览路线（如"南门进→金鞭溪→袁家界→百龙天梯→东门出"）
   - recommended_routes 字段中提供2-3条该景点的经典游览路线方案供用户参考（每条路线一句话描述，如"经典一日游：南门进→金鞭溪→袁家界→天子山→东门出（约6小时）"）
   - 普通景点 recommended_routes 设为空数组 []
10. 【出发/返程-最高优先级】departure_transport和return_transport必须认真填写。flight_number必须从提供的真实航班/车次班次中选择，不可随意编造！duration必须使用'航班号/车次号 + 耗时'格式（如'G1次 4h29min'、'CA1515 2h10min'）。出发时间不能太紧凑，必须留足缓冲时间。飞机需提前2小时到机场，火车需提前1小时到站。出发时间可以是任意时刻（上午/下午/晚上），根据实际航班/车次时刻表决定。如果下午到达，第一天可安排1个晚间景点；如果晚上到达，仅安排入住酒店。跨天到达必须标注"次日XX:XX"并设置cross_day=true。station_to_hotel字段必须填写从机场/车站到酒店的具体交通方式和时间。
11. 【中转/换乘-最高优先级】如果出发城市到目的城市没有直飞航班，或需要中途换乘，必须填写transfers数组。中转规则：
   a) 飞机中转：中转等待时间≥1.5小时（国内转国内），必须明确写出第一段航班号→中转机场→第二段航班号的变化
   b) 火车中转：中转等待时间≥30分钟，必须明确写出第一段车次→中转站→第二段车次的变化
   c) 飞机转火车/火车转飞机：中转等待时间≥2小时，考虑从机场到火车站的交通时间
   d) 每个中转段必须填写：step序号、type、flight_number、from_station、to_station、departure_time、arrival_time、duration、transfer_time
   e) 宁可少安排景点也不能赶时间！中转时间必须充裕，如有跨天中转必须标注
   f) vehicle_image字段：填写交通工具关键词（直飞填"airplane"，高铁填"high_speed_rail"，自驾填"car"，大巴填"bus"），用于前端显示交通工具配图
12. 【时间格式】所有时间必须使用24小时制（如9:00、14:30），绝对禁止出现>23:59的时间（如26:00）。如果活动跨天，请使用"次日8:00"等格式表示。每天上午/下午/晚上的景点安排间隔至少1小时，避免时间冲突
13. 只输出JSON，不要markdown代码块"""


def build_retry_prompt(original_prompt: str, validation_result: dict, transport_info: dict,
                       departure_city: str, dest: str, start_date: str, user_transport_mode: str = "") -> str:
    """构建重试提示词：当AI编造了不存在的班次号时，用严格提示词强制重生成"""
    issues = validation_result.get("issues", [])
    fabricated = validation_result.get("fabricated", [])

    # 构建真实班次列表
    real_schedule_text = ""
    vf_data = transport_info.get("variflight_data", {})
    if vf_data.get("success"):
        vf_flights = vf_data.get("flights", [])
        vf_trains = vf_data.get("trains", [])
        # 🔴 根据用户交通方式过滤
        if user_transport_mode == "飞机":
            vf_trains = []
        elif user_transport_mode == "高铁":
            vf_flights = []
        if vf_flights or vf_trains:
            real_schedule_text = "\n【🔴 飞常准API唯一真实班次 - 这是唯一可用的数据源！】"
            if vf_flights:
                real_schedule_text += "\n真实航班（飞常准API实时查询，唯一可用）："
                for f in vf_flights[:10]:
                    airport_info = ""
                    if f.get('from_airport') and f.get('to_airport'):
                        airport_info = f"  [{f.get('from_airport','')} → {f.get('to_airport','')}]"
                    price_info = f"  ¥{f.get('price','')}" if f.get('price') else ""
                    real_schedule_text += f"\n  ✈ {f['num']}：{f.get('dep','')}→{f.get('arr','')}（{f.get('duration','')}）{airport_info}{price_info}"
            if vf_trains:
                real_schedule_text += "\n真实火车/高铁（飞常准API实时查询，唯一可用）："
                for t in vf_trains[:10]:
                    station_info = ""
                    if t.get('from_station') and t.get('to_station'):
                        station_info = f"  [{t.get('from_station','')} → {t.get('to_station','')}]"
                    price_info = f"  ¥{t.get('price','')}" if t.get('price') else ""
                    real_schedule_text += f"\n  🚄 {t['num']}：{t.get('dep','')}→{t.get('arr','')}（{t.get('duration','')}）{station_info}{price_info}"

    # 从route_schedule获取飞常准API班次数据
    if not real_schedule_text:
        schedule = transport_info.get("route_schedule", {})
        if schedule.get("flights") or schedule.get("trains"):
            real_schedule_text = "\n【🔴 飞常准API实时班次 - 这是唯一可用的数据源！】"
            if schedule.get("flights"):
                real_schedule_text += "\n真实航班："
                for f in schedule["flights"]:
                    real_schedule_text += f"\n  ✈ {f['num']}：{f.get('dep','')}→{f.get('arr','')}（{f.get('duration','')}）"
            if schedule.get("trains"):
                real_schedule_text += "\n真实火车/高铁："
                for t in schedule["trains"]:
                    real_schedule_text += f"\n  🚄 {t['num']}：{t.get('dep','')}→{t.get('arr','')}（{t.get('duration','')}）"

    fabricated_text = "、".join(fabricated) if fabricated else "未知"

    # 🔴 用户指定交通方式：最高优先级，重试时也必须遵守
    user_mode_section = ""
    user_mode_reminder = ""
    if user_transport_mode and user_transport_mode in ("飞机", "高铁"):
        user_mode_section = f"""
【🔴 用户指定交通方式 - 最高优先级！不可违背！】
用户明确要求使用{user_transport_mode}出行！这是用户的选择，你无权更改！
type字段必须为"{user_transport_mode}"，绝对不允许改成其他任何交通方式！
如果用户选飞机，departure_transport.type和return_transport.type都必须是"飞机"！
如果用户选高铁，departure_transport.type和return_transport.type都必须是"高铁"！
即使飞常准API没有返回{user_transport_mode}类型的班次，type也必须填"{user_transport_mode}"，flight_number可以留空！"""
        user_mode_reminder = f'\n6. 🔴 用户要求使用{user_transport_mode}出行，type字段必须为"{user_transport_mode}"，禁止改为其他类型！'

    retry_header = f"""🔴🔴🔴 上次生成的行程被拒绝！请修正以下问题后重新生成：
{chr(10).join(['  ❌ ' + i for i in issues]) if issues else ''}

【编造的班次号（如有）】{fabricated_text}

【致命规则 - 违反将导致行程无效！】
1. 航班号/车次号必须且只能从飞常准API提供的真实班次中选择，绝对禁止编造！
2. 如果飞常准API没有返回任何数据，flight_number必须留空字符串""，绝对不能自己编一个！
3. station必须是飞常准API返回的机场/车站名，API返回的即为有效名，不需要参考任何其他名单！
4. departure_time和arrival_time必须与所选真实班次的dep/arr时间完全一致！
5. duration必须使用"班次号 + 耗时"格式，不可只写耗时！
{user_mode_section}

{real_schedule_text}

【重新生成要求】
1. 出发交通(departure_transport)和返程交通(return_transport)的flight_number必须从上表中选择
2. 如果上表为空（无任何真实班次），则flight_number留空""，只填交通方式类型和估算耗时
3. 出发/到达时间必须与上表所选班次完全一致，一字不差
4. 出发日期为{start_date}，确保所选班次在该日期有效
5. 行程时间安排必须与交通班次时间协调一致
{user_mode_reminder}

请严格按照原始JSON格式重新生成完整行程。记住：宁可flight_number留空，也绝不能编造！"""

    # 将重试头插入到原始prompt的"要求"部分之前
    if "要求：" in original_prompt:
        parts = original_prompt.split("要求：", 1)
        return parts[0] + retry_header + "\n要求：" + parts[1]
    return original_prompt + "\n" + retry_header


def build_booking_prompt(dest: str, start_date: str, end_date: str, budget: str,
                         itinerary: list, departure_city: str = "",
                         transport_info: dict = None,
                         departure_transport: dict = None,
                         return_transport: dict = None,
                         variflight_data: dict = None) -> str:
    """构建订票/酒店/门票查询提示词，包含出发城市、交通判断和飞常准数据"""
    spots_str = ""
    for day in itinerary:
        for slot in ["morning", "afternoon", "evening"]:
            s = day.get(slot, {})
            if s and s.get("spot"):
                spots_str += f"- Day{day['day']} {slot}: {s['spot']}（坐标：{s.get('location','')}，需预约：{'是' if s.get('need_booking') else '否'}）\n"

    dep_info = f"\n【出发城市】{departure_city}" if departure_city else ""
    trans_info = ""
    flight_info = ""
    if transport_info:
        trans_info = f"""
【交通判断】{transport_info.get('mode', '')} - {transport_info.get('reason', '')}
【城市代码】出发：{transport_info.get('dep_iata', '')}，到达：{transport_info.get('arr_iata', '')}"""
        if transport_info.get("need_flight"):
            flight_info = f"""
飞常准航班查询参数：
- 出发城市：{departure_city}（{transport_info.get('dep_iata', '')}）
- 到达城市：{dest}（{transport_info.get('arr_iata', '')}）
- 出发日期：{start_date}
- 返程日期：{end_date}
航班link：https://flights.ctrip.com/booking/{departure_city}-{dest}-day-1.html
火车票link：https://m.ctrip.com/html5/trains/"""
    else:
        flight_info = f"""
机票link：https://flights.ctrip.com/booking/{departure_city}-{dest}-day-1.html
火车票link：https://m.ctrip.com/html5/trains/"""

    # 将行程中已选班次强制传递给预订提示词
    transport_schedule_section = ""
    if departure_transport and departure_transport.get("flight_number"):
        verified_badge = "✅飞常准验证" if departure_transport.get("_verified") else ""
        transport_schedule_section += f"""
【去程已选班次-必须严格同步！{verified_badge}】
  航班号/车次号：{departure_transport.get('flight_number', '')}
  出发时间：{departure_transport.get('departure_time', '')}
  出发站/机场：{departure_transport.get('station', '')}
  到达时间：{departure_transport.get('arrival_time', '')}
  耗时：{departure_transport.get('duration', '')}
  类型：{departure_transport.get('type', '')}
  验证来源：{departure_transport.get('_verified_source', '未验证')}"""
    if return_transport and return_transport.get("flight_number"):
        verified_badge = "✅飞常准验证" if return_transport.get("_verified") else ""
        transport_schedule_section += f"""
【返程已选班次-必须严格同步！{verified_badge}】
  航班号/车次号：{return_transport.get('flight_number', '')}
  出发时间：{return_transport.get('departure_time', '')}
  出发站/机场：{return_transport.get('station', '')}
  到达时间：{return_transport.get('arrival_time', '')}
  耗时：{return_transport.get('duration', '')}
  类型：{return_transport.get('type', '')}
  验证来源：{return_transport.get('_verified_source', '未验证')}"""
    if transport_schedule_section:
        transport_schedule_section += """
【预订卡片班次同步-最高优先级】flights 数组中的 suggest 字段必须包含与上面完全一致的航班号/车次号！
  ① 去程suggest必须包含上述航班号/车次号，一字不差
  ② 返程suggest必须包含上述航班号/车次号，一字不差
  ③ 出发时间、到达时间、机场/车站名也必须与上面完全一致
  ④ link 使用正确链接：机票用 https://flights.ctrip.com/booking/{departure_city}-{dest}-day-1.html，火车票用 https://www.fliggy.com/train/
  ⑤ 绝对禁止在预订卡片中编造不同的航班号/车次号！"""

    # 飞常准API实时数据（用于预订卡片验证）
    vf_section = ""
    if variflight_data and variflight_data.get("success"):
        vf_flights = variflight_data.get("flights", [])
        vf_trains = variflight_data.get("trains", [])
        if vf_flights or vf_trains:
            vf_section = "\n【飞常准API实时数据-预订卡片必须使用以下真实班次】"
            if vf_flights:
                vf_section += "\n真实航班："
                for f in vf_flights[:5]:
                    vf_section += f"\n  ✈ {f['num']} {f.get('dep','')}→{f.get('arr','')} {f.get('from_airport','')}→{f.get('to_airport','')} ¥{f.get('price','')}"
            if vf_trains:
                vf_section += "\n真实火车票："
                for t in vf_trains[:5]:
                    vf_section += f"\n  🚄 {t['num']} {t.get('dep','')}→{t.get('arr','')} {t.get('from_station','')}→{t.get('to_station','')} ¥{t.get('price','')}"
            vf_section += "\n【飞常准数据规则】预订卡片中的suggest必须与上述真实班次完全一致，价格使用上述真实价格！"

    return f"""你是一个旅行服务顾问。请为以下行程查询机票、火车票、酒店和门票推荐信息。

【目的地】{dest}{dep_info}
【日期】{start_date} 至 {end_date}
【预算】{budget}
{trans_info}
{flight_info}
{transport_schedule_section}
{vf_section}

【行程景点】
{spots_str}

请输出以下 JSON 格式（不要输出其他内容）：
{{
  "flights": [
    {{"type": "去程/返程", "suggest": "推荐航班/车次（必须与行程中已选班次完全一致！）", "price": "参考价格", "link": "航班/火车票链接", "note": "说明"}}
  ],
  "hotels": [
    {{"name": "酒店名", "area": "推荐区域", "price": "参考价格/晚", "reason": "推荐理由", "location": "坐标", "link": "https://hotels.ctrip.com/", "note": "说明", "stay_days": "Day1-Day3"}}
  ],
  "hotel_changes": [
    {{"from_hotel": "原酒店", "to_hotel": "新酒店", "change_day": "第几天换", "reason": "换酒店原因（景点集中在不同区域等）", "new_area": "新区域", "price": "参考价格/晚", "location": "坐标", "link": "https://hotels.ctrip.com/"}}
  ],
  "tickets": [
    {{"spot": "景点名", "price": "门票价格", "need_booking": true, "booking_days": "提前N天", "platform": "预约平台", "link": "官方预约网址（必须是官网/公众号/小程序链接，绝对不能是携程链接！）", "note": "预约说明", "location": "坐标"}}
  ],
  "transport_mode": "推荐交通工具（高铁/动车/飞机/自驾等）",
  "booking_tips": ["预约贴士1", "预约贴士2"]
}}

要求：
1. 根据用户偏好和飞常准API数据选择合适的交通工具，有飞机/高铁时优先推荐
2. flights中机票link使用 https://flights.ctrip.com/booking/{departure_city}-{dest}-day-1.html，火车票link使用 https://www.fliggy.com/train/
3. 机票/火车票给出真实航线建议和参考价格
4. transport_mode字段：说明推荐的交通工具及理由
5. 酒店推荐位置方便、性价比高的，给出具体名称和位置坐标
6. 门票中 need_booking=true 的景点必须说明提前几天预约
7. 【景区预约链接-最高优先级】只有 need_booking=true 的景点才需要填官方预约网址（公众号链接、小程序链接或官网链接），link必须是完整可访问的URL（以http://或https://开头），绝对不能填携程(ctrip.com)链接！如故宫填"https://gugong.ktmtech.cn/"，莫高窟填"https://www.mgk.org.cn/"，迪士尼填"https://www.shanghaidisneyresort.com/"等。如不确定官方网址，填平台名称并在note中说明，link留空字符串。need_booking=false的景点link填携程链接"https://piao.ctrip.com/"即可。
8. 【换酒店推荐】分析行程中景点的地理分布，如果不同天的景点集中在不同区域（如Day1-3在城东，Day4-5在城西），建议中途换酒店减少通勤时间，在 hotel_changes 中列出换酒店建议
9. hotels 中 stay_days 字段注明该酒店适合入住的日期范围
10. 只输出JSON，不要markdown代码块"""


def build_regenerate_prompt(dest: str, days: int, user_input: str, old_itinerary: list,
                           weather_data: list, start_date: str = "", end_date: str = "",
                           is_self_drive: bool = False, departure_city: str = "",
                           transport_info: dict = None, transport_mode: str = "",
                           mixed_transport: dict = None, travel_group: str = "") -> str:
    """构建重新生成计划的提示词，重点参考用户输入的新需求"""
    transport_info = transport_info or {}  # 防止为None时报错
    old_summary = ""
    for day in old_itinerary:
        spots = []
        for slot in ["morning", "afternoon", "evening"]:
            s = day.get(slot, {})
            if s and s.get("spot"):
                spots.append(f"{slot}: {s['spot']}")
        old_summary += f"Day{day['day']}({day.get('date','')}): {', '.join(spots)}\n"

    weather_str = "\n".join([
        f"  {w['date']}: {w['dayweather']} {w['daytemp']}°C~{w['nighttemp']}°C"
        for w in (weather_data or [])[:days+2]
    ]) if weather_data else "（请根据常识判断）"

    # 🔴 用户指定出行方式：最高优先级
    transport_mode_map = {"plane": "飞机", "train": "高铁", "taxi": "打车", "selfdrive": "自驾"}
    user_transport_mode = transport_mode_map.get(transport_mode, "")
    if user_transport_mode:
        transport_mode_display = user_transport_mode
        is_self_drive = (user_transport_mode == "自驾")
    else:
        transport_mode_display = "自驾" if is_self_drive else "公共交通"

    # 交通判断信息
    transport_section = ""
    if departure_city:
        # 🔴 混合交通方式检测：出发和返程使用不同交通工具
        if mixed_transport and mixed_transport.get("has_mixed"):
            dep_cn = mixed_transport.get("departure_cn", "")
            ret_cn = mixed_transport.get("return_cn", "")
            transport_section = f"\n【🔴🔴 混合交通方式 - 最高优先级，绝对不可更改！】"
            transport_section += f"\n用户明确要求出发和返程使用不同交通工具！"
            transport_section += f"\n  - departure_transport.type 必须为'{dep_cn}'（出发交通）"
            transport_section += f"\n  - return_transport.type 必须为'{ret_cn}'（返程交通）"
            transport_section += f"\n  - 绝对禁止往返都使用同一种交通方式！出发和返程的type必须不同！"
            transport_section += f"\n  - 如果飞常准API有数据，出发从{transport_info.get('variflight_data',{}).get('flights',[]) if dep_cn=='飞机' else ''}中选择{dep_cn}班次"
            transport_section += f"\n  - 返程从{ret_cn}班次中选择，绝对禁止混淆"
            transport_section += f"\n  - 即使飞常准API无数据，type也必须分别填'{dep_cn}'和'{ret_cn}'，flight_number可留空"
        elif user_transport_mode:
            # 用户指定了具体出行方式：强制使用该方式
            transport_section = f"\n【🔴 用户指定出行方式 - 最高优先级，绝对不可更改！】{user_transport_mode}"
            if user_transport_mode == "自驾":
                transport_section += f"\n【出行方式】自驾从{departure_city}出发，需计算驾驶时间，长途需安排过夜停留"
            elif user_transport_mode == "飞机":
                transport_section += f"\n【🔴 不可更改】用户指定飞机出行，departure_transport和return_transport的type字段必须为'飞机'，绝对禁止填写'高铁'或任何其他交通方式！"
                transport_section += f"\n必须安排{departure_city}到{dest}的往返航班，即使飞常准API无航班数据，type也必须填'飞机'"
            elif user_transport_mode == "高铁":
                transport_section += f"\n【🔴 不可更改】用户指定高铁出行，departure_transport和return_transport的type字段必须为'高铁'，绝对禁止填写'飞机'或任何其他交通方式！"
                transport_section += f"\n必须安排{departure_city}到{dest}的往返高铁/动车，即使飞常准API无高铁数据，type也必须填'高铁'"
            elif user_transport_mode == "打车":
                transport_section += f"\n【出行方式】用户指定打车出行，{departure_city}到{dest}使用打车/出租车"
        elif is_self_drive:
            transport_section = f"\n【出行方式】自驾从{departure_city}出发，需计算驾驶时间，长途需安排过夜停留"
        else:
            transport_section = f"\n【出发城市】{departure_city}，公共交通出行"
            if transport_info:
                ti = transport_info.get("transport", {})
                transport_section += f"\n【交通建议】{ti.get('mode','')} - {ti.get('reason','')}"
                dep_hub = transport_info.get("dep_hub", {})
                dest_hub = transport_info.get("dest_hub", {})
                if dep_hub.get("has_hub"):
                    transport_section += f"\n【出发地枢纽】{dep_hub['note']}"
                if dest_hub.get("has_hub"):
                    transport_section += f"\n【目的地枢纽】{dest_hub['note']}"
    transport_section += f"""
【交通选择原则】根据用户偏好和飞常准API数据选择合适的交通工具："""
    if mixed_transport and mixed_transport.get("has_mixed"):
        dep_cn = mixed_transport.get("departure_cn", "")
        ret_cn = mixed_transport.get("return_cn", "")
        transport_section += f"\n【🔴 混合交通方式已在上面声明，出发={dep_cn}、返程={ret_cn}，绝对不可混淆！】"
    elif user_transport_mode:
        transport_section += f"\n【用户已指定出行方式为{user_transport_mode}，必须严格遵循！】"
    transport_section += f"""
第一天必须包含从{departure_city}出发前往{dest}的交通规划，最后一天必须包含从{dest}返回{departure_city}的交通规划。
【出发时间灵活】出发时间不固定，根据航班/车次时刻表决定，可以是上午、下午、傍晚甚至晚上。下午到达可安排晚间景点，晚上到达仅入住酒店。跨天到达需标注"次日XX:XX"并规划好到达后交通。需预留充足缓冲时间（飞机提前2小时到机场，火车提前1小时到站）。
【大巴/自驾出行】如果选择大巴或自驾，flight_number留空，只填type和预估耗时，station留空或填大巴站名。"""
    if transport_info:
        # 真实班次数据
        schedule = transport_info.get("route_schedule", {})
        if schedule.get("flights") or schedule.get("trains"):
            transport_section += "\n【以下是真实可选的航班/火车班次，必须从中选择，不可随意编造！】"
            if schedule.get("flights"):
                transport_section += "\n可选航班："
                for f in schedule["flights"]:
                    transport_section += f"\n  ✈ {f['num']}：{f['dep']}出发 → {f['arr']}到达（{f['duration']}）"
            if schedule.get("trains"):
                transport_section += "\n可选火车/高铁："
                for t in schedule["trains"]:
                    transport_section += f"\n  🚄 {t['num']}：{t['dep']}出发 → {t['arr']}到达（{t['duration']}）"
            transport_section += "\n【强制要求-班次严格匹配】必须从以上班次中选择，但如果下面有飞常准API实时数据，必须以飞常准API为准！严格使用该班次全部信息：flight_number=班次号、departure_time=出发时间、arrival_time=到达时间、duration='班次号+耗时'格式，4个字段必须来自同一班次！飞常准API返回的机场/车站名即为有效名，无需参考其他名单！"
            # 飞常准API无数据时的致命警告
            if schedule.get("_no_data"):
                if mixed_transport and mixed_transport.get("has_mixed"):
                    dep_cn = mixed_transport.get("departure_cn", "")
                    ret_cn = mixed_transport.get("return_cn", "")
                    transport_section += f"\n【致命警告-飞常准API无数据】该路线飞常准API未返回实时班次！flight_number必须留空，departure_transport.type必须填'{dep_cn}'，return_transport.type必须填'{ret_cn}'，绝对禁止往返使用相同交通方式！禁止编造航班号！"
                elif user_transport_mode:
                    transport_section += f"\n【致命警告-飞常准API无数据】该路线飞常准API未返回实时班次！flight_number必须留空，type必须填'{user_transport_mode}'（用户已指定），绝对禁止改为其他交通方式，禁止编造航班号！"
                else:
                    transport_section += "\n【致命警告-飞常准API无数据】该路线飞常准API未返回实时班次！flight_number留空，只填交通方式类型，禁止编造航班号！"
            # 中转方案
            transfer_info = transport_info.get("transfer_info", {})
            if transfer_info.get("transfer_options"):
                transport_section += "\n【中转方案参考】如果无合适直飞，可考虑以下中转："
                for to in transfer_info["transfer_options"]:
                    transport_section += f"\n  经{to['transfer_city']}中转：{to['note']}"
            # 飞常准API实时数据（唯一数据源，优先级最高，覆盖静态数据）
            vf_data = transport_info.get("variflight_data", {})
            if vf_data.get("success"):
                vf_flights = vf_data.get("flights", [])
                vf_trains = vf_data.get("trains", [])
                # 🔴 根据用户交通方式过滤（混合交通时保留全部）
                transport_mode_map = {"plane": "飞机", "train": "高铁", "taxi": "打车", "selfdrive": "自驾"}
                user_transport = transport_mode_map.get(transport_mode, "")
                if not (mixed_transport and mixed_transport.get("has_mixed")):
                    if user_transport == "飞机":
                        vf_trains = []
                    elif user_transport == "高铁":
                        vf_flights = []
                if vf_flights or vf_trains:
                    transport_section += "\n\n【🔴 飞常准API实时数据 - 唯一权威数据源 - 必须100%严格使用！】"
                    transport_section += "\n以下班次为飞常准API实时查询结果，静态数据全部作废，只使用以下数据！"
                    if vf_flights:
                        transport_section += "\n🚀 实时航班（飞常准API验证，真实可购）："
                        for f in vf_flights[:10]:
                            airport_info = ""
                            if f.get('from_airport') and f.get('to_airport'):
                                airport_info = f"  [{f.get('from_airport','')} → {f.get('to_airport','')}]"
                            transport_section += f"\n  ✈ {f['num']}：{f.get('dep','')}→{f.get('arr','')}（{f.get('duration','')}）{airport_info}"
                    if vf_trains:
                        transport_section += "\n🚄 实时火车票（飞常准API验证，真实可购）："
                        for t in vf_trains[:10]:
                            station_info = ""
                            if t.get('from_station') and t.get('to_station'):
                                station_info = f"  [{t.get('from_station','')} → {t.get('to_station','')}]"
                            transport_section += f"\n  🚄 {t['num']}：{t.get('dep','')}→{t.get('arr','')}（{t.get('duration','')}）{station_info}"
                    transport_section += "\n\n【🔴 飞常准API强制规则】flight_number/departure_time/arrival_time/duration必须来自同一班次，一字不差，禁止编造！"

    # 出行人群偏好（重新生成时也需遵循）
    regen_travel_group_info = ""
    if travel_group == "youth":
        regen_travel_group_info = "\n【出行人群-青少年】优先安排年轻人喜爱的热门打卡地、网红景点、主题乐园、户外探险类项目。住宿推荐青旅/民宿，餐饮以本地小吃和网红餐厅为主，节奏可偏紧凑，优先公共交通，预算偏经济型。"
    elif travel_group == "senior":
        regen_travel_group_info = "\n【出行人群-中老年人】优先安排文化历史类景点、自然风光、温泉养生类项目，避免高强度爬山和刺激性项目。住宿推荐安静舒适酒店，餐饮以清淡健康为主，必须采用慢节奏（每天最多2个景点，9:30后出发），优先打车/包车，预算适当放宽。"
    elif travel_group == "family":
        regen_travel_group_info = "\n【出行人群-全家出行】兼顾各年龄段需求，优先老少皆宜景点（主题乐园、动物园、海洋馆等），住宿推荐家庭房/套房，餐饮以家庭聚餐为主，必须采用慢节奏（每天最多2-3个景点，9:00后出发，12:00-14:00午休），优先包车/自驾，关注儿童和老人设施。"

    return f"""你是一个资深旅行规划师。用户查看已有行程后提出了新的需求，请根据新需求重新制定计划。

【🔴 用户的新需求 - 最高优先级！必须100%满足！】
"{user_input}"

请仔细分析用户需求中的关键信息：
- 如果用户要求更换交通方式（如改为大巴、改为飞机、改为高铁），必须严格执行，不可使用原计划中的交通方式
- 如果用户要求更换班次，必须从下方可选班次中重新选择，不可沿用原班次
- 如果用户要求调整时间，必须按新时间重新安排全天行程
- 如果用户要求增删景点，必须执行
- 用户需求是整个计划的核心，不可偏离！

【目的地】{dest}
【天数】{days}天
【出行日期】{start_date} 至 {end_date}
【出行方式】{transport_mode_display}{regen_travel_group_info}{transport_section}

【原行程概览（仅供参考，新需求优先）】
{old_summary}

【天气预报】
{weather_str}

请严格按照以下 JSON 格式输出（不要输出其他内容）：
{{
  "destination": "{dest}",
  "days": {days},
  "start_date": "{start_date}",
  "end_date": "{end_date}",
  "departure_transport": {{"type": "", "flight_number": "", "departure_time": "", "station": "", "arrival_time": "", "duration": "", "cost": "", "note": "", "transfers": [], "vehicle_image": ""}},
  "return_transport": {{"type": "", "flight_number": "", "departure_time": "", "station": "", "arrival_time": "", "duration": "", "cost": "", "note": "", "transfers": [], "vehicle_image": ""}},
  "itinerary": [
    {{
      "day": 1, "date": "日期",
      "weather": {{"desc": "天气", "temp": "温度", "icon": "emoji"}},
      "morning": {{"spot": "景点名", "time_slot": "游玩时间段（如8:00-10:30）", "duration": "建议时长（如2.5小时）", "reason": "推荐理由", "location": "坐标", "need_booking": false, "route_detail": "", "recommended_routes": []}},
      "afternoon": {{"spot": "景点名", "time_slot": "游玩时间段（如13:00-15:30）", "duration": "建议时长（如2.5小时）", "reason": "推荐理由", "location": "坐标", "need_booking": false, "route_detail": "", "recommended_routes": []}},
      "evening": {{"spot": "景点名", "time_slot": "游玩时间段（如18:30-20:30）", "duration": "建议时长（如2小时）", "reason": "推荐理由", "location": "坐标", "need_booking": false, "route_detail": "", "recommended_routes": []}},
      "lunch": "午餐推荐", "dinner": "晚餐推荐", "transport": "交通建议"
    }}
  ],
  "budget_breakdown": {{"交通": "金额", "住宿": "金额", "餐饮": "金额", "门票": "金额", "其他": "金额"}},
  "tips": ["贴士1", "贴士2", "贴士3"]
}}

要求：
1. 【最高优先级-用户需求】必须严格遵循用户的新需求！如果用户要求改用大巴/自驾，则绝对不使用飞机/高铁；如果用户要求换班次，必须从可选班次中改选；如果用户要求增删景点，必须执行。整个计划的核心就是满足用户需求！
2. 如果用户要求的具体交通方式与可选班次表冲突，优先满足用户需求（如用户要求大巴，即使有航班可选也改用大巴，flight_number留空）
3. 结合天气预报合理安排室内外活动，雨天优先安排室内景点
4. 景点名绝对不能重复，同一商区/相邻区域景点安排在同一天，减少交通时间
5. 【时间安排-最高优先级】每个景点必须填写 time_slot 字段（如"9:00-11:30"），严格遵循：上午8:00-12:00，下午13:00-17:30，晚上18:00-21:30；午餐12:00-13:00和晚餐17:30-18:30不可安排景点；景点间预留至少30-60分钟交通损耗；大景点3-4小时，小景点1.5-2.5小时；节奏慢则游玩时间+30%，节奏快可缩短但≥1小时。上午结束与下午开始间隔≥1小时，下午结束与晚上开始间隔≥1小时
6. 所有时间使用24小时制，禁止>23:59的时间（如26:00），跨天活动使用"次日XX:XX"格式
7. 公共交通出行时，根据实际情况和用户偏好选择合适的交通工具
8. 出发/返程时间不能太紧凑，必须留足缓冲。飞机需提前2小时到机场，火车需提前1小时到站
9. 只输出JSON，不要markdown代码块"""