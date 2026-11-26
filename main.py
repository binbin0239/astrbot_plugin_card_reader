"""把 QQ 分享卡片（OneBot 的 json 段）转成 AI 能读懂的文本，抓取链接正文，并保留卡片配图。

处理链：
    收到消息 → 遇到 Json 段
      ├─ 按 view/meta 优先级提取标题、摘要、链接（避免抓到不相干的同名字段）
      ├─ 有链接 → 调 DeepSeek fetch 拿正文（缓存 1 小时）
      ├─ 有配图 → 追加一个 Image 组件，交给下游多模态模型
      └─ 替换成 Plain 文本 → 后续插件和 AI 看到的就是正常内容
"""

from __future__ import annotations

import asyncio
import html as _html
import json
import re
import time
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Json, Plain, Reply
from astrbot.api.star import Context, Star, register

# ============ 可调参数 ============
FETCH_ENABLED = True           # 是否自动抓取卡片链接正文
FETCH_TIMEOUT = 10             # 抓取超时（秒）;B 站首刷偶尔超过 8 秒，别调太小
FETCH_RETRY = 1                # 超时后重试几次（B 站首次渲染慢，第二次基本命中缓存）
FETCH_MAX_CHARS = 0            # 正文最多拼多少字；0 = 不截断（不限长度）
FETCH_GROUPS: set[str] = set()  # 空集合 = 所有群都抓；要限制就填群号
CACHE_TTL = 3600               # 同一链接缓存多久（秒）
CACHE_MAX = 50                 # 缓存最多保留多少条，避免长跑吃内存

# 这些站点的正文抓不到（登录墙 / 纯前端渲染），直接跳过省时间
# 注意：B 站不走这里也不走 DeepSeek，它在 _convert 里单独走公开 API（见 fetch_bili_meta）。
# 抖音/公众号群里不用，暂不做。
SKIP_FETCH_HOSTS = (
    "douyin.com", "iesdouyin.com",
    "weibo.com", "weibo.cn", "t.cn",
    "zhihu.com", "zhimg.com",
)

# B 站卡片统一走公开 API（比 DeepSeek 渲染一整页更快、更准、无界面噪音）
BILI_HOSTS = ("bilibili.com", "b23.tv")
BILI_API = "https://api.bilibili.com/x/web-interface/view?bvid="
BILI_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
           "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


# 小红书：裸链接一定 404，必须用卡片里带 xsec_token 的完整 URL + 手机 UA 抓 SSR 页面
XHS_HOSTS = ("xiaohongshu.com", "xhslink.com")
XHS_UA = ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
          "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
          "Mobile/15E148 Safari/604.1")
XHS_TIMEOUT = 10

ATTACH_IMAGE = True            # 是否把卡片配图交给多模态模型
MAX_IMAGES = 9                 # 每张卡片最多带几张图（小红书多图笔记按顺序取前 9 张）

DUMP_RAW = True                # 调试用：打印卡片原始 JSON（确认字段结构后可关掉）
# ==================================

FIELD_ORDER = ("title", "prompt", "desc", "tag", "singer", "nickname")
WANTED_KEYS = set(FIELD_ORDER) | {
    "appname", "app", "summary", "qqdocurl", "docurl", "jumpUrl", "url",
}
URL_KEYS = ("jumpUrl", "url", "qqdocurl", "docurl")
IMAGE_KEYS = ("preview", "picUrl", "thumbUrl", "cover", "imageUrl", "pic_url")
MAX_FIELD_LEN = 0              # 单个字段（标题/摘要等）最多保留多少字；0 = 不截断

# 缓存：资源 id（小红书 note id / B 站 BV 号，取不到才用完整 URL）-> (时间, 正文, 配图)
_CACHE: dict[str, tuple[float, str, list[str]]] = {}


def _clip(text: str, limit: int) -> str:
    """按上限截断；limit <= 0 表示不截断。

    正文长度现在默认不限制（FETCH_MAX_CHARS = 0），需要时可以改回具体字数。
    """
    return text if limit is None or limit <= 0 else text[:limit]


def _load(data: Any) -> Any:
    """卡片数据可能是 dict，也可能是被转义过的 JSON 字符串。"""
    if isinstance(data, str):
        for _ in range(3):
            text = data.strip()
            if not text or text[0] not in "{[":
                break
            try:
                data = json.loads(text)
            except Exception:
                break
            if not isinstance(data, str):
                break
    return data


def _walk(obj: Any, sink: dict[str, list[str]], depth: int = 0) -> None:
    """收集所有候选值（同一字段可能出现在多个层级）。"""
    if depth > 10:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                if k in (WANTED_KEYS | set(IMAGE_KEYS)):
                    val = v.strip()
                    if val:
                        sink.setdefault(k, []).append(val)
            elif isinstance(v, (dict, list)):
                _walk(v, sink, depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            _walk(item, sink, depth + 1)


def _candidates(data: Any) -> dict[str, list[str]]:
    data = _load(data)
    sink: dict[str, list[str]] = {}
    if isinstance(data, (dict, list)):
        _walk(data, sink)
    return sink


def _pick(data: Any, sink: dict[str, list[str]], names: tuple[str, ...]) -> str:
    """按「当前 view 的 meta → 任意 meta → 顶层 → 其它位置」的优先级取值。

    QQ 卡片里 title 这类通用字段可能出现多次，小红书/公众号卡片尤其容易
    在非正文位置也带一个 title，先遇到先用就会抓错。
    """
    data = _load(data)
    meta = data.get("meta") if isinstance(data, dict) else None
    view = data.get("view") if isinstance(data, dict) else None

    # 1) meta[view][name]
    if isinstance(meta, dict) and isinstance(view, str) and isinstance(meta.get(view), dict):
        for name in names:
            v = meta[view].get(name)
            if isinstance(v, str) and v.strip():
                return _clip(v.strip(), MAX_FIELD_LEN)

    # 2) 任意 meta 子对象
    if isinstance(meta, dict):
        for sub in meta.values():
            if isinstance(sub, dict):
                for name in names:
                    v = sub.get(name)
                    if isinstance(v, str) and v.strip():
                        return _clip(v.strip(), MAX_FIELD_LEN)

    # 3) 顶层
    if isinstance(data, dict):
        for name in names:
            v = data.get(name)
            if isinstance(v, str) and v.strip():
                return _clip(v.strip(), MAX_FIELD_LEN)

    # 4) 兜底：遍历到的第一个
    for name in names:
        for v in sink.get(name, []):
            if v.strip():
                return _clip(v.strip(), MAX_FIELD_LEN)
    return ""


def _clean_url(v: str) -> str:
    """卡片 JSON 里的 URL 可能被 HTML 转义（&amp;）或 JSON 转义，先还原。"""
    v = _html.unescape(v.strip())
    v = v.replace("\\u0026", "&").replace("\\/", "/").replace("\\u003D", "=")
    return v


def _looks_like_image(url: str) -> bool:
    """判断一个 URL 是不是图片，用来避免把卡片封面当成网页去抓。"""
    low = (url or "").lower().split("?")[0].split("#")[0]
    if low.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".ico", ".avif")):
        return True
    return any(k in low for k in ("/icon/", "avatar", "/logo"))


def extract_url(data: Any, sink: dict[str, list[str]] | None = None) -> str:
    """提取卡片里的完整链接。

    注意：不能走 _pick —— 那个函数会按 MAX_FIELD_LEN 截断，
    而小红书链接的 xsec_token 恰好在 160 字之后，一截就废。
    """
    sink = sink if sink is not None else _candidates(data)
    for name in URL_KEYS:
        for raw in sink.get(name, []):
            v = _clean_url(raw)
            if (v.startswith("http://") or v.startswith("https://")) and not _looks_like_image(v):
                return v
    # 兜底：任何以 http 开头的候选值，但要跳过图片字段和图片地址
    # （否则只有封面、没有链接的卡片会把封面当网页去抓，白等十几秒）
    for key, values in sink.items():
        if key in IMAGE_KEYS:
            continue
        for raw in values:
            v = _clean_url(raw)
            if (v.startswith("http://") or v.startswith("https://")) and not _looks_like_image(v):
                return v
    return ""


def extract_image(data: Any, sink: dict[str, list[str]] | None = None) -> str:
    sink = sink if sink is not None else _candidates(data)
    for name in IMAGE_KEYS:
        for v in sink.get(name, []):
            v = v.strip()
            if v.startswith("http://") or v.startswith("https://"):
                # 排除明显是应用图标的
                low = v.lower()
                if any(x in low for x in ("icon", "avatar", "logo")) and "preview" not in low:
                    continue
                return v
    return ""


def card_to_text(data: Any, include_desc: bool = True) -> str:
    sink = _candidates(data)
    if not sink:
        return ""

    title = _pick(data, sink, ("title",))
    # 小红书这类卡片顶层还有 prompt，往往是「[分享]真实标题」
    prompt = _pick(data, sink, ("prompt",))
    desc = _pick(data, sink, ("desc", "summary"))
    tag = _pick(data, sink, ("tag",))
    singer = _pick(data, sink, ("singer", "nickname"))
    url = extract_url(data, sink)

    if not (title or prompt or desc or url):
        return ""

    parts = ["[分享卡片]"]
    if prompt and prompt not in (title or "") and len(prompt) > len(title or ""):
        title = prompt  # prompt 更完整时优先
    if title:
        parts.append("标题：" + title)
    if singer and singer not in title:
        parts.append("来源：" + singer)
    # include_desc=False：已经有完整正文时不再带卡片摘要（QQ 那份是截断版）
    if include_desc and desc and desc not in title:
        parts.append("摘要：" + desc)
    if tag and tag not in (title or "") and tag not in (desc or ""):
        parts.append("标签：" + tag)

    # 链接不参与截断：链接最长、排在最后，一截就断在半路，模型看到的 URL 直接不可用
    text = " ｜ ".join(parts)
    if url:
        text += " ｜ 链接：" + url
    return text


def _skip_host(url: str) -> bool:
    """命中跳过名单的站点不抓正文（抓了也是空壳，白等 8 秒）。"""
    try:
        from urllib.parse import urlparse

        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return any(host == h or host.endswith("." + h) for h in SKIP_FETCH_HOSTS)


def _is_bili(url: str) -> bool:
    try:
        from urllib.parse import urlparse

        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return any(host == h or host.endswith("." + h) for h in BILI_HOSTS)


BODY_MARKERS = ("[网页正文]", "[小红书正文]")


def _norm_line(line: str) -> str:
    """把一行正文归一化，用来判断它是不是和卡片摘要重复。

    去掉分区标记、markdown 标题符号、字段前缀、B 站标题后缀和所有空白，
    这样「[网页正文] 标题_哔哩哔哩_bilibili」和卡片里的「标题」就能对上。
    """
    s = line.strip()
    for mk in BODY_MARKERS:
        if s.startswith(mk):
            s = s[len(mk):].strip()
            break
    if not s:
        return ""
    s = re.sub(r"^#+\s*", "", s)
    s = re.sub(r"^(标题|摘要|标签|来源|链接)[：:]", "", s)
    s = re.sub(r"_(哔哩哔哩_bilibili|游戏热门视频|哔哩哔哩)$", "", s)
    s = re.sub(r"\s+", "", s)
    return s.strip("。，、,.!！?？:：;；|｜")


def _dedupe_body(body: str, seeds: list[str]) -> str:
    """去掉正文里和卡片摘要重复的行，以及正文内部的重复行。

    B 站卡片尤其明显：标题在卡片、网页标题、markdown 大标题里各出现一次，
    简介和卡片「摘要」一字不差。只判断方向：正文行是摘要的**子集**才丢，
    反过来（正文比摘要长、是全文）必须保留 —— 否则会把刚拿到的全文又丢掉。
    标题被丢掉时保留「[网页正文]」这个分区标记。
    """
    seed_norms = {n for n in (_norm_line(x) for x in seeds if x) if len(n) >= 6}
    seen: set[str] = set()
    out: list[str] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue

        marker = ""
        for mk in BODY_MARKERS:
            if line.startswith(mk):
                marker = mk
                line = line[len(mk):].strip()
                break
        if not line:                       # 只有标记
            if marker not in out:
                out.append(marker)
            continue

        n = _norm_line(line)
        if not n or n in seen:
            continue
        if any(n == s or (len(n) >= 6 and n in s) for s in seed_norms):
            if marker and marker not in out:   # 标题是重复的，但保留分区标记
                out.append(marker)
            continue
        seen.add(n)
        out.append(f"{marker} {line}".strip() if marker else line)

    # 只剩标记、没有实际内容时，把标记也去掉
    if all(l in BODY_MARKERS for l in out):
        return ""
    return "\n".join(out)


def _bili_bvid(url: str) -> str:
    m = re.search(r"(BV[0-9A-Za-z]{10})", url or "")
    return m.group(1) if m else ""


def _bili_dyn_id(url: str) -> str:
    """从动态链接里取 id：/opus/<数字>、t.bilibili.com/<数字>、dynamic_id=<数字>。"""
    u = url or ""
    m = (
        re.search(r"/opus/(\d+)", u)
        or re.search(r"t\.bilibili\.com/(\d+)", u)
        or re.search(r"[?&]dynamic_id=(\d+)", u)
    )
    return m.group(1) if m else ""


def _fmt_duration(sec: Any) -> str:
    """秒数 → 32:42 / 1:05:09；拿不到或为 0 返回空串。"""
    try:
        total = int(sec)
    except (TypeError, ValueError):
        return ""
    if total <= 0:
        return ""
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_pubdate(ts: Any) -> str:
    """发布时间戳 → 2026-11-07 01:20（按北京时间，容器默认是 UTC）。"""
    try:
        from datetime import datetime, timedelta, timezone

        return datetime.fromtimestamp(
            int(ts), timezone(timedelta(hours=8))
        ).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError):
        return ""


def _fmt_view(count: Any) -> str:
    """播放数 → 46.9万 / 9832。"""
    if not isinstance(count, int) or count <= 0:
        return ""
    return f"{count / 10000:.1f}万" if count >= 10000 else str(count)


BILI_DYN_API = "https://api.bilibili.com/x/polymer/web-dynamic/v1/detail?id="


async def fetch_bili_dynamic(url: str) -> tuple[str, list[str]]:
    """B 站动态（图文 / 专栏 / 转发）→ (正文, 配图列表)。

    动态不是视频，没有 BV 号，`view` 接口用不上。这里走动态详情接口：
        短链 → 302 到 m.bilibili.com/opus/<id> 或 t.bilibili.com/<id>
        → /x/polymer/web-dynamic/v1/detail?id=<id>
    公开动态返回 code=0；被删/仅粉丝可见会返回 4101152「动态不可见」。
    """
    headers = {"User-Agent": BILI_UA, "Referer": "https://www.bilibili.com/"}
    try:
        import aiohttp

        jar = aiohttp.CookieJar(unsafe=True)
        async with aiohttp.ClientSession(
            headers=headers, cookie_jar=jar, trust_env=True
        ) as session:
            dyn_id = _bili_dyn_id(url)
            if not dyn_id:
                # 短链：不跟随跳转，从 Location 里取 opus / dynamic id
                async with session.get(
                    url, allow_redirects=False, timeout=aiohttp.ClientTimeout(total=8)
                ) as resp:
                    loc = resp.headers.get("Location", "") or ""
                dyn_id = _bili_dyn_id(loc)
            if not dyn_id:
                return "", []
            try:
                async with session.get(
                    "https://www.bilibili.com/", timeout=aiohttp.ClientTimeout(total=8)
                ) as warm:
                    await warm.read()
            except Exception:
                pass
            async with session.get(
                BILI_DYN_API + dyn_id, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                data = await resp.json(content_type=None)
    except Exception as exc:
        logger.info("[card_reader] B站动态接口失败（%s）: %s", type(exc).__name__, url)
        return "", []

    code = (data or {}).get("code")
    if code != 0:
        logger.info(
            "[card_reader] B站动态不可用（code=%s %s）: %s",
            code, str((data or {}).get("message"))[:30], url,
        )
        return "", []

    item = ((data.get("data") or {}).get("item")) or {}
    modules = item.get("modules") or {}
    author = modules.get("module_author") or {}
    dyn = modules.get("module_dynamic") or {}
    major = dyn.get("major") or {}
    stat = modules.get("module_stat") or {}

    # 正文：普通动态在 desc.text；专栏/opus 在 opus.summary 或 article 里
    desc_text = ((dyn.get("desc") or {}).get("text") or "").strip()
    opus = major.get("opus") or {}
    article = major.get("article") or {}
    title = (opus.get("title") or article.get("title") or "").strip()
    if not desc_text:
        desc_text = ((opus.get("summary") or {}).get("text") or "").strip()
    if not desc_text:
        desc_text = str(article.get("desc") or "").strip()

    # 配图：draw.items[].src / opus.pics[].url
    images: list[str] = []
    for it in (major.get("draw") or {}).get("items") or []:
        u = str((it or {}).get("src") or "").strip()
        if u:
            images.append(u)
    for it in opus.get("pics") or []:
        u = str((it or {}).get("url") or "").strip()
        if u:
            images.append(u)
    images = [("https://" + u[len("http://"):]) if u.startswith("http://") else u
              for u in images if u.startswith("http")]

    if not (title or desc_text or images):
        return "", []

    parts = ["[网页正文] " + (title or "B站动态")]
    name = str(author.get("name") or "").strip()
    if name:
        parts.append("UP主：" + name)

    meta_bits: list[str] = []
    pub = _fmt_pubdate(author.get("pub_ts"))
    if pub:
        meta_bits.append(pub + " 发布")
    for label, key in (("赞", "like"), ("评论", "comment"), ("转发", "forward")):
        cnt = ((stat.get(key) or {}).get("count"))
        if isinstance(cnt, int) and cnt > 0:
            meta_bits.append(_fmt_view(cnt) + label)
    if meta_bits:
        parts.append(" ｜ ".join(meta_bits))

    if desc_text:
        parts.append(desc_text)
    if images:
        parts.append(f"[配图 {len(images)} 张]")

    logger.info(
        "[card_reader] B站动态抓到 %d 字，配图 %d 张（%s）",
        len(desc_text), len(images), dyn.get("major", {}).get("type") or item.get("type"),
    )
    return "\n".join(p for p in parts if p).strip(), images[:MAX_IMAGES]


async def fetch_bili_meta(url: str) -> tuple[str, str]:
    """B 站：只走公开 API，返回（正文, 封面图 url）。

    B 站页面本身有反爬，直连会返回「出错啦」页面；但公开 API 只要先访问一次首页
    拿到 buvid cookie 就能调。相比 DeepSeek 渲染整页：快 20 倍，没有界面文案噪音，
    播放量也是准确的。

    封面用 API 的 pic（B 站原图），调用方拿不到时再退回卡片自带的 preview。
    """
    headers = {"User-Agent": BILI_UA, "Referer": "https://www.bilibili.com/"}
    try:
        import aiohttp

        jar = aiohttp.CookieJar(unsafe=True)
        async with aiohttp.ClientSession(
            headers=headers, cookie_jar=jar, trust_env=True
        ) as session:
            bvid = _bili_bvid(url)
            if not bvid:
                # b23 短链：不跟随跳转，只从 Location 里读 BV 号
                async with session.get(
                    url, allow_redirects=False, timeout=aiohttp.ClientTimeout(total=8)
                ) as resp:
                    loc = resp.headers.get("Location", "") or ""
                bvid = _bili_bvid(loc)
            if not bvid:
                logger.info("[card_reader] B站：链接里没有 BV 号，按动态处理 %s", url)
                return "", ""
            try:
                async with session.get(
                    "https://www.bilibili.com/", timeout=aiohttp.ClientTimeout(total=8)
                ) as warm:
                    await warm.read()          # 预热拿 buvid cookie，没有它会返回「出错啦」
            except Exception:
                pass
            async with session.get(
                BILI_API + bvid, timeout=aiohttp.ClientTimeout(total=8)
            ) as resp:
                data = await resp.json(content_type=None)
    except Exception as exc:
        logger.info("[card_reader] B站接口失败（%s）: %s", type(exc).__name__, url)
        return "", ""

    if not isinstance(data, dict) or data.get("code") != 0:
        logger.info("[card_reader] B站接口返回异常: %s", str(data)[:120])
        return "", ""

    d = data.get("data") or {}
    title = str(d.get("title") or "").strip()
    desc = str(d.get("desc") or "").strip()
    owner = str(((d.get("owner") or {}).get("name")) or "").strip()
    view = (d.get("stat") or {}).get("view")
    tname = str(d.get("tname") or "").strip()

    # 封面：pic 是 B 站原图，http 统一升到 https
    cover = str(d.get("pic") or "").strip()
    if cover.startswith("http://"):
        cover = "https://" + cover[len("http://"):]
    if not cover.startswith("https://"):
        cover = ""

    if not (title or desc):
        return "", cover

    parts = ["[网页正文] " + (title or "")]
    if owner:
        parts.append("UP主：" + owner)
    if tname:
        parts.append("分区：" + tname)

    meta_bits: list[str] = []
    dur = _fmt_duration(d.get("duration"))
    if dur:
        meta_bits.append("时长 " + dur)
    pub = _fmt_pubdate(d.get("pubdate"))
    if pub:
        meta_bits.append(pub + " 发布")
    views = _fmt_view(view)
    if views:
        meta_bits.append(views + "播放")
    danmaku = _fmt_view((d.get("stat") or {}).get("danmaku"))
    if danmaku:
        meta_bits.append(danmaku + "弹幕")
    if meta_bits:
        parts.append(" ｜ ".join(meta_bits))

    if desc:
        parts.append(desc)
    return "\n".join(p for p in parts if p).strip(), cover


def _body_covers(desc: str, body: str) -> bool:
    """卡片摘要是不是已经被正文完整覆盖了。

    QQ 给的摘要通常是截断版（末尾带 …），正文才是全文。只要摘要开头那句能在
    正文里找到，就没必要再显示一份残缺的摘要 —— 否则同一段出现两遍。
    比较前先按 # 切开，避免「前半句 + 话题标签」因为标签写法不同而匹配不上。
    """
    d = _norm_line(desc).rstrip("…")
    b = _norm_line(body)
    if not d or not b:
        return False
    head = d.split("#")[0].strip("。，、,.!！?？:：;；|｜")
    probe = head[:40] if len(head) >= 8 else d[:40]
    return len(probe) >= 8 and probe in b


def _who(event: AstrMessageEvent) -> str:
    """日志用：这条消息来自哪个聊天、哪个人。

    群聊显示「群<群号>」，私聊显示「私聊」；后面跟发送者昵称和 QQ 号。
    加这个是为了排查时能分清「同一条卡片在群里和私聊各发了一次」
    和「同一条消息被处理了两次」——以前这两种情况在日志里长得一模一样。
    """
    gid = ""
    try:
        gid = str(getattr(event.message_obj, "group_id", "") or "")
    except Exception:
        pass
    chat = ("群" + gid) if gid else "私聊"

    name = ""
    sid = ""
    try:
        sid = str(event.get_sender_id() or "")
    except Exception:
        pass
    try:
        raw_name = event.get_sender_name()
        name = str(raw_name) if raw_name else ""
    except Exception:
        pass

    person = f"{name}({sid})" if name else f"({sid})"
    return f"{chat} / {person}"


def _cache_key(url: str) -> str:
    """缓存 key 用内容 id，而不是完整 URL。

    同一条小红书笔记换个 xsec_token 就是另一条 URL，按 URL 缓存等于没缓存；
    B 站短链也有各种分享参数。取不到 id 才退回完整 URL。
    """
    try:
        nid = _xhs_note_id(url)
        if nid:
            return "xhs:" + nid
        bv = _bili_bvid(url)
        if bv:
            return "bili:" + bv
    except Exception:
        pass
    return url


async def fetch_page(url: str, key: str, timeout: int = FETCH_TIMEOUT) -> str:
    """调用 DeepSeek fetch 接口抓网页正文；失败返回空串，绝不抛异常。

    B 站这类页面要服务端渲染，首次偶尔会超过 10 秒。超时后**再试一次**
    （同样 10 秒）——实测第二次基本是命中对方缓存，几毫秒就回来。
    超时不重试会白丢一次正文；返回 404/403 则不重试，那种重试也没用。
    """
    now = time.time()
    ckey = _cache_key(url)
    cached = _CACHE.get(ckey)
    if cached and now - cached[0] < CACHE_TTL:
        return cached[1]

    data = None
    for attempt in range(1 + FETCH_RETRY):
        t0 = time.time()
        try:
            import aiohttp

            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.post(
                    "https://api.deepseek.com/v1/tools/fetch",
                    json={"url": url},
                    headers={
                        "Authorization": "Bearer " + key,
                        "Content-Type": "application/json",
                    },
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status != 200:
                        logger.info("[card_reader] 抓取返回 %s（不重试）: %s", resp.status, url)
                        return ""
                    data = await resp.json()
            break
        except (asyncio.TimeoutError, TimeoutError):
            logger.info(
                "[card_reader] 抓取超时 %.1fs（第 %d/%d 次）: %s",
                time.time() - t0, attempt + 1, 1 + FETCH_RETRY, url,
            )
        except Exception as exc:
            # 注意：超时类异常的 str() 是空的，只打 exc 会得到「抓取失败 url: 」
            logger.info(
                "[card_reader] 抓取失败（%s，第 %d/%d 次）: %s",
                type(exc).__name__, attempt + 1, 1 + FETCH_RETRY, url,
            )
    if data is None:
        return ""

    content = str(data.get("content") or "").strip()
    if not content:
        return ""

    title = str(data.get("title") or "").strip()
    head = "[网页正文] " + (title + "\n" if title else "")
    text = (head + _clip(content, FETCH_MAX_CHARS)).strip()

    # 写入缓存，超过上限就淘汰最早的一条（长跑不吃内存）
    if len(_CACHE) >= CACHE_MAX:
        try:
            oldest = min(_CACHE.items(), key=lambda kv: kv[1][0])[0]
            _CACHE.pop(oldest, None)
        except Exception:
            _CACHE.clear()
    _CACHE[ckey] = (now, text, [])
    return text


def _is_xhs(url: str) -> bool:
    try:
        from urllib.parse import urlparse

        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return any(host == h or host.endswith("." + h) for h in XHS_HOSTS)


def _pick_from_xhs_html(html: str) -> str:
    """从小红书 SSR 页面里抠出笔记正文。

    页面把数据塞在 window.__INITIAL_STATE__ 里，但里面含 undefined/NaN，
    不是合法 JSON，所以先用正则粗取，取不到再尝试修补后解析。
    """
    title = ""
    desc = ""
    if not html:
        return ""

    # 方式一：直接正则在页面里找 desc / title（最稳，不依赖 JSON 合法性）
    m = re.search(r'"desc"\s*:\s*"((?:[^"\\]|\\.){1,4000})"', html)
    if m:
        try:
            desc = json.loads('"' + m.group(1) + '"')
        except Exception:
            desc = m.group(1)

    m2 = re.search(r'"title"\s*:\s*"((?:[^"\\]|\\.){1,300})"', html)
    if m2:
        try:
            title = json.loads('"' + m2.group(1) + '"')
        except Exception:
            title = m2.group(1)

    # 方式二：修 undefined 之后整体解析
    if not desc:
        m3 = re.search(r"window\.__INITIAL_STATE__\s*=\s*(\{.*?\})\s*</script>", html, re.S)
        if m3:
            raw = m3.group(1).replace("undefined", "null")
            try:
                data = json.loads(raw)
                fields: dict[str, str] = {}
                _walk(data, fields)
                title = title or fields.get("title", "")
                desc = fields.get("desc", "")
            except Exception:
                pass

    if not (title or desc):
        return ""

    lines = ["[小红书正文]"]
    if title and title not in desc:
        lines.append("标题：" + title.strip())
    if desc:
        lines.append(desc.strip())
    return _clip("\n".join(lines).strip(), FETCH_MAX_CHARS)


def _xhs_note_id(url: str) -> str:
    """从 URL 里取笔记 ID，用于校验抓回来的页面是不是同一篇。"""
    m = re.search(r"/(?:explore|discovery/item|item)/([0-9a-fA-F]{16,})", url or "")
    return m.group(1) if m else ""


def _xhs_token(url: str) -> str:
    """从卡片 URL 里取 xsec_token（授权凭证）。"""
    m = re.search(r"[?&]xsec_token=([^&#]+)", url or "")
    return m.group(1) if m else ""


def _xhs_img_key(url: str) -> str:
    """取图片的「文件名」当去重键。

    同一张图在 xhscdn 上有多个镜像地址：目录里的 hash 不一样，结尾文件名一样，例如
        /202611182305/514baad8.../1040g008327n41hn26i00
        /202611182305/5fb0e170.../1040g008327n41hn26i00
    按完整 URL 去重会把同一张图当成两张，按结尾文件名才不会重复。
    """
    try:
        from urllib.parse import urlparse

        name = (urlparse(url).path or "").rsplit("/", 1)[-1]
    except Exception:
        name = (url or "").rsplit("/", 1)[-1]
    return name.split("!", 1)[0].lower()


def _xhs_images(html: str, limit: int = 9) -> list[str]:
    """从 SSR 页面里抽笔记配图（按笔记顺序，最多 limit 张）。

    页面里每张图会同时出现 urlDefault / url 等多个字段，同一张图还有不同 CDN 镜像，
    所以按 _xhs_img_key() 的文件名去重，否则一张图会被当成两三张塞给模型。

    页面里的图片地址是转义过的，形如：
        "url":"http:\\u002F\\u002Fsns-webpic-qc.xhscdn.com\\u002F2026...!nd_dft_wlteh_webp_3"
    所以要先还原转义，再统一升到 https。
    """
    urls: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r'"(?:urlDefault|url)"\s*:\s*"([^"]{20,600})"', html):
        u = m.group(1).replace("\\u002F", "/").replace("\\/", "/")
        if not u.startswith("http"):
            continue
        if u.startswith("http://"):
            u = "https://" + u[len("http://"):]
        if not any(k in u for k in ("sns-webpic", "ci.xiaohongshu", "sns-img")):
            continue
        if any(k in u for k in ("avatar", "user/profile", "undefined")):
            continue
        key = _xhs_img_key(u) or u
        if key in seen:
            continue
        seen.add(key)
        urls.append(u)
        if len(urls) >= limit:
            break
    return urls


# 小红书长连接：cookie 预热一次就复用，不必每张卡片都重新访问首页
_XHS_STATE: dict[str, Any] = {"session": None, "warmed": False}


def _xhs_reset() -> None:
    """丢掉当前复用的小红书会话（出错或事件循环换了的时候）。"""
    sess = _XHS_STATE.get("session")
    _XHS_STATE["session"] = None
    _XHS_STATE["warmed"] = False
    if sess is not None:
        try:
            asyncio.create_task(sess.close())
        except Exception:
            pass


async def _xhs_session():
    """拿到（必要时新建）小红书会话，并确保已经预热过 cookie。

    没有 a1 cookie 时笔记页必然 404，所以第一次要访问一下首页。预热成功后
    一直复用同一个会话；预热失败则下次再试。
    """
    import aiohttp

    sess = _XHS_STATE.get("session")
    if sess is None or sess.closed:
        sess = aiohttp.ClientSession(
            headers={
                "User-Agent": XHS_UA,
                "Referer": "https://www.xiaohongshu.com/",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh-Hans;q=0.9",
            },
            cookie_jar=aiohttp.CookieJar(unsafe=True),
            trust_env=True,
        )
        _XHS_STATE["session"] = sess
        _XHS_STATE["warmed"] = False

    if not _XHS_STATE["warmed"]:
        jar = sess.cookie_jar
        for path in ("/", "/explore"):
            if len(jar) > 0:
                break
            try:
                async with sess.get(
                    "https://www.xiaohongshu.com" + path,
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as warm:
                    await warm.read()
            except Exception as exc:
                logger.info(
                    "[card_reader] 小红书预热失败(%s): %s", path, type(exc).__name__
                )
        _XHS_STATE["warmed"] = len(jar) > 0
        logger.info("[card_reader] 小红书预热 cookie 数: %s", len(jar))
    return sess


async def fetch_xhs(url: str) -> tuple[str, list[str]]:
    """抓小红书笔记正文和多图。

    实测要点（缺一不可）：
    1. 必须用 /discovery/item/<id> 路径，/explore/ 会 302 跳走
    2. 必须先访问首页拿到 a1 cookie，否则笔记页直接返回 404
    3. 手机 UA + Referer
    4. URL 只用 id + xsec_token 重建，卡片里那串杂乱参数带进来反而容易触发 404
    """
    now = time.time()
    ckey = _cache_key(url)
    cached = _CACHE.get(ckey)
    if cached and now - cached[0] < CACHE_TTL:
        return cached[1], list(cached[2])

    note_id = _xhs_note_id(url)
    token = _xhs_token(url)
    if not note_id or not token:
        logger.info("[card_reader] 小红书链接缺少 id 或 token，跳过")
        return "", []

    req_url = (
        "https://www.xiaohongshu.com/discovery/item/%s?xsec_token=%s&xsec_source=app_share"
        % (note_id, token)
    )

    try:
        import aiohttp

        session = await _xhs_session()
        async with session.get(
            req_url, timeout=aiohttp.ClientTimeout(total=XHS_TIMEOUT)
        ) as resp:
            status = resp.status
            final_url = str(resp.url)
            html = await resp.text()
    except Exception as exc:
        logger.info("[card_reader] 小红书抓取失败（%s）: %s", type(exc).__name__, exc)
        _xhs_reset()
        return "", []

    if status != 200:
        logger.info("[card_reader] 小红书返回 %s", status)
        return "", []

    # 笔记不可见时会跳 /404/，页面里塞的是"推荐笔记"，必须挡掉
    if "/404/" in final_url or "error_code=" in final_url:
        logger.info("[card_reader] 小红书跳转到 404 页（笔记不可见或 token 失效）")
        return "", []

    if note_id not in html:
        logger.info("[card_reader] 页面里没有目标笔记 %s", note_id)
        return "", []

    text = _pick_from_xhs_html(html)
    if not text:
        logger.info("[card_reader] 小红书页面 %d 字节，未解析出正文", len(html))
        return "", []

    images = _xhs_images(html, MAX_IMAGES)
    logger.info("[card_reader] 小红书正文抓到 %d 字，配图 %d 张", len(text), len(images))
    if len(_CACHE) >= CACHE_MAX:
        try:
            oldest = min(_CACHE.items(), key=lambda kv: kv[1][0])[0]
            _CACHE.pop(oldest, None)
        except Exception:
            _CACHE.clear()
    _CACHE[ckey] = (now, text, images)
    return text, images


def _apply_config(cfg: dict) -> None:
    """把插件配置（_conf_schema.json）覆盖到模块级常量上。

    好处是面板里改完即时生效，不用改代码；没配的项保持当前默认值。
    """
    global FETCH_ENABLED, FETCH_TIMEOUT, FETCH_RETRY, FETCH_MAX_CHARS
    global FETCH_GROUPS, CACHE_TTL, CACHE_MAX, SKIP_FETCH_HOSTS
    global ATTACH_IMAGE, MAX_IMAGES, MAX_FIELD_LEN, DUMP_RAW

    def as_bool(v):
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in ("1", "true", "yes", "on")

    def as_int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    def as_list(v):
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str):
            return [x.strip() for x in v.replace("，", ",").split(",") if x.strip()]
        return []

    if "fetch_enabled" in cfg:
        FETCH_ENABLED = as_bool(cfg["fetch_enabled"])
    if "fetch_timeout" in cfg:
        n = as_int(cfg["fetch_timeout"])
        if n and n > 0:
            FETCH_TIMEOUT = n
    if "fetch_retry" in cfg:
        n = as_int(cfg["fetch_retry"])
        if n is not None and 0 <= n <= 3:
            FETCH_RETRY = n
    if "fetch_max_chars" in cfg:
        n = as_int(cfg["fetch_max_chars"])
        if n is not None and n >= 0:
            FETCH_MAX_CHARS = n
    if "fetch_groups" in cfg:
        FETCH_GROUPS = set(as_list(cfg["fetch_groups"]))
    if "cache_ttl" in cfg:
        n = as_int(cfg["cache_ttl"])
        if n and n > 0:
            CACHE_TTL = n
    if "cache_max" in cfg:
        n = as_int(cfg["cache_max"])
        if n and n > 0:
            CACHE_MAX = n
    if "skip_fetch_hosts" in cfg:
        hosts = as_list(cfg["skip_fetch_hosts"])
        if hosts:
            SKIP_FETCH_HOSTS = tuple(hosts)
    if "attach_image" in cfg:
        ATTACH_IMAGE = as_bool(cfg["attach_image"])
    if "max_images" in cfg:
        n = as_int(cfg["max_images"])
        if n is not None and 0 <= n <= 20:
            MAX_IMAGES = n
    if "max_field_len" in cfg:
        n = as_int(cfg["max_field_len"])
        if n is not None and n >= 0:
            MAX_FIELD_LEN = n
    if "dump_raw" in cfg:
        DUMP_RAW = as_bool(cfg["dump_raw"])


@register("card_reader", "local", "卡片转文字 + 抓正文 + 保留配图", "1.2.0")
class CardReader(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.config = config or {}
        _apply_config(self.config)

    async def terminate(self) -> None:
        """插件被卸载/重载时，关掉复用的那个小红书会话，避免连接泄漏。"""
        sess = _XHS_STATE.get("session")
        _XHS_STATE["session"] = None
        _XHS_STATE["warmed"] = False
        if sess is not None:
            try:
                await sess.close()
            except Exception:
                pass

    def _deepseek_key(self) -> str:
        try:
            cfg = self.context.get_config() or {}
            keys = (cfg.get("provider_settings") or {}).get("websearch_deepseek_key") or []
            return keys[0] if keys else ""
        except Exception:
            return ""

    def _group_allowed(self, event: AstrMessageEvent) -> bool:
        if not FETCH_GROUPS:
            return True
        gid = str(getattr(event.message_obj, "group_id", "") or "")
        return gid in FETCH_GROUPS

    async def _convert(self, comp: Json, event: AstrMessageEvent) -> list:
        """把一张卡片转换成 [Plain, Image?] 组件列表。"""
        card_data = getattr(comp, "data", None)
        if DUMP_RAW:
            try:
                raw = json.dumps(card_data, ensure_ascii=False)
            except Exception:
                raw = str(card_data)
            logger.info("[card_reader] 原始卡片: %s", raw[:700])

        text = card_to_text(card_data)
        if not text:
            return []

        sink = _candidates(card_data)
        url = extract_url(card_data, sink)
        body = ""
        note_images: list[str] = []
        bili_cover = ""            # B 站 API 的原图封面，优先于卡片预览图
        dyn_images: list[str] = []  # B 站动态的配图（动态没有封面概念，直接多图）
        if FETCH_ENABLED and url and self._group_allowed(event):
            if _is_xhs(url):
                body, note_images = await fetch_xhs(url)   # 小红书：正文 + 多图
            elif _is_bili(url):
                # B 站：只走公开 API。实测 DeepSeek 渲染一页要 10 秒，还带一堆
                # 界面文案和错的播放量；API 0.5 秒就能拿到标题/UP主/完整简介/播放量
                body, bili_cover = await fetch_bili_meta(url)
                if not body:
                    # 没有 BV 号 = 不是视频，可能是动态 / 专栏 / opus
                    body, dyn_images = await fetch_bili_dynamic(url)
            elif not _skip_host(url):
                key = self._deepseek_key()
                if key:
                    body = await fetch_page(url, key)      # 其它站点走 DeepSeek
            if body:
                # 正文里和卡片标题重复的行去掉（B 站标题在正文里又出现一次）。
                # 注意：不拿卡片「摘要」当去重依据 —— QQ 的摘要是截断版，拿它去比
                # 会把正文里完整的那句当成重复删掉（小红书就踩过这个坑）。
                title_val = _pick(card_data, sink, ("title",))
                seeds = [
                    title_val,
                    _pick(card_data, sink, ("prompt",)),
                    _pick(card_data, sink, ("tag",)),
                ]
                raw_body = body
                body = _dedupe_body(body, seeds)
                if body:
                    desc_val = _pick(card_data, sink, ("desc", "summary"))
                    # 卡片摘要已被正文覆盖时就不再显示，避免同一段出现两遍且残缺
                    text = card_to_text(
                        card_data, include_desc=not _body_covers(desc_val, body)
                    )
                    text = text + "\n" + body
                logger.info(
                    "[card_reader] 正文 %d 字（去重后 %d 字）[%s] %s",
                    len(raw_body), len(body), _who(event), url[:100],
                )

        out: list = [Plain(text=text)]

        if ATTACH_IMAGE and MAX_IMAGES > 0:
            # 优先用笔记里的图（多图笔记能带出真正的内容图），没有才退回封面。
            # 笔记图已经按图片 id 去重过，而卡片封面就是笔记第一张，所以有笔记图时
            # 不再追加封面——否则同一张图会出现两遍。
            # B 站卡片：封面优先用 API 的 pic（原图），拿不到才用卡片 preview（QQ 缩略图）。
            # 配图优先级：小红书笔记图 > B 站动态配图 > B 站视频封面 > 卡片 preview
            candidates = note_images[:MAX_IMAGES] if note_images else []
            if not candidates and dyn_images:
                candidates = dyn_images[:MAX_IMAGES]
            if not candidates:
                cover = bili_cover or extract_image(card_data, sink)
                if cover:
                    candidates = [cover]
            for img in candidates:
                try:
                    out.append(Image(file=img))
                    logger.info("[card_reader] 附带配图: %s", img[:100])
                except Exception as exc:
                    logger.info("[card_reader] 配图组件创建失败: %s", exc)

        return out

    async def _rewrite_quote(self, event: AstrMessageEvent, reply) -> bool:
        """把「引用」里那条卡片也转成文本。

        QQ 引用卡片时，适配器把原始消息整条取回来放进 Reply.chain，里面是 Json 组件；
        而 AstrBot 的引用解析只认文字，Json 会被解析成空 → 模型看到空的 <Quoted Message>。
        这里把 Reply.chain 里的 Json 换成 Plain（+Image），并同步 message_str。
        """
        try:
            inner = getattr(reply, "chain", None) or []
            if not isinstance(inner, list) or not any(isinstance(c, Json) for c in inner):
                return False

            new_inner: list = []
            changed = False
            for comp in inner:
                if not isinstance(comp, Json):
                    new_inner.append(comp)
                    continue
                converted = await self._convert(comp, event)
                if converted:
                    new_inner.extend(converted)
                    changed = True
                else:
                    new_inner.append(comp)
            if not changed:
                return False

            reply.chain = new_inner
            rebuilt = " ".join(
                c.text for c in new_inner
                if isinstance(c, Plain) and getattr(c, "text", None)
            ).strip()
            if rebuilt:
                try:
                    reply.message_str = rebuilt
                except Exception:
                    pass
            logger.info(
                "[card_reader] 引用里的卡片已转文本（%d 字 · %s）[%s]",
                len(rebuilt), len([c for c in new_inner if isinstance(c, Image)]),
                _who(event),
            )
            return True
        except Exception as exc:
            logger.info("[card_reader] 处理引用卡片失败: %s", exc)
            return False

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def rewrite_cards(self, event: AstrMessageEvent):
        """把消息链里的 Json 段（QQ 分享卡片）改写成纯文本，必要时附带配图。

        除了当前消息里的卡片，也会处理「引用」里那张卡片（Reply.chain 里的 Json）。
        """
        try:
            message_obj = event.message_obj
            chain = getattr(message_obj, "message", None)
            if not chain:
                return

            # 先处理引用：引用卡片时当前消息里通常没有 Json，只有 Reply
            for comp in chain:
                if isinstance(comp, Reply):
                    await self._rewrite_quote(event, comp)

            if not any(isinstance(c, Json) for c in chain):
                return  # 普通消息（或只有引用），不处理

            who = _who(event)
            logger.info("[card_reader] 收到卡片 [%s]", who)

            new_chain: list = []
            changed = False
            for comp in chain:
                if not isinstance(comp, Json):
                    new_chain.append(comp)
                    continue
                converted = await self._convert(comp, event)
                if converted:
                    new_chain.extend(converted)
                    changed = True
                else:
                    new_chain.append(comp)  # 解析不出内容就原样保留

            if not changed:
                return

            message_obj.message = new_chain
            rebuilt = " ".join(
                c.text for c in new_chain if isinstance(c, Plain) and getattr(c, "text", None)
            ).strip()
            try:
                message_obj.message_str = rebuilt
            except Exception:
                pass
            try:
                event.message_str = rebuilt
            except Exception:
                pass

            imgs = [c.file for c in new_chain if isinstance(c, Image) and getattr(c, "file", None)]
            logger.info(
                "[card_reader] 卡片已转文本（配图 %d 张 · %s）: %s",
                len(imgs), who, rebuilt[:240],
            )

            # 不再回写 platform_message_history：实测平台写库用的是改写后的事件链，
            # 历史里本来就是正确文本；私聊那张表根本不记录。回写只会白等 2.4 秒
            # 并可能误改旧行，已整体删除（0.7.0）。
        except Exception as exc:  # 不让插件错误影响正常收发
            logger.error("[card_reader] 处理卡片失败: %s", exc)
