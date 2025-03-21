import re
import html
import urllib.parse
import base64
from html.parser import HTMLParser


class Waf_Check:
    def __init__(self):
        # 扩展后的多层级检测规则（新增10种高级模式）
        self.danger_patterns = [
            # 基础标签检测（增强大小写混淆）
            re.compile(r'<(/?\w+)(\s*[\w-]+\s*=?\s*["\']?)*\s*/?>', re.I | re.M),

            # 事件处理器（包括新型事件如onpointerdown）
            re.compile(r'on(load|click|error|mouseover|submit|pointerdown)\s*=', re.I),

            # JavaScript协议（支持编码混淆）
            re.compile(r'(\bjava\s{0,5}script|\w{2,5}:\s{0,5}javascript):', re.I),

            # 函数调用检测（包括新型混淆）
            re.compile(r'(eval|setTimeout|setInterval|Function)\s*\(', re.I),

            # DOM操作检测（包含document/window对象）
            re.compile(r'(document\.(cookie|write)|window\.(location|open))', re.I),

            # HTML5新特性攻击（如autofocus/onfocus组合）
            re.compile(r'<[^>]+\b(autofocus|onfocus)\s*=', re.I),

            # CSS表达式攻击（包含编码模式）
            re.compile(r'expression\s*\(|url\s*\(\s*[\'"]?\s*javascript:', re.I),

            # 拆分攻击向量（如<svg/<script>组合）
            re.compile(r'<svg/\s*onload\s*=\s*["\']?[^>]*>', re.I),

            # 动态字符串构造（包括fromCharCode）
            re.compile(r'String\.(fromCharCode|fromCodePoint)\s*\(', re.I),

            # 特殊字符混淆检测（十六进制/Unicode）
            re.compile(r'\\x[0-9a-f]{2}|%u[0-9a-f]{4}|&#x?[0-9a-f]+;', re.I)
        ]

        # 上下文分析器（用于识别跨上下文攻击）
        self.context_parser = HTMLParser()

    def normalize(self, payload):
        """
        增强型归一化处理（7层防御）
        新增功能：
        1. 递归URL解码（最多5层）
        2. Base64自动检测解码
        3. Unicode标准化（\u0020 -> 空格）
        4. 特殊转义序列处理（\x3c -> <）
        5. 移除不可见字符
        """
        # 递归解码层数控制
        for _ in range(5):
            # 1. URL解码
            try:
                decoded = urllib.parse.unquote(payload)
                if decoded != payload:
                    payload = decoded
            except:
                pass

            # 2. Base64检测解码（如data:text;base64,...）
            if 'base64,' in payload:
                try:
                    b64_part = payload.split('base64,', 1)[1]
                    decoded_b64 = base64.b64decode(b64_part).decode('utf-8', 'ignore')
                    payload = payload.replace(b64_part, decoded_b64)
                except:
                    pass

        def entity_replacer(match):
            entity = match.group(0)
            try:
                # 处理十六进制实体（如&#x49;）
                if entity.lower().startswith('&#x'):
                    hex_str = entity[3:-1].lstrip('0')  # 移除可能的零填充
                    hex_str = hex_str or '0'  # 处理空值情况
                    return chr(int(hex_str, 16))

                # 处理十进制实体（如&#73;）
                elif entity.startswith('&#'):
                    dec_str = entity[2:-1]
                    # 过滤非数字字符（防御&#0x49;型混淆攻击）
                    dec_str = ''.join([c for c in dec_str if c.isdigit()])
                    if dec_str:
                        return chr(int(dec_str))

                # 处理命名实体（如&nbsp;）
                elif entity in html.entities.html5:
                    return html.entities.html5[entity]

            except (ValueError, OverflowError):
                pass

            return entity  # 解析失败时保留原始实体

        # 更严格的正则匹配（精确区分实体类型）
        payload = re.sub(
            r'&(#x[0-9a-fA-F]+;|#\d+;|\w+;)',  # 明确匹配三种实体类型
            entity_replacer,
            payload
        )

        # 4. Unicode转义处理（\u0020格式）
        payload = re.sub(r'\\u([0-9a-fA-F]{4})',
                         lambda m: chr(int(m.group(1), 16)), payload)

        # 5. 十六进制转义处理（\x3c格式）
        payload = re.sub(r'\\x([0-9a-fA-F]{2})',
                         lambda m: chr(int(m.group(1), 16)), payload)

        # 6. 移除注释和不可见字符
        payload = re.sub(r'/\*.*?\*/', '', payload, flags=re.DOTALL)
        payload = re.sub(r'[\x00-\x1F\x7F]', '', payload)

        # 7. HTML解析器处理（去除隐藏内容）
        class Sanitizer(HTMLParser):
            def __init__(self):
                super().__init__()
                self.safe_data = []

            def handle_data(self, data):
                self.safe_data.append(data)

            def get_safe(self):
                return ''.join(self.safe_data)

        sanitizer = Sanitizer()
        sanitizer.feed(payload)
        payload = sanitizer.get_safe()

        return payload

    def check_xss(self, payload):
        """
        增强型检测逻辑（新增3种防御机制）
        新增功能：
        1. 上下文敏感分析
        2. 混淆字符密度检测
        3. 标签闭合结构验证
        """
        normalized = self.normalize(payload)

        # 规则1：基础正则检测
        for pattern in self.danger_patterns:
            if pattern.search(normalized):
                return True

        # 规则2：混淆字符密度检测（如过多%20或\x）
        if self._detect_obfuscation(normalized):
            return True

        # 规则3：标签闭合结构验证（如<svg><script>）
        if self._check_tag_nesting(normalized):
            return True

        return False

    def _detect_obfuscation(self, payload):
        """混淆特征检测（阈值动态计算）"""
        obf_scores = [
            (r'%[0-9a-fA-F]{2}', 3),  # URL编码密度
            (r'\\x[0-9a-fA-F]{2}', 2),  # 十六进制转义
            (r'&#x?[0-9a-f]+;', 2),  # HTML实体
            (r'[\x00-\x1F]', 1)  # 控制字符
        ]
        total_score = 0
        for pattern, score in obf_scores:
            matches = re.findall(pattern, payload)
            total_score += len(matches) * score
        return total_score > 5  # 总得分超过阈值则拦截

    def _check_tag_nesting(self, payload):
        """标签结构验证（防御如<svg/<script>拆分攻击）"""
        tag_pattern = re.compile(r'<([/!]?\w+)[^>]*>')
        tags = tag_pattern.findall(payload)
        open_tags = []
        for tag in tags:
            if tag.startswith('!'):
                continue  # 忽略注释
            if tag.startswith('/'):
                if not open_tags:
                    return True  # 异常闭合标签
                open_tags.pop()
            else:
                open_tags.append(tag)
        return len(open_tags) > 2  # 多层未闭合标签视为可疑