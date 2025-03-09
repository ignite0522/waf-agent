import re

class Waf_Check(object):
    def __init__(self):
        self.regXSS = r'(prompt|alert|confirm|expression])' \
                      r'|(javascript|script|eval)' \
                      r'|(onload|onerror|onfocus|onclick|ontoggle|onmousemove|ondrag)' \
                      r'|(String.fromCharCode)' \
                      r'|(;base64,)' \
                      r'|(onblur=write)' \
                      r'|(xlink:href)' \
                      r'|(color=)'

    def check_xss(self, str):
        flag = False
        if re.search(self.regXSS, str, re.IGNORECASE):
            flag = True
        return flag

#测试
if __name__ == '__main__':
    waf_check = Waf_Check()
    print(waf_check.check_xss('alert(1);'))
