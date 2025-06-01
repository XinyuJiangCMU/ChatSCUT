from bs4 import BeautifulSoup
import re
import os
import importlib.util
from urllib.parse import unquote, quote


from spcore import GetSourceCode, get_tz_main, use_lxml

html_parser="html.parser"



def use_lxmlm():
    global html_parser
    if importlib.util.find_spec("lxml") is not None:
        html_parser="lxml"
        use_lxml()
    else:
        os.system("clear")
        print("\033[31m=======你没有安装lxml库=========\033[0m")
        return 1
    os.system("clear")
    print("\033[32m=========正在使用lxml解析库=========\033[0m")
    return 1
    
    
    
def get_btz_info(html_page):
    tz_dic = []
    soup = BeautifulSoup(html_page, html_parser)
    tz_info = soup.select(" .j_threadlist_li_right")

    i = 0
    for item in tz_info:
        purl = item.select(" .j_th_tit")[0].a["href"]
        title = item.select(" .j_th_tit")[0].a["title"]
        author = item.select(" .tb_icon_author")[0]["title"]
        i += 1
        lastest_replay_time = item.select(" .threadlist_reply_date")
        if lastest_replay_time == []:
            lastest_replay_time = "无"
        else:
            lastest_replay_time = lastest_replay_time[0].text
        topic = item.select(" .threadlist_abs")
        if topic == []:
            topic = "无"
        else:
            topic = topic[0].text
        create_time = item.select(" .is_show_create_time")[0].text
        tz_dic.append([title, purl, author, lastest_replay_time, topic, create_time, i])
    return tz_dic




def show_ba_tz(html_page):
    tz_l = get_btz_info(html_page)
    os.system("clear")
    for tz in tz_l:
        print(
            f"{tz[-1]}.标题:{tz[0]}\n内容:'{tz[-3]}'\n{tz[2]}\n最后回复时间:{tz[3]}\n发帖时间:{tz[-2]}\n\n\n"
        )
    return 0


def self_define_tb_name(bname=None):
    if bname:
        bname = quote(bname)
    else:
        bname = quote(input("输入你要进入的贴吧名字"))
    host_url = f"https://tieba.baidu.com/f?kw={bname}&ie=utf-8"
    pn = 0

    burl = f"{host_url}&pn={pn}"

    html_page = GetSourceCode(burl)
    show_ba_tz(html_page)

    def ch_tz():
        head_url = "https://tieba.baidu.com"
        burl = f"{host_url}&pn={pn}"
        ch_urls = input("请输入你要选择下载的帖子(用空格分割，例如1 2 3,可以单选,输入all下载全部(时间较久)):\n")
        tz_list = get_btz_info(html_page)
        tz_url_list = []
        for href_url in tz_list:
            tz_url_list.append(f"{head_url}{href_url[1]}")
        if ch_urls == "all":
            for tz_url in tz_url_list:
                get_tz_main(tz_url)
            os._exit(0)
        ch_urls = ch_urls.split(" ")
        ch_urls = set(ch_urls)
        ch_urls = list(ch_urls)
        length_urls = len(tz_url_list)
        for url_tz_index in ch_urls:
            try:
                url_tz_index = int(url_tz_index)
                if url_tz_index > length_urls:
                    print(f"{url_tz_index}这个选项超出范围,跳过")
                    continue
                elif url_tz_index < 0:
                    print(f"{url_tz_index}这个选项超出范围,跳过")
                    continue
                tz_url = tz_url_list[url_tz_index - 1]
                get_tz_main(tz_url)
            except ValueError:
                print("有错误的选项，我希望你是不小心的，而不是故意的。")
        print("爬取完成,已退出")
        os._exit(0)

    def re_show():
        show_ba_tz(html_page)

    def before_page():
        nonlocal pn
        nonlocal html_page
        if pn > 0:
            pn -= 50
            burl = f"{host_url}&pn={pn}"
            html_page = GetSourceCode(burl)
            show_ba_tz(html_page)
        else:
            print("这是第一页😰别闹")

    def next_page():
        nonlocal pn
        nonlocal html_page
        pn += 50
        burl = f"{host_url}&pn={pn}"
        html_page = GetSourceCode(burl)
        show_ba_tz(html_page)

    def your_page():
        nonlocal pn
        nonlocal html_page
        ypn = input("请输入你要跳转的页数:")
        try:
            ypn = int(ypn)
            if ypn < 1:
                print("别玩了😢")
                return 0

            pn = (ypn - 1) * 50
            burl = f"{host_url}&pn={pn}"
            html_page = GetSourceCode(burl)
            show_ba_tz(html_page)
        except:
            print("别闹😡😡😡,返回上一级了")

    ch_dic = {
        "1": ch_tz,
        "2": before_page,
        "3": next_page,
        "4": your_page,
        "5": re_show,
    }
    while True:
        if pn == 0:
            print("========当前在第1页=========")
        else:
            print(f"=========当前在第{int(pn/50+1)}页===========")
        ch = input(
            "请输入你的选择:\n1.输入序号下载对应帖子(如果要选择这个先输入1回车后再输入帖子序号回车)\n2.上一页\n3.下一页\n4.跳转页数\n5.重新显示帖子序号(请输入序号,输入q退出)\n"
        )
        if ch == "q":
            break
        if ch in ["1", "2", "3", "4", "5"]:
            ch_dic[ch]()
        else:
            os.system("clear")
            print("输入选项有误")

    return 0


def self_define_tz_url():
    tb_host = "https://tieba.baidu.com/p/"

    tz_url = input("输入要爬取的帖子链接")
    if re.findall(tb_host, tz_url) == []:
        print("输入链接有误，返回主菜单")
    else:
        if "?share=" in tz_url:
            tz_url = re.findall("(.*?)\?share=", tz_url)[0]
    get_tz_main(tz_url)
    return 0


def get_main_page():
    data_tieba_info = {}
    html_page = GetSourceCode("https://tieba.baidu.com/index.html")
    soup = BeautifulSoup(html_page, html_parser)
    a = soup.select(".rcmd_forum_list > .clearfix .rcmd_forum_item")
    n = 1
    for i in a:
        data_tieba_info[i.div.select("div")[0].text] = [
            i.div.select("div")[1].text,
            i.div.select("div")[2].text,
            i.a["href"],
            n,
        ]
        n += 1
    return data_tieba_info


def main():
    dic_main_page = get_main_page()

    for name in dic_main_page:
        index = dic_main_page[name][-1]
        sub_num = dic_main_page[name][0]
        tz_num = dic_main_page[name][1]
        p_url = dic_main_page[name][2]
        print(f"{index}.{name}\n关注人数:{sub_num}\n帖子数{tz_num}")

    length_b = len(dic_main_page)
    while 1:
        try:
            ch = input("请输入你的选择(输入q退出程序)")
            if ch == "q":
                break
            ch = int(ch)
            if ch > length_b:
                print("超出选择范围")
                continue
            else:
                break
        except:
            print("请输入正确的数字")
    if ch == "q":
        exit()
    for i in dic_main_page:
        if dic_main_page[i][-1] == ch:
            os.system("clear")
            print(f"你选择的是{i}\n正在进入{i}\n\n\n")
            html_page = GetSourceCode(f"https://tieba.baidu.com{dic_main_page[i][2]}")
            self_define_tb_name(bname=i)

    return 0
