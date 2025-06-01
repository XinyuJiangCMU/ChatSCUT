import main_page as mp
import os



while True:
    ch_m = input("1.进入贴吧主页后选择吧爬取\n2.输入吧名爬取\n3.指定帖子的链接爬取评论\n4.使用lxml解析库(默认为python自带的html.parser速度较慢)\n(输入数字选择，其他无效，输入q退出)")
    if ch_m == "q":
        break
    if ch_m in ["1", "2", "3", "4"]:
        fun_dic = {
            "1": mp.main,
            "2": mp.self_define_tb_name,
            "3": mp.self_define_tz_url,
            "4": mp.use_lxmlm
        }
        stat = fun_dic[ch_m]()
        if not stat:
            break
    else:
        os.system("clear")
        print("输入无效,重新选择\n\n\n")

