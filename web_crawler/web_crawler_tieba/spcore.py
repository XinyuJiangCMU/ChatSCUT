import requests
import os
import time
from bs4 import BeautifulSoup
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re

html_parser = "html.parser"

def use_lxml():
    global html_parser
    html_parser = "lxml"

def create_session_with_retries(retries=3, backoff_factor=0.3, status_forcelist=None):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist or [429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def GetSourceCode(url, SetReferer="", setcookie="", remove_note=True, timeout=15):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36"
    }
    if SetReferer:
        headers["Referer"] = SetReferer

    session = create_session_with_retries()
    try:
        response = session.get(url=url, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.content
        if remove_note:
            data = data.decode("utf8").replace("<!--", "").replace("-->", "")
            soup = BeautifulSoup(data, html_parser)
            if soup.select("title")[0].text == "百度安全验证":
                print("触发百度安全验证，请更换网络或者稍后再试\n已退出")
                os._exit(1)
            return data
        return data.decode("utf8")
    except requests.exceptions.RequestException as e:
        print(f"请求错误：{e}")
        return ""

def sanitize_filename(filename, default_ext="jpg"):
    """清理非法文件名字符并确保文件扩展名正确"""
    # 分离文件名和扩展名，去掉 URL 查询参数
    filename = filename.split("?")[0]
    base, ext = os.path.splitext(filename)
    if not ext:
        ext = f".{default_ext}"  # 默认扩展名
    sanitized_filename = re.sub(r'[\\/*?:"<>|]', '', base) + ext
    return sanitized_filename

def download_image(url, folder, filename):
    """下载图片并确保正确扩展名"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36"
    }
    session = create_session_with_retries()
    sanitized_filename = sanitize_filename(filename)
    filepath = os.path.join(folder, sanitized_filename)

    try:
        response = session.get(url=url, headers=headers, timeout=15, stream=True)
        response.raise_for_status()

        # 确保完整流式写入
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(1024):
                if chunk:
                    f.write(chunk)

        return sanitized_filename
    except requests.exceptions.RequestException as e:
        print(f"图片下载错误：{e}")
        return None

def get_reply_comments(tid, pid, fid, fwp, img_folder):
    set_time = int(time.time() * 1000)
    api_url = f"https://tieba.baidu.com/p/comment?tid={tid}&pid={pid}&pn=1&fid={fid}&t={set_time}"

    fwp.write("======================此楼回复==========================\n")
    page_source = GetSourceCode(url=api_url, remove_note=False)

    if not page_source:
        fwp.write("无法获取此楼回复。\n")
        return

    soup = BeautifulSoup(page_source, html_parser)
    pn_total = json.loads(soup.select(".lzl_li_pager_s")[0]["data-field"])["total_page"]

    pn_total = int(pn_total)

    for num_pn in range(1, pn_total + 1):
        print(f"楼爬取评论{num_pn}页")
        set_time = int(time.time() * 1000)
        next_url = f"https://tieba.baidu.com/p/comment?tid={tid}&pid={pid}&pn={num_pn}&fid={fid}&t={set_time}"
        page_source = GetSourceCode(url=next_url, SetReferer=api_url, remove_note=False)
        api_url = next_url

        if not page_source:
            fwp.write("无法获取此楼回复。\n")
            break

        soup = BeautifulSoup(page_source, html_parser)
        for mt in soup.select(" .lzl_single_post .lzl_cnt"):
            fwp.write(mt.text)

            if mt.img:
                for img_url in mt.select(" img"):
                    img_link = img_url["src"]
                    img_filename = os.path.basename(img_link)
                    downloaded_img = download_image(img_link, img_folder, img_filename)
                    if downloaded_img:
                        fwp.write(f"图片链接:{downloaded_img}\n")
                    else:
                        fwp.write(f"图片链接:{img_link}\n")
            fwp.write("\n\n")
    fwp.write("====================================================\n\n\n\n\n")

def get_tz_text(soup, fwp, img_folder):
    for i in soup.select(" .l_post_bright"):
        main_text = i.select(" .d_post_content")[0].text
        img_comm = i.select(" .d_post_content")[0].select("img")
        if i.select(" .post-tail-wrap") == []:
            other_info = "无"
        else:
            other_info = (
                i.select(" .post-tail-wrap")[0].text.replace("回复", "").replace("收起", "")
            )
        floorer_nickname = i.select(" .d_name >a")[0].text

        if [] == img_comm:
            fwp.write(
                f"昵称:{floorer_nickname}\n回复内容:{main_text}\n其他信息:{other_info}\n\n\n\n"
            )
        else:
            fwp.write(f"昵称:{floorer_nickname}\n回复内容:{main_text}\n")
            for img_url in img_comm:
                img_link = img_url["src"]
                img_filename = os.path.basename(img_link)
                downloaded_img = download_image(img_link, img_folder, img_filename)
                if downloaded_img:
                    fwp.write(f"图片链接:{downloaded_img}\n")
                else:
                    fwp.write(f"图片链接:{img_link}\n")
            fwp.write(f"其他信息:{other_info}\n\n\n\n")

        if i.select(" .lzl_link_unfold") == []:
            continue
        if i.select(" .lzl_link_unfold")[0].text == "回复":
            continue
        else:
            param_list = json.loads(i["data-field"])["content"]
            pid = param_list["post_id"]
            fid = param_list["forum_id"]
            tid = param_list["thread_id"]
            get_reply_comments(tid=tid, pid=pid, fid=fid, fwp=fwp, img_folder=img_folder)

def get_tz_main(url):
    html_page = GetSourceCode(url)

    if not html_page:
        print(f"无法获取帖子内容：{url}")
        return

    turl = "https://tieba.baidu.com/p/"

    soup = BeautifulSoup(html_page, html_parser)

    title = soup.select(" title")[0].text
    b_name = soup.find_all("meta", {"fname": True})[0]["fname"]
    if not os.path.exists(b_name):
        os.makedirs(b_name)
    txt_filename = url.replace(turl, "") + ".txt"
    img_folder = os.path.join(b_name, url.replace(turl, ""))
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    fn = os.path.join(b_name, txt_filename)
    with open(fn, "w", encoding="utf8") as fwriter:
        fwriter.write(f"标题:{title}\n\n\n\n")
        max_page = soup.select(" .l_reply_num")[0].select("span")[-1].text
        max_page = int(max_page)
        print(
            f"该帖子共{max_page}页,正在爬取{url}的1页,预计剩余时间{(max_page-1)/2}分钟(不定时的楼层评论爬取无法计入，理论时间更长)"
        )
        get_tz_text(soup=soup, fwp=fwriter, img_folder=img_folder)

        if max_page > 1:
            for num in range(2, max_page + 1):
                print("暂停30秒,防止反爬虫检测")
                time.sleep(30)

                purl = url + f"?pn={num}"
                referer_url = url + f"?pn={num-1}"
                html_page = GetSourceCode(purl, referer_url)
                soup = BeautifulSoup(html_page, html_parser)
                if soup.select("title")[0].text == "百度安全验证":
                    print(f"触发百度安全验证，请稍后再试\n{url}")
                    os.system("touch 触发安全验证")
                    break
                print(
                    f"该帖子共{max_page}页,正在爬取{url}的{num}页,预计剩余时间{(max_page-num)/2}分钟(不定时的楼层评论爬取无法计入，理论时间更长)"
                )
                get_tz_text(soup=soup, fwp=fwriter, img_folder=img_folder)

if __name__ == "__main__":
    url_l = [
        "https://tieba.baidu.com/p/8569726272",
        "https://tieba.baidu.com/p/8137879530",
        "https://tieba.baidu.com/p/8476580991",
        "https://tieba.baidu.com/p/8762118814",
        "https://tieba.baidu.com/p/8304931193",
        "https://tieba.baidu.com/p/8623766425",
        "https://tieba.baidu.com/p/8744566859",
        "https://tieba.baidu.com/p/8562470169",
        "https://tieba.baidu.com/p/8564715153",
    ]
    for u_url in url_l:
        get_tz_main(u_url)
