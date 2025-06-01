import requests

# text = "在华工拿一等奖学金的条件是什么？"
# text = "未来技术学院师生有几篇论文被AAAI 2024收录？"
# text = "在华工拿一等奖学金的条件是什么？"
# text = "你有蜘蛛侠的资源吗？"
text = "逻辑与思维选修课怎么样"
# text = "介绍下未来技术学院舒琳老师？"
# text = "我想报xx老师的学生研究项目（srp），请问有什么方向推荐吗？"
# text = "给我一些微积分的学习资料？"


#url是你接口的地址
url = ""
ans = requests.post(url=url,json={"message":text})
print(ans)