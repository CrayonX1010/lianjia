import requests
import time  # 导入 time 模块

headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0',
    'referer': 'https://cs.lianjia.com/ershoufang/',
    'accept-language': 'zh-CN,zh;q=0.9',
    'cookie': 'select_city=430100; lianjia_uuid=da14690a-fbe4-4219-bf16-c20465fd22d3; _jzqckmp=1; sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%221969fc28b4c9d7-09cd30a66cb3b28-4c657b58-1327104-1969fc28b4d297a%22%2C%22%24device_id%22%3A%221969fc28b4c9d7-09cd30a66cb3b28-4c657b58-1327104-1969fc28b4d297a%22%2C%22props%22%3A%7B%22%24latest_traffic_source_type%22%3A%22%E7%9B%B4%E6%8E%A5%E6%B5%81%E9%87%8F%22%2C%22%24latest_referrer%22%3A%22%22%2C%22%24latest_referrer_host%22%3A%22%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC_%E7%9B%B4%E6%8E%A5%E6%89%93%E5%BC%80%22%7D%7D; _ga=GA1.2.1161941373.1746437061; _gid=GA1.2.1023882359.1746437061; _jzqx=1.1746448691.1746448691.1.jzqsr=cs%2Elianjia%2Ecom|jzqct=/ershoufang/.-; crosSdkDT2019DeviceId=x3buir-bohvfg-3gyiywdwvpd0nxr-g6bka17gs; login_ucid=2000000481382078; lianjia_token=2.001436209e411e03cc059b09afa0d8dc92; lianjia_token_secure=2.001436209e411e03cc059b09afa0d8dc92; security_ticket=prVrkyGG2K2fIjdpWxYABt5hZT6INwpHHZ6qsgjiksX61GqLKVs33PdtCmqMJQRZz3gg0k5IpPgCGjOyh2kBfXYeNr1ZfPjWOKdX6jm/CpOk1OR4I0Xg8EIVEaxiHTvd4qE0iuTgKk5a/mNcVwaRPJmVttY7k4CY9day720/6F8=; ftkrc_=562ab878-6130-447c-b63d-4bba54b6f023; lfrc_=74e483db-e34e-4993-9940-6e3be7253d8a; lianjia_ssid=d4a2f648-207e-454e-9737-1e1928f40a0c; Hm_lvt_46bf127ac9b856df503ec2dbf942b67e=1746437049,1746445066,1746493949; HMACCOUNT=C54C983D578C6EA3; _jzqa=1.1082967149107023400.1746437049.1746448691.1746493949.4; _jzqc=1; Hm_lpvt_46bf127ac9b856df503ec2dbf942b67e=1746494114; _jzqb=1.3.10.1746493949.1; _ga_4JBJY7Y7MX=GS2.2.s1746493959$o4$g1$t1746494125$j0$l0$h0'
}

import parsel
import re
import csv

data_list = []

for page in range(1, 101):  # 循环1到100页
    url = f'https://cs.lianjia.com/ershoufang/pg{page}/'
    print(f"正在爬取第 {page} 页: {url}") # 增加打印提示
    try:
        response = requests.get(url=url, headers=headers, timeout=10) # 增加超时设置
        response.raise_for_status() # 检查请求是否成功
        html = response.text

        # 把html(网页源代码）转换为可解析的对象
        selector = parsel.Selector(html)
        # 第一次提取，提取本页房源信息对应li标签
        lis = selector.css('.sellListContent li .info')
        print(f"本页提取到 {len(lis)} 个房源") # 增加调试信息

        if not lis: # 如果当前页没有提取到数据，可能被反爬了，跳出循环或采取其他策略
            print(f"警告：第 {page} 页未能提取到房源信息，可能触发反爬机制。")
            # 可以选择 break 或者 continue
            # break # 如果希望停止爬取
            time.sleep(5) # 尝试等待更长时间
            continue # 跳过当前页，继续下一页

        for li in lis:
            title = li.css('.title a::text').get()  # 标题
            href = li.css('.title a::attr(href)').get()  # 详情页链接
            totalprice = li.css(' .priceInfo .totalPrice span::text').get()  # 售价
            unitprice = li.css('.priceInfo .unitPrice::attr(data-price)').get()  # 单价
            positionInfo = li.css('.flood .positionInfo a::text').getall()
            community = positionInfo[0] if len(positionInfo) > 0 else ''
            area = positionInfo[1] if len(positionInfo) > 1 else ''
            houseInfo = li.css('.address .houseInfo::text').get()
            houseInfo = houseInfo.split(' | ') if houseInfo else []

            if len(houseInfo) == 6:
                year = 'unknown'
            elif len(houseInfo) > 5:
                year = houseInfo[5]  # 年份
            else:
                year = ''
            houseType = houseInfo[0] if len(houseInfo) > 0 else ''
            houseArea = houseInfo[1] if len(houseInfo) > 1 else ''
            face = houseInfo[2] if len(houseInfo) > 2 else ''
            decoration = houseInfo[3] if len(houseInfo) > 3 else ''
            floor = houseInfo[4] if len(houseInfo) > 4 else ''
            floor_1 = floor[0] if floor else ''
            floor_num = int(re.search(r'\d+', floor).group()) if floor and re.search(r'\d+', floor) else None

            building = houseInfo[-1] if houseInfo else ''

            dit = {
                '标题': title,
                '售价': totalprice,
                '单价': unitprice,
                '小区': community,
                '区域': area,
                '户型': houseType,
                '面积': houseArea,
                '朝向': face,
                '装修': decoration,
                '楼层': floor,
                '楼层类型': floor_1,
                '楼层数': floor_num,
                '建筑结构': building,
                '年份': year,
                '详情页': href
            }
            data_list.append(dit)
            # print(dit) # 减少打印，避免过多输出

    except requests.exceptions.RequestException as e:
        print(f"请求第 {page} 页时发生错误: {e}")
        time.sleep(5) # 请求失败时也等待一段时间
        continue # 继续尝试下一页

    # 在每次页面请求成功并处理后，暂停2秒
    time.sleep(2)

# 确保 data_list 不是空列表
if data_list:
    # 获取字典的键作为 CSV 文件的列名
    fieldnames = data_list[0].keys()
    try:
        # 打开 CSV 文件并写入数据
        with open('houses.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # 写入表头
            writer.writeheader()
            # 写入数据行
            for row in data_list:
                writer.writerow(row)
        print("数据已成功写入 houses.csv 文件。")
    except Exception as e:
        print(f"写入文件时出现错误: {e}")
else:
    print("数据列表为空，没有数据可写入。")