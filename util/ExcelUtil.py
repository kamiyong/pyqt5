import pandas as pd
import xlrd
import os
import time

excel_path = os.getcwd() + "\\resource\\test_pics\\test.xlsx"


def xlrd_read():
    excel = xlrd.open_workbook(excel_path)
    table = excel.sheet_by_name("Sheet1")

    print(table.name, table.nrows, table.ncols, table.number)

    print(table.cell(4, 3).value)

    container = []

    for t in range(5):
        print(t)

    for i in range(table.nrows):
        item = []
        for j in range(table.ncols):
            item.append(str(table.cell(i, j).value))
            container.append(item)

    print(container)


def pandas_read(test_code):

    """
        @param test_code 测试代码
        @return 返回根据测试代码找到Excel的一行
                如果没找到则返回None
    """
    table = pd.read_excel(excel_path, header=0)
    h, w = table.shape
    print(w, h)
    for i in range(h + 1):

        #读取: 读取的是每行第一列
        code = str(table.iloc[i, 0])
        # print([code, table.iloc[i, 1], table.iloc[i, 2], table.iloc[i, 3]])
        # print(code)
        if test_code == code:
            # 返回整行数据
            #       测试代码   文件名称        类型               应输出结果
            return [code, table.iloc[i, 1], table.iloc[i, 2], table.iloc[i, 3]]

    return None


aw5 = os.getcwd() + "\\resource\\excel\\AW5.xls"
aw5_table = pd.read_excel(aw5, header=0)


def read_station(CODE, NAMEE):
    """
    根据iAUTO查询返回路由信息中的关键字在表中寻找对应的工站名
    这是寻找的两个关键字
     :param CODE
     :param NAMEE
     :return 找到返回具体值，找不到返回None
    """
    h, w = aw5_table.shape
    # print("Excel size: [", w, ",", h, "]")
    for i in range(h + 1):
        row = aw5_table.iloc
        # 第i行 第2列
        _code_ = row[i, 2]
        # 第i行 第4列
        _name_e_ = row[i, 4]
        # print("code: ", _code_)
        # print("name_e: ", _name_e_)
        if CODE == _code_ and _name_e_ == NAMEE:
            return row[i, 3]
    return None


# example
def example():
    start = time.time()
    print(read_station("BA", "Check comp3"))
    cost = int(((time.time() - start) * 1000))
    print("Time cost to read: ", cost, "ms")


# example()

